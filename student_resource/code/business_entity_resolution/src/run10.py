"""Run 10 (incremental, reuses run-8/9 caches): run 9 + two extra candidate sources + name frequency.

  round 1   = run-8 candidates and stage-1 scores (unchanged, cached)
  round 2   = new pairs from
              d) records with an EMPTY address -> top-50 S1s by core-name word TF-IDF (same country)
              c) sibling expansion: each S1's top-5 round-1 candidates by stage-1 P -> their top-10
                 record neighbours (word TF-IDF on core + address, both sources, same country)
  stage 1   = run-8 model: train round-2 pairs are scored by the fold model that did not see their S1
              (out-of-fold, cache/v2_stage1_fold*.txt + v2_folds.npy); test round-2 pairs by the refit.
              Round-2 features use context (ranks among competitors) over round 1 + round 2; round-1
              pairs keep their round-1 features and scores (same rule on train and test).
  stage 2   = run 9 stack + `is_round2` + two name-frequency columns, trained on round-1 pairs with
              P >= 0.01 plus round-2 pairs with P >= R2_MIN (round-2 pairs have no blocking-pass
              scores, so stage 2 - which sees them with labels - decides how far to trust them).

Each step is a subcommand writing cache/r10_*.  Memory: round-2 feature matrices live on disk
(memmap), stage-1 scoring is chunked, and test stage 2 runs one country at a time (within-country
blocking, zero cross-country matches) to keep peak RAM <= ~12 GB.

python run10.py folds | expand train | r2feats train | s2train | expand test | r2feats test | s2test
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL  # noqa: E402
from blocking import PASSES, _vectorizer, pair_key, par_transform, split_key, topk  # noqa: E402
from data import load_split  # noqa: E402
from features import TextStats  # noqa: E402
from postprocess import Decider, tune_thresholds  # noqa: E402
from stage2_sibling import P_MIN, sibling_features  # noqa: E402
from training_scaled import load_rows  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE = os.path.join(ROOT, "cache")
D_K, C_TOP, C_NB = 50, 5, 10          # pass d top-k; sibling expansion anchors per S1 x neighbours
R2_MIN = 1e-3                          # round-2 pairs re-scored by stage 2
CHUNK = 2_000_000
PASS_COLS = ["rank_na", "sim_na", "rank_rev", "sim_rev", "rank_c4", "sim_c4", "rank_ph", "sim_ph", "rank_nm", "sim_nm"]
SMOKE = os.environ.get("R10_SMOKE") == "1"   # tiny S1 slice (s1_pos % 250 == 0), r10smoke_* files
T0 = time.time()


def log(msg):
    """Timestamped progress line."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def cp(name):
    """Cache path of a run-10 artefact."""
    return os.path.join(CACHE, f"{'r10smoke' if SMOKE else 'r10'}_{name}")


def stage1_names():
    """Feature names of the run-8 stage-1 model."""
    return lgb.Booster(model_file=os.path.join(CACHE, "v2_model_stage1.txt")).feature_name()


# ---------------------------------------------------------------------------
# round 1 (cached)
# ---------------------------------------------------------------------------
def round1(split, full=False):
    """Round-1 candidates in the order their stage-1 scores are stored, and the S1 mask.
    train: v2_oof_pairs (40 % S1s, OOF P, labels); test: sorted test_cands with the run-8 refit P.
    Column _orig = row index in the stored order (feature stores are indexed by it).  full=True
    also returns the blocking-pass columns (test only; needed to rebuild stage-1 features)."""
    if split == "train":
        c = pd.read_parquet(os.path.join(CACHE, "v2_oof_pairs.parquet"))
        p = c.p.values.astype(np.float32)
        c = c[["s1_pos", "src", "r_pos", "y"]]
        n = pd.read_parquet(os.path.join(CACHE, "train_source1.parquet"), columns=["entity_id"]).shape[0]
        mask = np.random.default_rng(PL.CFG["seed"]).random(n) < 0.40
    else:
        c = pd.read_parquet(os.path.join(CACHE, "test_cands.parquet"), columns=None if full else ["s1_pos", "src", "r_pos"])
        c = c.sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True)
        p = np.load(os.path.join(CACHE, "v2_test_prob.npy")).astype(np.float32)
        assert len(p) == len(c)
        n = pd.read_parquet(os.path.join(CACHE, "test_source1.parquet"), columns=["entity_id"]).shape[0]
        mask = np.ones(n, bool)
    c = c.assign(_orig=np.arange(len(c)))
    if SMOKE:
        mask &= (np.arange(len(mask)) % 250) == 0
        keep = mask[c.s1_pos.values]
        c, p = c[keep].reset_index(drop=True), p[keep]
    return c, p, mask


# ---------------------------------------------------------------------------
# step: round-2 candidate expansion
# ---------------------------------------------------------------------------
def expand(split, threads=14):
    """Round-2 pairs (not already in round 1) from pass d and sibling expansion c; one country at a time."""
    cols = ["country", "core", "addr"]
    s1 = pd.read_parquet(os.path.join(CACHE, f"{split}_source1.parquet"), columns=cols)
    rec = {k: pd.read_parquet(os.path.join(CACHE, f"{split}_source{k}.parquet"), columns=cols) for k in (2, 3)}
    c, p, mask = round1(split)
    cur = np.unique(pair_key(c.s1_pos.values, c.src.values, c.r_pos.values))
    a = pd.DataFrame({"s1_pos": c.s1_pos.values, "src": c.src.values, "r_pos": c.r_pos.values, "p": p})
    a = a.sort_values(["s1_pos", "p"], ascending=[True, False]).groupby("s1_pos").head(C_TOP)
    del c, p
    n2 = len(rec[2])
    allrec = pd.concat([rec[2], rec[3]], ignore_index=True)
    a_row = np.where(a.src.values == 3, n2, 0) + a.r_pos.values
    out = []
    for cty in sorted(x for x in s1.country.unique() if x):
        ms = np.flatnonzero(s1.country.values == cty)
        vec = _vectorizer(PASSES["nm"], 0, s1.core.values[ms])          # d) empty address -> name only
        T = par_transform(vec, s1.core.values[ms])
        nd = 0
        for k in (2, 3):
            mr = np.flatnonzero((rec[k].country.values == cty) & (rec[k].addr.values == ""))
            row, col, _ = topk(par_transform(vec, rec[k].core.values[mr]), T, D_K, threads)
            s1p = ms[col]
            keep = mask[s1p]
            out.append(pair_key(s1p[keep], np.full(int(keep.sum()), k), mr[row[keep]]))
            nd += int(keep.sum())
        del T, vec
        mr = np.flatnonzero(allrec.country.values == cty)                # c) sibling expansion
        text = (allrec.core.values[mr] + " " + allrec.addr.values[mr]).astype(object)
        vec = _vectorizer(PASSES["na"], 0, text)
        T = par_transform(vec, text)
        loc = np.full(len(allrec), -1, np.int64)
        loc[mr] = np.arange(len(mr))
        sel = loc[a_row] >= 0
        a_s1, a_loc = a.s1_pos.values[sel], loc[a_row[sel]]
        nc = 0
        for b in range(0, len(a_loc), 500_000):
            row, col, _ = topk(T[a_loc[b:b + 500_000]], T, C_NB + 1, threads)
            g = mr[col]
            src = np.where(g >= n2, 3, 2)
            out.append(pair_key(a_s1[b:b + 500_000][row], src, np.where(src == 3, g - n2, g)))
            nc += len(row)
        del T, vec, text
        log(f"  expand {split} {cty}: d {nd} hits, c {nc} hits from {len(a_loc)} anchors")
    new = np.setdiff1d(np.unique(np.concatenate(out)), cur)
    s1p, src, rpos = split_key(new)
    r2 = pd.DataFrame({"s1_pos": s1p, "src": src, "r_pos": rpos})
    r2.to_parquet(cp(f"{split}_r2_pairs.parquet"), index=False)
    log(f"expand {split}: {len(r2)} round-2 pairs = {len(r2) / mask.sum():.1f} per S1")


# ---------------------------------------------------------------------------
# step: round-2 stage-1 features + scores
# ---------------------------------------------------------------------------
def r2feats(split):
    """Stage-1 features for round-2 pairs (context over round 1 + round 2, per country; on-disk
    memmap) and their stage-1 P (train: out-of-fold by the S1's fold; test: run-8 refit), chunked."""
    data = load_split(os.path.join(ROOT, "dataset"), split, CACHE)
    stats = TextStats(seed=PL.CFG["seed"]).fit([data["source1"], data["source2"], data["source3"]])
    r2 = pd.read_parquet(cp(f"{split}_r2_pairs.parquet"))
    r1 = pd.read_parquet(os.path.join(CACHE, "train_cands_f0.4.parquet" if split == "train" else "test_cands.parquet"))
    if SMOKE:
        r1 = r1[(r1.s1_pos.values % 250) == 0]
    names = stage1_names()
    cty = data["source1"].country.values
    r2["_i"] = np.arange(len(r2))
    feats = np.lib.format.open_memmap(cp(f"{split}_r2_feats.npy"), mode="w+", dtype=np.float32,
                                      shape=(len(r2), len(names)))
    base2 = np.full(len(r2), np.nan, np.float32)        # union-context base score (density matching)
    for c_ in sorted(x for x in np.unique(cty) if x):
        m1 = cty[r1.s1_pos.values] == c_
        m2 = cty[r2.s1_pos.values] == c_
        u = pd.concat([r1[m1].assign(_i=-1), r2[m2].assign(**{k: np.nan for k in PASS_COLS})], ignore_index=True)
        u = PL.base_scores(u, data, stats)
        new = u[u._i.values >= 0]
        base2[new._i.values] = new.base.values
        del u
        for b in range(0, len(new), CHUNK):
            part = new.iloc[b:b + CHUNK]
            feats[part._i.values] = PL.build_features(part, data, stats)[names].values.astype(np.float32)
        del new
        log(f"  r2feats {split} {c_}: {int(m2.sum())} pairs")
    feats.flush()
    np.save(cp(f"{split}_r2_base.npy"), base2)
    if split == "train":
        fpath = os.path.join(CACHE, "v2_folds.npy")
        folds = np.load(fpath) if os.path.exists(fpath) or not SMOKE else             np.zeros(len(pd.read_parquet(os.path.join(CACHE, "v2_oof_pairs.parquet"), columns=["s1_pos"])), np.int64)
        oof = pd.read_parquet(os.path.join(CACHE, "v2_oof_pairs.parquet"), columns=["s1_pos"])
        s1_fold = pd.Series(folds, index=oof.s1_pos.values).groupby(level=0).first()
        fold_of = s1_fold.reindex(r2.s1_pos.values).values.astype(np.int64)
        fold_files = [os.path.join(CACHE, f"v2_stage1_fold{k}.txt") for k in range(5)]
        if SMOKE and not all(os.path.exists(x) for x in fold_files):
            fold_files = [os.path.join(CACHE, "v2_model_stage1.txt")] * 5      # smoke only: refit stand-in
        models = [lgb.Booster(model_file=x) for x in fold_files]
    else:
        fold_of = np.zeros(len(r2), np.int64)
        models = [lgb.Booster(model_file=os.path.join(CACHE, "v2_model_stage1.txt"))]
    p = np.empty(len(r2), np.float32)
    for b in range(0, len(r2), CHUNK):
        F = np.asarray(feats[b:b + CHUNK])
        fo = fold_of[b:b + CHUNK]
        pb = np.empty(len(F), np.float32)
        for k, m in enumerate(models):
            sel = fo == k
            if sel.any():
                pb[sel] = m.predict(F[sel])
        p[b:b + CHUNK] = pb
    np.save(cp(f"{split}_r2_p1.npy"), p)
    log(f"r2feats {split}: done; stage-1 P >= {R2_MIN} on {(p >= R2_MIN).mean():.3f} of round-2 pairs")


# ---------------------------------------------------------------------------
# stage-2 inputs on the union
# ---------------------------------------------------------------------------
def name_freq(data, d):
    """Name-frequency columns for pairs d: how many S1s of the country carry the record's exact core
    name, and how many records (S2+S3) of the country carry it (statistics of the split itself)."""
    s1 = data["source1"]
    recs = pd.concat([data["source2"][["country", "core"]], data["source3"][["country", "core"]]])
    f_s1 = s1.groupby(["country", "core"]).size()
    f_rec = recs.groupby(["country", "core"]).size()
    rcore = np.empty(len(d), object)
    for k in (2, 3):
        m = d.src.values == k
        rcore[m] = data[f"source{k}"].core.values[d.r_pos.values[m]]
    key = pd.MultiIndex.from_arrays([s1.country.values[d.s1_pos.values], rcore])
    return pd.DataFrame({"nf_s1_same_rcore": f_s1.reindex(key).fillna(0).values.astype(np.float32),
                         "nf_rec_same_rcore": f_rec.reindex(key).fillna(0).values.astype(np.float32)})


def label(r2, data):
    """Ground-truth label of training pairs r2 (s1_pos, src, r_pos)."""
    gt = pd.read_csv(os.path.join(ROOT, "dataset", "train", "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    lists = gt.matched_entity_ids.str.split(",")
    pairs = pd.DataFrame({"s1": np.repeat(gt.source1_entity_id.values, lists.map(len).values),
                          "rid": [i for l in lists for i in l]})
    pairs = pairs[pairs.rid != ""]
    pos1 = pd.Series(np.arange(len(data["source1"])), index=data["source1"].entity_id.values)
    s1p = pos1.reindex(pairs.s1).values.astype(np.int64)
    src = np.where(pairs.rid.str.startswith("S3-"), 3, 2)
    rp = np.zeros(len(pairs), np.int64)
    for k in (2, 3):
        idx = pd.Series(np.arange(len(data[f"source{k}"])), index=data[f"source{k}"].entity_id.values)
        m = src == k
        rp[m] = idx.reindex(pairs.rid.values[m]).values
    return np.isin(pair_key(r2.s1_pos.values, r2.src.values, r2.r_pos.values), pair_key(s1p, src, rp)).astype(np.int8)


def stage1_rows(split, data, g1, g2):
    """Stage-1 feature rows: g1 = sorted round-1 indices (into round1(split)), g2 = round-2 indices.
    Round 1 from the cached run-8 store (train, by _orig) or recomputed over the S1s' country round-1
    set (test: original context); round 2 from the r10 on-disk store."""
    names = stage1_names()
    X = np.empty((len(g1) + len(g2), len(names)), np.float32)
    if split == "train" and len(g1):
        with open(os.path.join(CACHE, "v2_train_feats", "_complete")) as fh:
            assert fh.read().split(",") == names
        c1, _, _ = round1(split)
        X[:len(g1)] = load_rows(os.path.join(CACHE, "v2_train_feats"), c1._orig.values[g1], len(names))
    elif len(g1):
        c1, _, _ = round1(split, full=True)
        stats = TextStats(seed=PL.CFG["seed"]).fit([data["source1"], data["source2"], data["source3"]])
        cty = data["source1"].country.values[c1.s1_pos.values]
        for c_ in sorted(np.unique(cty[g1])):
            idx = np.flatnonzero(cty == c_)
            cc = PL.base_scores(c1.iloc[idx].reset_index(drop=True), data, stats)
            want = g1[cty[g1] == c_]
            sub = cc.iloc[np.searchsorted(idx, want)]
            del cc
            for b in range(0, len(sub), CHUNK):
                X[np.searchsorted(g1, want[b:b + CHUNK])] =                     PL.build_features(sub.iloc[b:b + CHUNK], data, stats)[names].values
            log(f"  round-1 stage-1 features {split} {c_}: {len(want)} rows")
    X[len(g1):] = np.load(cp(f"{split}_r2_feats.npy"), mmap_mode="r")[g2]
    return X


def union_inputs(split, data, country=None):
    """Stage-2 rows of the union (round-1 P >= P_MIN, round-2 P >= R2_MIN) and their stage-2 matrix,
    optionally restricted to one country's S1s.  Returns (u, rows, X, names); u holds every union
    pair of the selection, rows index the re-scored ones (sorted: round 1 first, then round 2)."""
    c1, p1, _ = round1(split)
    r2 = pd.read_parquet(cp(f"{split}_r2_pairs.parquet"))
    p2 = np.load(cp(f"{split}_r2_p1.npy"))
    s1_cty = data["source1"].country.values
    i1 = np.arange(len(c1)) if country is None else np.flatnonzero(s1_cty[c1.s1_pos.values] == country)
    i2 = np.arange(len(r2)) if country is None else np.flatnonzero(s1_cty[r2.s1_pos.values] == country)
    u = pd.DataFrame({"s1_pos": np.r_[c1.s1_pos.values[i1], r2.s1_pos.values[i2]],
                      "src": np.r_[c1.src.values[i1], r2.src.values[i2]],
                      "r_pos": np.r_[c1.r_pos.values[i1], r2.r_pos.values[i2]],
                      "p": np.r_[p1[i1], p2[i2]].astype(np.float32),
                      "is_round2": np.r_[np.zeros(len(i1), np.float32), np.ones(len(i2), np.float32)]})
    if split == "train":
        u["y"] = np.r_[c1.y.values[i1], label(r2.iloc[i2], data)].astype(np.int8)
    n1 = len(i1)
    rows = np.flatnonzero(np.r_[p1[i1] >= P_MIN, p2[i2] >= R2_MIN])
    del c1, r2, p1, p2
    log(f"union {split} {country or 'all'}: {len(u)} pairs ({len(i2)} round 2); stage-2 rows {len(rows)} "
        f"({int((rows >= n1).sum())} round 2)")
    cty = s1_cty[u.s1_pos.values]
    ctxv, ctx_cols = None, None
    for c_ in sorted(x for x in np.unique(cty) if x):
        q = np.flatnonzero((cty == c_) & (u.p.values >= 1e-4))
        cq = PL.stage2_context(u.iloc[q][["s1_pos", "src", "r_pos"]].reset_index(drop=True), u.p.values[q])
        if ctxv is None:
            ctxv, ctx_cols = np.full((len(rows), cq.shape[1]), np.nan, np.float32), list(cq.columns)
        hit = np.isin(rows, q)
        ctxv[hit] = cq.values[np.searchsorted(q, rows[hit])]
        del cq
    d = u.iloc[rows].reset_index(drop=True)
    sibf = sibling_features(d, {k: data[f"source{k}"][["core", "addr", "nums"]] for k in (2, 3)},
                            data["source1"][["core", "addr", "nums"]])
    nf = name_freq(data, d)
    log("  context / sibling / name-frequency features done")
    X1 = stage1_rows(split, data, i1[rows[rows < n1]], i2[rows[rows >= n1] - n1])
    X = np.hstack([X1, ctxv, sibf.values, nf.values, d[["is_round2"]].values]).astype(np.float32)
    del X1, ctxv
    names = stage1_names() + ctx_cols + list(sibf.columns) + list(nf.columns) + ["is_round2"]
    return u, rows, X, names


# ---------------------------------------------------------------------------
# stage 2 train / test
# ---------------------------------------------------------------------------
def true_counts(data):
    """True match count per train S1."""
    gt = pd.read_csv(os.path.join(ROOT, "dataset", "train", "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    cnt = gt.matched_entity_ids.map(lambda x: len([i for i in x.split(",") if i]))
    return pd.Series(cnt.values, index=gt.source1_entity_id.values).reindex(
        data["source1"].entity_id.values).fillna(0).values.astype(np.int64)


def fast_score(u, p, tca, mask):
    """Fixed-grid (t_first, t_add) search over the masked S1s using pairs with P >= 0.01 (identical
    score: lower pairs can never be kept).  Returns (score, t_first, t_add)."""
    sub = np.flatnonzero(mask)
    sel = mask[u.s1_pos.values] & (p >= 0.01)
    code = np.full(len(mask), -1, np.int64)
    code[sub] = np.arange(len(sub))
    rid = (u.src.values[sel].astype(np.int64) << 23) | u.r_pos.values[sel]
    return tune_thresholds(code[u.s1_pos.values[sel]], rid, p[sel].astype(np.float64),
                           u.y.values[sel].astype(bool), tca[sub], len(sub), n_q=50)


def s2train():
    """Stage-2 CV on the train union (5-fold GroupKFold by S1), fixed-grid thresholds, refit, save."""
    data = load_split(os.path.join(ROOT, "dataset"), "train", CACHE)
    u, rows, X, names = union_inputs("train", data)
    y = u.y.values[rows]
    g = u.s1_pos.values[rows]
    oof = np.zeros(len(rows), np.float32)
    best = []
    for f, (tr, va) in enumerate(GroupKFold(5).split(X, y, g)):
        dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(dict(PL.CFG["lgb"], seed=PL.CFG["seed"] + 100 + f), dtr, PL.CFG["num_boost_round"],
                      valid_sets=[dva], callbacks=[lgb.early_stopping(PL.CFG["early_stopping"], verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        best.append(m.best_iteration)
        log(f"  s2 fold {f}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.5f}")
    p = u.p.values.copy()
    p[rows] = oof
    np.save(cp("train_s2_oof.npy"), p)
    tca = true_counts(data)
    u40 = np.random.default_rng(PL.CFG["seed"]).random(len(data["source1"]))
    if SMOKE:
        u40[(np.arange(len(u40)) % 250) != 0] = 1.0
    cty = data["source1"].country.values
    res = {}
    for name, m in (("full40", u40 < 0.40), ("nested15", u40 < 0.15),
                    ("India", (u40 < 0.40) & (cty == "India")), ("US", (u40 < 0.40) & (cty == "US"))):
        sc, tf, ta = fast_score(u, p, tca, m)
        res[name] = {"macro_f05": sc, "t_first": tf, "t_add": ta}
        log(f"RESULT run 10 {name}: {sc:.5f} (tf {tf:.3f} ta {ta:.3f})")
    log("  run 9 reference: full40 0.97233 / nested15 0.97232 / India 0.96246 / US 0.97871")
    sel = (u40 < 0.40)[u.s1_pos.values]
    res["recall"] = float(u.y.values[sel].sum() / tca[u40 < 0.40].sum())
    log(f"union candidate recall {res['recall']:.4f} (run 9: 0.9717); candidates/S1 {len(u) / (u40 < 0.40).sum():.1f}")
    final = lgb.train(dict(PL.CFG["lgb"], seed=PL.CFG["seed"]), lgb.Dataset(X, y, feature_name=names),
                      int(np.mean(best)))
    final.save_model(cp("stage2_model.txt"))
    res.update(best_iters=best, n_rows=int(len(rows)))
    with open(cp("train_result.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    imp = pd.Series(final.feature_importance("gain"), index=names).sort_values(ascending=False)
    log("saved r10_stage2_model.txt; top features: " + ", ".join(imp.index[:15]))


def s2test(out="runs/v4smoke" if SMOKE else "runs/v4"):
    """Stage 2 on the test union one country at a time, decision with the run-10 OOF thresholds,
    write both TSVs (candidates = the full union that was scored)."""
    with open(cp("train_result.json")) as fh:
        thr = json.load(fh)["full40"]
    data = load_split(os.path.join(ROOT, "dataset"), "test", CACHE)
    model = lgb.Booster(model_file=cp("stage2_model.txt"))
    parts = []
    for c_ in sorted(x for x in data["source1"].country.unique() if x):
        u, rows, X, names = union_inputs("test", data, c_)
        assert model.feature_name() == names, "stage-2 feature layout differs from the trained model"
        p = u.p.values.copy()
        for b in range(0, len(rows), CHUNK):
            p[rows[b:b + CHUNK]] = model.predict(X[b:b + CHUNK])
        del X
        parts.append(u[["s1_pos", "src", "r_pos"]].assign(p=p))
        log(f"  s2test {c_} scored")
    u = pd.concat(parts, ignore_index=True).sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True)
    np.save(cp("test_prob.npy"), u.p.values)
    s1_pos, src, r_pos = (u[k].values.astype(np.int64) for k in ("s1_pos", "src", "r_pos"))
    keep = Decider(s1_pos, (src << 23) | r_pos, u.p.values.astype(np.float64)).keep(thr["t_first"], thr["t_add"])
    s1 = data["source1"]
    ids = {k: np.asarray(data[f"source{k}"].entity_id.values, dtype=object) for k in (2, 3)}
    s1_ids = np.asarray(s1.entity_id.values, dtype=object)
    os.makedirs(os.path.join(ROOT, out), exist_ok=True)
    PL.write_lists(os.path.join(ROOT, out, "candidate_pairs.tsv"), s1_ids, s1_pos, src, r_pos, ids[2], ids[3],
                   "candidate_entity_ids")
    PL.write_lists(os.path.join(ROOT, out, "matching_results.tsv"), s1_ids, s1_pos[keep], src[keep], r_pos[keep],
                   ids[2], ids[3], "matched_entity_ids")
    cty = np.asarray(s1.country.values, dtype=object)
    kept = s1_pos[keep]
    summ = {"t_first": thr["t_first"], "t_add": thr["t_add"], "pred_pairs": int(keep.sum()),
            "candidates_per_s1": len(u) / len(s1),
            "pred_singleton_fraction_by_country": {k: float(1 - len(np.unique(kept[cty[kept] == k])) / (cty == k).sum())
                                                   for k in np.unique(cty)}}
    with open(os.path.join(ROOT, out, "test_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2)
    log(f"s2test: {summ}")


def main():
    """Subcommand dispatch (see module docstring for the order)."""
    step = sys.argv[1]
    if step == "folds":
        import train_fold_models
        train_fold_models.main()
    elif step == "expand":
        expand(sys.argv[2])
    elif step == "r2feats":
        r2feats(sys.argv[2])
    elif step == "s2train":
        s2train()
    elif step == "s2test":
        s2test()


if __name__ == "__main__":
    main()
