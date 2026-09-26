"""Stage 4b experiment: stacked stage-2 model with SIBLING-AGREEMENT features.

Idea: the true matches of one S1 are duplicates of each other, so a real match should agree with
the S1's other confident candidates ("siblings", stage-1 P >= SIB_P).  A "twin" distractor (same
name, shifted house number) disagrees with the sibling cluster on numbers; a name-only record
(empty address) that is a real duplicate agrees with the siblings' names.

Inputs (all cached by run 8, train only): cache/v2_oof_pairs.parquet (stage-1 OOF P, full
candidate sets of the 40 % S1 subset) and the stage-1 feature chunk store cache/v2_train_feats.
Only pairs with stage-1 P >= P_MIN are re-scored (every other pair keeps its stage-1 P; the
tuned thresholds are ~0.7, so they are never kept either way).

Reports the macro F0.5 (fixed threshold search) for stage 1 alone, stage 2 with the existing
rank/gap context only, and stage 2 with context + sibling features, on the full 40 % set and the
nested 15 % subset.  Nothing here touches the test set.
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import CFG, stage2_context  # noqa: E402
from postprocess import tune_thresholds  # noqa: E402
from training_scaled import load_rows  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE = os.path.join(ROOT, "cache")
P_MIN = 0.01     # pairs re-scored by stage 2
SIB_P = 0.5      # stage-1 P needed to count as a sibling
T0 = time.time()


def log(msg):
    """Timestamped progress line."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def _numset(s):
    """Space-separated number string -> frozenset (empty set if none)."""
    return frozenset(s.split()) if s else frozenset()


def sibling_features(d, rec, s1):
    """Stage-2 sibling-agreement features.
    d: pairs (s1_pos, src, r_pos, p) to score, with p = stage-1 P.  rec: {src: DataFrame(core,
    addr, nums)} indexed by r_pos.  s1: DataFrame(core, addr, nums) indexed by s1_pos.
    For every pair, siblings = the S1's OTHER candidates with p >= SIB_P.  Returns a float32
    DataFrame aligned with d: sibling count / P mass, and max / P-weighted mean agreement between
    this record and its siblings on core name, address and number set; NaN when there are no
    siblings or the compared field is empty on either side."""
    key = d.src.values.astype(np.int64) * 10**9 + d.r_pos.values
    txt = {}
    for k in (2, 3):
        m = d.src.values == k
        r = rec[k].iloc[d.r_pos.values[m]]
        txt.update(zip(key[m], zip(r.core.values, r.addr.values, map(_numset, r.nums.values))))
    sib = d[d.p.values >= SIB_P]
    sib_by_s1 = {}
    for s, kk, pp in zip(sib.s1_pos.values, sib.src.values.astype(np.int64) * 10**9 + sib.r_pos.values, sib.p.values):
        sib_by_s1.setdefault(s, []).append((kk, pp))
    s1_nums = dict(zip(s1.index.values, map(_numset, s1.nums.values)))
    n = len(d)
    out = {c: np.full(n, np.nan, np.float32) for c in
           ("sib_n", "sib_pmass", "sib_name_max", "sib_name_wmean", "sib_addr_max", "sib_addr_wmean",
            "sib_num_jacc_max", "sib_num_jacc_wmean", "sib_num_eq_frac", "sib_num_supported_by_s1")}
    for i, (s, kk) in enumerate(zip(d.s1_pos.values, key)):
        core, addr, nums = txt[kk]
        sibs = [(k2, p2) for k2, p2 in sib_by_s1.get(s, ()) if k2 != kk]
        out["sib_n"][i] = len(sibs)
        if not sibs:
            continue
        w = np.array([p2 for _, p2 in sibs], np.float32)
        out["sib_pmass"][i] = w.sum()
        nm, ad, nj, ne = [], [], [], []
        for k2, _ in sibs:
            c2, a2, n2 = txt[k2]
            nm.append(fuzz.token_set_ratio(core, c2) / 100 if core and c2 else np.nan)
            ad.append(fuzz.token_set_ratio(addr, a2) / 100 if addr and a2 else np.nan)
            if nums and n2:
                nj.append(len(nums & n2) / len(nums | n2))
                ne.append(float(nums == n2))
            else:
                nj.append(np.nan)
                ne.append(np.nan)
        for name, v in (("name", nm), ("addr", ad), ("num_jacc", nj)):
            v = np.array(v, np.float32)
            ok = ~np.isnan(v)
            if ok.any():
                out[f"sib_{name}_max"][i] = v[ok].max()
                out[f"sib_{name}_wmean"][i] = (v[ok] * w[ok]).sum() / w[ok].sum()
        ne = np.array(ne, np.float32)
        if (~np.isnan(ne)).any():
            out["sib_num_eq_frac"][i] = np.nanmean(ne)
        # does the S1's own number set back this record against the siblings? (twin detector)
        sn = s1_nums.get(s, frozenset())
        if nums and sn:
            out["sib_num_supported_by_s1"][i] = float(bool(nums & sn))
    return pd.DataFrame(out)


def grouped_cv(X, y, groups, seed):
    """5-fold GroupKFold LightGBM (same params as stage 1, early stopping on the held-out fold).
    Returns out-of-fold probabilities."""
    oof = np.zeros(len(y), np.float32)
    for f, (tr, va) in enumerate(GroupKFold(5).split(X, y, groups)):
        dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(dict(CFG["lgb"], seed=seed + f), dtr, CFG["num_boost_round"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(CFG["early_stopping"], verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        log(f"    fold {f}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.5f}")
    return oof


def score(c, prob, tca, u, frac):
    """Macro F0.5 with the fixed joint threshold search over the S1s with u < frac."""
    sub = np.flatnonzero(u < frac)
    sel = np.isin(c.s1_pos.values, sub)
    d = c[sel]
    code = pd.Series(np.arange(len(sub)), index=sub)
    s1c = code.reindex(d.s1_pos.values).values.astype(np.int64)
    rid = (d.src.values.astype(np.int64) << 23) | d.r_pos.values
    return tune_thresholds(s1c, rid, prob[sel].astype(np.float64), d.y.values.astype(bool), tca[sub], len(sub), n_q=50)


def build_train_inputs():
    """Rebuild the stage-2 training matrix from run-8 caches.  Returns (c, rows, X, names, y, g,
    tca, u, s1_country): c = all OOF pairs, rows = indices re-scored by stage 2, X aligned with rows."""
    c = pd.read_parquet(os.path.join(CACHE, "v2_oof_pairs.parquet"))
    p1 = c.p.values.astype(np.float32)
    s1 = pd.read_parquet(os.path.join(CACHE, "train_source1.parquet"), columns=["entity_id", "country", "core", "addr", "nums"])
    u = np.random.default_rng(CFG["seed"]).random(len(s1))
    gt = pd.read_csv(os.path.join(ROOT, "dataset", "train", "train_ground_truth.tsv"), sep="	", dtype=str,
                     keep_default_na=False)
    cnt = gt.matched_entity_ids.map(lambda x: len([i for i in x.split(",") if i]))
    tca = pd.Series(cnt.values, index=gt.source1_entity_id.values).reindex(s1.entity_id.values).fillna(0).values.astype(np.int64)
    rows = np.flatnonzero(p1 >= P_MIN)
    d = c.iloc[rows].reset_index(drop=True)
    log(f"{len(c)} pairs, {len(rows)} with stage-1 P >= {P_MIN} (pos {int(d.y.sum())} of {int(c.y.sum())})")
    rec = {k: pd.read_parquet(os.path.join(CACHE, f"train_source{k}.parquet"), columns=["core", "addr", "nums"])
           for k in (2, 3)}
    sibf = sibling_features(d, rec, s1[["core", "addr", "nums"]])
    del rec
    ctx = stage2_context(c, p1).iloc[rows].reset_index(drop=True)
    with open(os.path.join(CACHE, "v2_train_feats", "_complete")) as fh:
        names1 = fh.read().split(",")
    X1 = load_rows(os.path.join(CACHE, "v2_train_feats"), rows, len(names1))
    X = np.hstack([X1, ctx.values, sibf.values]).astype(np.float32)
    names = names1 + list(ctx.columns) + list(sibf.columns)
    log(f"stage-2 matrix {X.shape}")
    return c, rows, X, names, d.y.values.astype(np.int8), d.s1_pos.values, tca, u, s1.country.values


def refit_and_shift():
    """Shift check for the stage-2 layer (train stage 2 on one country's rows, apply to the other
    country's stage-1 OOF; compare with stage 1 alone on the same S1s, retuned and with the training
    country's thresholds), then refit the final stage-2 model on all rows and save it."""
    c, rows, X, names, y, g, tca, u, cty_s1 = build_train_inputs()
    p_s1 = c.p.values.astype(np.float32)
    p_cv = np.load(os.path.join(CACHE, "v2_stage2_ctx_sib_oof.npy"))
    cty_rows = cty_s1[g]
    n_iter = int(np.mean([589, 562, 614, 512, 581]))      # best iterations of the ctx+sib CV arm
    for held in ("India", "US"):
        other = "US" if held == "India" else "India"
        m = lgb.train(dict(CFG["lgb"], seed=CFG["seed"]), lgb.Dataset(X[cty_rows != held], y[cty_rows != held]), n_iter)
        p2 = p_s1.copy()
        va = cty_rows == held
        p2[rows[va]] = m.predict(X[va])
        _, tf1, ta1 = score_country(c, p_s1, tca, u, cty_s1, other)
        _, tf2, ta2 = score_country(c, p_cv, tca, u, cty_s1, other)
        s1_re = score_country(c, p_s1, tca, u, cty_s1, held)[0]
        s2_re = score_country(c, p2, tca, u, cty_s1, held)[0]
        s1_x = score_country(c, p_s1, tca, u, cty_s1, held, (tf1, ta1))
        s2_x = score_country(c, p2, tca, u, cty_s1, held, (tf2, ta2))
        log(f"SHIFT held-out {held}: retuned stage1 {s1_re:.5f} -> stage2(trained on {other}) {s2_re:.5f} | "
            f"with {other} thresholds: stage1 {s1_x:.5f} -> stage2 {s2_x:.5f}")
    final = lgb.train(dict(CFG["lgb"], seed=CFG["seed"]), lgb.Dataset(X, y, feature_name=names), n_iter)
    final.save_model(os.path.join(CACHE, "v3_stage2_model.txt"))
    imp = pd.Series(final.feature_importance("gain"), index=names).sort_values(ascending=False)
    log("saved cache/v3_stage2_model.txt; top features: " + ", ".join(imp.index[:12]))


def score_country(c, prob, tca, u, cty_s1, country, thresholds=None):
    """Macro F0.5 over the 40 % S1s of one country: tuned (returns (score, tf, ta)) or, with
    thresholds=(tf, ta), the fixed-threshold score."""
    from postprocess import Decider, macro_f05_arrays
    sub = np.flatnonzero((u < 0.40) & (cty_s1 == country))
    sel = np.isin(c.s1_pos.values, sub) & (prob >= 0.01)
    d = c[sel]
    code = pd.Series(np.arange(len(sub)), index=sub)
    s1c = code.reindex(d.s1_pos.values).values.astype(np.int64)
    rid = (d.src.values.astype(np.int64) << 23) | d.r_pos.values
    pr = prob[sel].astype(np.float64)
    yy = d.y.values.astype(bool)
    if thresholds is None:
        return tune_thresholds(s1c, rid, pr, yy, tca[sub], len(sub), n_q=50)
    keep = Decider(s1c, rid, pr).keep(*thresholds)
    return macro_f05_arrays(s1c, yy, keep, tca[sub], len(sub))


def main():
    """--mode refit: shift check + final stage-2 model.  (The CV comparison that produced
    EXPERIMENTS.md row 16 is recorded in runs/v2/stage2_results.log.)"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["refit"], default="refit")
    ap.parse_args()
    refit_and_shift()


if __name__ == "__main__":
    main()
