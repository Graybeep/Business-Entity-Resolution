"""Orchestration: Stage 0 diagnostics, blocking, features, Stage 4 model, Stage 5 decision, I/O.

python pipeline.py --data-dir dataset --out output [--shift-check]
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import generate_candidates  # noqa: E402
from data import load_ground_truth, load_split  # noqa: E402
from features import TextStats, add_context, pair_features, _rowdot  # noqa: E402
from postprocess import Decider, macro_f05_arrays, near_tie_rate, one_owner, tune_thresholds  # noqa: E402

CFG = {
    "seed": 42,
    "blocking": {"k": {"na": 10, "c4": 5, "ph": 5, "nm": 5}, "rev_k": 2, "threads": 14, "seed": 0},
    "train_s1_frac": 0.15,     # S1 entities used for model training / CV (memory bound)
    "n_folds": 5,
    "lgb": {"objective": "binary", "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 100,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
            "max_bin": 255, "verbose": -1, "num_threads": 14},
    "num_boost_round": 3000,
    "early_stopping": 50,
    "predict_chunk_s1": 150_000,
    "stage2": False,           # stacked 2nd stage: +0.0006 CV (noise) in ablation, so off
}

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------
def diagnostics(tr, gt, pairs):
    s1, s2, s3 = tr["source1"], tr["source2"], tr["source3"]
    sizes = gt.matched_entity_ids.map(lambda x: len([i for i in x.split(",") if i]))
    rec = pd.concat([s2[["entity_id", "country"]], s3[["entity_id", "country"]]]).set_index("entity_id").country
    c1 = s1.set_index("entity_id").country
    cross = int((c1.reindex(pairs.s1).values != rec.reindex(pairs.rid).values).sum())
    miss = {}
    for name, df in (("s1", s1), ("s2", s2), ("s3", s3)):
        miss[name] = {
            "n": int(len(df)),
            "empty_name": float((df.business_name == "").mean()),
            "empty_address": float((df.business_address == "").mean()),
            "empty_country": float((df.country == "").mean()),
            "address_no_digits": float((~df.business_address.str.contains(r"\d", regex=True)).mean()),
            "address_no_postal": float((df.postal == "").mean()),
            "countries": df.country.value_counts().to_dict(),
        }
    d = {
        "n_s1": int(len(s1)),
        "singleton_fraction": float((sizes == 0).mean()),
        "match_list_size_dist": {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        "n_true_pairs": int(len(pairs)),
        "ids_under_multiple_s1": int(pairs.rid.duplicated().sum()),
        "cross_country_true_pairs": cross,
        "missingness": miss,
    }
    log(f"diagnostics: singletons={d['singleton_fraction']:.4f} shared_ids={d['ids_under_multiple_s1']} "
        f"cross_country={cross}")
    return d


# ---------------------------------------------------------------------------
# Blocking (cached)
# ---------------------------------------------------------------------------
def get_candidates(split, data, cache_dir, s1_mask=None):
    cp = os.path.join(cache_dir, f"{split}_cands.parquet")
    if os.path.exists(cp):
        return pd.read_parquet(cp)
    log(f"blocking {split} ...")
    c = generate_candidates(data["source1"], data["source2"], data["source3"], CFG["blocking"], log=log,
                            s1_mask=s1_mask)
    c.to_parquet(cp, index=False)
    return c


def label_candidates(c, tr, pairs):
    s1_ids = tr["source1"].entity_id.values
    pos1 = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    rid_pos = {}
    for src, key in ((2, "source2"), (3, "source3")):
        rid_pos[src] = pd.Series(np.arange(len(tr[key])), index=tr[key].entity_id.values)
    src = np.where(pairs.rid.str.startswith("S3-"), 3, 2)
    tp_s1 = pos1.reindex(pairs.s1).values
    tp_r = np.where(src == 2, rid_pos[2].reindex(pairs.rid).values, rid_pos[3].reindex(pairs.rid).values)
    ok = ~(np.isnan(tp_s1) | np.isnan(tp_r))  # pairs whose S1 / record are not loaded
    src, tp_s1, tp_r = src[ok], tp_s1[ok].astype(np.int64), tp_r[ok].astype(np.int64)
    tkey = (tp_s1.astype(np.int64) << 24) | ((src == 3).astype(np.int64) << 23) | tp_r.astype(np.int64)
    ckey = (c.s1_pos.values.astype(np.int64) << 24) | ((c.src.values == 3).astype(np.int64) << 23) | c.r_pos.values
    c["y"] = np.isin(ckey, tkey).astype(np.int8)
    true_count = np.bincount(tp_s1, minlength=len(s1_ids))
    return c, true_count


def blocking_report(c, true_count, data, s1_mask=None):
    """Recall over the S1s that were queried (s1_mask); reduction ratio vs. the full
    within-country and full cross products for those S1s."""
    n_true = int(true_count.sum())
    found = int(c.y.sum())
    s1 = data["source1"] if s1_mask is None else data["source1"][s1_mask]
    total = 0
    for lab in s1.country.unique():
        n1 = (s1.country == lab).sum()
        n2 = sum(((data[k].country == lab) | (data[k].country == "")).sum() for k in ("source2", "source3"))
        total += int(n1) * int(n2)
    rep = {"n_candidate_pairs": int(len(c)), "true_pairs": n_true, "true_pairs_found": found,
           "pair_recall": found / max(n_true, 1),
           "reduction_ratio_vs_within_country": 1 - len(c) / total,
           "reduction_ratio_vs_full_cross_product": 1 - len(c) / (len(s1) * (len(data["source2"]) + len(data["source3"]))),
           "mean_candidates_per_s1": len(c) / len(s1)}
    # recall per pass at various k
    for col in [x for x in c.columns if x.startswith("rank_")]:
        r = c.loc[c.y == 1, col].values
        rep[f"recall_{col}"] = {k: float((r < k).sum() / n_true) for k in (1, 3, 5, 10, 20) if k <= 255}
    log(f"blocking: pairs={len(c)} recall={rep['pair_recall']:.4f} RR={rep['reduction_ratio_vs_within_country']:.6f} "
        f"cands/S1={rep['mean_candidates_per_s1']:.1f}")
    return rep


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _side_frames(c, data):
    s1 = data["source1"]
    L = s1.iloc[c.s1_pos.values].reset_index(drop=True)
    r2, r3 = data["source2"], data["source3"]
    is3 = c.src.values == 3
    R = pd.concat([r2.iloc[c.r_pos.values[~is3]].assign(src=2), r3.iloc[c.r_pos.values[is3]].assign(src=3)])
    order = np.concatenate([np.flatnonzero(~is3), np.flatnonzero(is3)])
    R.index = order
    R = R.sort_index().reset_index(drop=True)
    return L, R


def _base_chunk(L, R, stats):
    from features import _enc
    n = _rowdot(_enc(stats.name_char, L.core.values), _enc(stats.name_char, R.core.values))
    a = _rowdot(_enc(stats.addr_char, L.addr.values), _enc(stats.addr_char, R.addr.values))
    a[(L.addr.values == "") | (R.addr.values == "")] = np.nan
    return n.astype(np.float32), a.astype(np.float32)


def base_scores(c, data, stats, chunk=500_000, n_jobs=14):
    """Cheap exact TF-IDF cosines for every candidate pair; used for context features."""
    small = {k: v[["core", "addr"]] for k, v in data.items()}
    res = []
    block = chunk * n_jobs
    with Parallel(n_jobs=n_jobs) as par:
        for b in range(0, len(c), block):
            jobs = []
            for i in range(b, min(b + block, len(c)), chunk):
                L, R = _side_frames(c.iloc[i:i + chunk], small)
                jobs.append(delayed(_base_chunk)(L, R, stats))
            res.extend(par(jobs))
    out_n = np.concatenate([r[0] for r in res])
    out_a = np.concatenate([r[1] for r in res])
    c["base_name"] = out_n
    c["base_addr"] = out_a
    c["base"] = np.where(np.isnan(out_a), out_n, 0.5 * out_n + 0.5 * out_a).astype(np.float32)
    return add_context(c, "base", "ctx")


def _feature_chunk(L, R, stats, extra):
    import features
    features.CP_WORKERS = 1
    F = pair_features(L, R, stats)
    return pd.concat([F, extra.reset_index(drop=True)], axis=1)


def build_features(c, data, stats, chunk=200_000, n_jobs=14):
    """Pair features in parallel chunks (rows stay in the order of `c`)."""
    keep = [x for x in c.columns if x.startswith(("rank_", "sim_", "base", "ctx_", "n_cand"))]
    parts = []
    block = chunk * n_jobs * 2
    cols = ["entity_id", "name", "core", "variants", "addr", "nums", "postal", "phon", "country"]
    small = {k: v[cols] for k, v in data.items()}
    with Parallel(n_jobs=n_jobs) as par:
        for b in range(0, len(c), block):
            jobs = []
            for i in range(b, min(b + block, len(c)), chunk):
                sub = c.iloc[i:i + chunk]
                L, R = _side_frames(sub, small)
                jobs.append(delayed(_feature_chunk)(L, R, stats, sub[keep]))
            parts.extend(par(jobs))
            log(f"  features {min(b + block, len(c))}/{len(c)}")
    return pd.concat(parts, ignore_index=True)


def stage2_context(c, p):
    """Context of the stage-1 probability p: where this pair stands among its S1's candidates
    and among the S1s competing for the same record."""
    d = pd.DataFrame({"s1": c.s1_pos.values, "r": (c.src.values.astype(np.int64) << 23) | c.r_pos.values,
                      "p": p.astype(np.float32)})
    out = pd.DataFrame(index=d.index)
    out["p1"] = d.p
    for key, pre in (("s1", "s1"), ("r", "rec")):
        g = d.groupby(key).p
        mx = g.transform("max")
        out[f"p1_{pre}_max"] = mx
        out[f"p1_{pre}_gap"] = mx - d.p
        out[f"p1_{pre}_rank"] = g.rank(ascending=False, method="min").astype(np.float32)
        out[f"p1_{pre}_n_hi"] = (d.p > 0.5).groupby(d[key]).transform("sum").astype(np.float32)
        out[f"p1_{pre}_sum"] = g.transform("sum")
        # best competitor excluding this pair
        srt = d.sort_values([key, "p"], ascending=[True, False])
        first = np.r_[True, srt[key].values[1:] != srt[key].values[:-1]]
        second = np.r_[False, first[:-1]] & ~first
        top2 = pd.Series(srt.p.values[second], index=srt[key].values[second])
        t2 = top2.reindex(d[key].values).fillna(0).values
        is_top = d.p.values >= mx.values
        out[f"p1_{pre}_best_other"] = np.where(is_top, t2, mx.values).astype(np.float32)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Model + evaluation
# ---------------------------------------------------------------------------
def cross_validate(X, y, groups):
    oof = np.zeros(len(y), np.float32)
    best_iters, models = [], []
    for fold, (tr_i, va_i) in enumerate(GroupKFold(CFG["n_folds"]).split(X, y, groups)):
        params = dict(CFG["lgb"], seed=CFG["seed"] + fold)
        dtr = lgb.Dataset(X.iloc[tr_i], y[tr_i], free_raw_data=True)
        dva = lgb.Dataset(X.iloc[va_i], y[va_i], reference=dtr)
        m = lgb.train(params, dtr, CFG["num_boost_round"], valid_sets=[dva],
                      callbacks=[lgb.early_stopping(CFG["early_stopping"], verbose=False)])
        oof[va_i] = m.predict(X.iloc[va_i], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration)
        models.append(m)
        log(f"  fold {fold}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.5f}")
    return oof, best_iters, models


def evaluate(c, prob, true_count, s1_subset):
    """Tune (t_first, t_add) and score over ALL S1 in s1_subset (no-candidate S1s included)."""
    code = pd.Series(np.arange(len(s1_subset)), index=s1_subset)
    s1c = code.reindex(c.s1_pos.values).values.astype(np.int64)
    rid = (c.src.values.astype(np.int64) << 23) | c.r_pos.values
    tc = true_count[s1_subset]
    sc, tf, ta = tune_thresholds(s1c, rid, prob, c.y.values.astype(bool), tc, len(s1_subset))
    return sc, tf, ta, s1c, rid


def score_fixed(c, prob, true_count, s1_subset, tf, ta):
    code = pd.Series(np.arange(len(s1_subset)), index=s1_subset)
    s1c = code.reindex(c.s1_pos.values).values.astype(np.int64)
    rid = (c.src.values.astype(np.int64) << 23) | c.r_pos.values
    keep = Decider(s1c, rid, prob).keep(tf, ta)
    return macro_f05_arrays(s1c, c.y.values.astype(bool), keep, true_count[s1_subset], len(s1_subset))


def shift_check(X, y, c, true_count, s1_country):
    res = {}
    cty = s1_country[c.s1_pos.values]
    labels = [l for l in np.unique(s1_country) if l]
    params = dict(CFG["lgb"], seed=CFG["seed"])
    tuned = {}
    for held in labels:
        tr_m = cty != held
        va_m = cty == held
        m = lgb.train(params, lgb.Dataset(X[tr_m], y[tr_m]), CFG["mean_best_iter"])
        p_va = m.predict(X[va_m]).astype(np.float32)
        # thresholds tuned on the TRAINING countries (their own OOF-free in-sample preds are
        # optimistic, so use the main OOF thresholds as the "other country" setting)
        s1_sub = np.unique(c.s1_pos.values[va_m])
        s1_sub = np.union1d(s1_sub, np.flatnonzero((s1_country == held) & CFG["_train_s1_mask"]))
        cv = c[va_m]
        sc_retuned, tf, ta, _, _ = evaluate(cv, p_va, true_count, s1_sub)
        sc_main = score_fixed(cv, p_va, true_count, s1_sub, CFG["_t_first"], CFG["_t_add"])
        tuned[held] = (tf, ta)
        res[held] = {"retuned": sc_retuned, "t_first": tf, "t_add": ta, "with_main_oof_thresholds": sc_main}
        log(f"  shift: held-out {held}: retuned={sc_retuned:.4f} (tf={tf:.3f}, ta={ta:.3f}) "
            f"main-thresholds={sc_main:.4f}")
        res[held]["_p"] = (p_va, s1_sub, va_m)
    # cross: each held-out country scored with the thresholds tuned on the other country
    for held in labels:
        others = [l for l in labels if l != held]
        if not others:
            continue
        p_va, s1_sub, va_m = res[held].pop("_p")
        tf, ta = tuned[others[0]]
        res[held]["with_other_country_thresholds"] = score_fixed(c[va_m], p_va, true_count, s1_sub, tf, ta)
        log(f"  shift: held-out {held} with {others[0]} thresholds: {res[held]['with_other_country_thresholds']:.4f}")
    return res


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_lists(path, s1_ids, s1_pos, src, r_pos, ids2, ids3, header):
    """One row per S1 (sorted by S1 id).  Pairs must be unique; IDs are joined per S1
    without materialising a table of strings (63M test pairs would not fit in RAM)."""
    order = np.lexsort((r_pos, src, s1_pos))
    s1_pos, src, r_pos = s1_pos[order], src[order], r_pos[order]
    starts = np.searchsorted(s1_pos, np.arange(len(s1_ids) + 1))
    tab, nl = "\t", "\n"
    with open(path, "w", encoding="utf-8", newline=nl) as fh:
        fh.write(f"source1_entity_id{tab}{header}{nl}")
        for i in np.argsort(s1_ids, kind="stable"):
            a, b = starts[i], starts[i + 1]
            if b > a:
                s3 = src[a:b] == 3
                rp = r_pos[a:b]
                lst = ",".join(np.where(s3, ids3[np.minimum(rp, len(ids3) - 1)], ids2[np.minimum(rp, len(ids2) - 1)]))
            else:
                lst = ""
            fh.write(f"{s1_ids[i]}{tab}{lst}{nl}")


def predict_test(args, summary, models, feat_names, feat2, tf, ta):
    te = load_split(args.data_dir, "test", args.cache)
    ct = get_candidates("test", te, args.cache)
    n_s1 = len(te["source1"])
    summary["blocking_test"] = {"n_candidate_pairs": int(len(ct)), "mean_candidates_per_s1": len(ct) / n_s1,
                                "reduction_ratio_vs_full_cross_product":
                                    1 - len(ct) / (n_s1 * (len(te["source2"]) + len(te["source3"])))}
    ct = ct.sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True)
    prob_path = os.path.join(args.cache, "test_prob.npy")
    if os.path.exists(prob_path) and args.reuse_test_prob:
        prob = np.load(prob_path)
        log("loaded cached test probabilities")
    else:
        stats_te = TextStats(seed=CFG["seed"]).fit([te["source1"], te["source2"], te["source3"]])
        ct = base_scores(ct, te, stats_te)
        prob = np.empty(len(ct), np.float32)
        bounds = np.searchsorted(ct.s1_pos.values, np.arange(0, n_s1 + CFG["predict_chunk_s1"], CFG["predict_chunk_s1"]))
        fdir = os.path.join(args.cache, "test_feats")
        os.makedirs(fdir, exist_ok=True)
        chunks = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        for j, (a, b) in enumerate(chunks):
            F = build_features(ct.iloc[a:b], te, stats_te)[feat_names]
            prob[a:b] = models[1].predict(F)
            if 2 in models:
                F.to_parquet(os.path.join(fdir, f"chunk{j}.parquet"))
            log(f"  stage-1 predicted {b}/{len(ct)}")
        if 2 in models:
            ctx = stage2_context(ct, prob)
            for j, (a, b) in enumerate(chunks):
                F = pd.read_parquet(os.path.join(fdir, f"chunk{j}.parquet"))
                F = pd.concat([F, ctx.iloc[a:b].reset_index(drop=True)], axis=1)[feat2]
                prob[a:b] = models[2].predict(F)
            log("  stage-2 predicted")
        np.save(prob_path, prob)
    # keep only what the decision and the writers need
    s1_pos = ct.s1_pos.values.astype(np.int64)
    src = ct.src.values.astype(np.int64)
    r_pos = ct.r_pos.values.astype(np.int64)
    del ct
    # plain object arrays: indexing pandas' Arrow-backed strings per S1 is ~3000x slower
    s1_ids = np.asarray(te["source1"].entity_id.values, dtype=object)
    cty = np.asarray(te["source1"].country.values, dtype=object)
    ids2 = np.asarray(te["source2"].entity_id.values, dtype=object)
    ids3 = np.asarray(te["source3"].entity_id.values, dtype=object)
    del te
    rid = (src << 23) | r_pos
    keep = Decider(s1_pos, rid, prob).keep(tf, ta)
    del rid
    write_lists(os.path.join(args.out, "candidate_pairs.tsv"), s1_ids, s1_pos, src, r_pos, ids2, ids3,
                "candidate_entity_ids")
    write_lists(os.path.join(args.out, "matching_results.tsv"), s1_ids, s1_pos[keep], src[keep], r_pos[keep],
                ids2, ids3, "matched_entity_ids")
    kept_s1 = s1_pos[keep]
    summary["test"] = {
        "t_first": tf, "t_add": ta, "pred_pairs": int(keep.sum()),
        "pred_singleton_fraction": float(1 - len(np.unique(kept_s1)) / n_s1),
        "pred_singleton_fraction_by_country": {
            k: float(1 - len(np.unique(kept_s1[cty[kept_s1] == k])) / (cty == k).sum()) for k in np.unique(cty)},
        "mean_pred_matches_by_country": {k: float((cty[kept_s1] == k).sum() / (cty == k).sum()) for k in np.unique(cty)}}
    log(f"test: {keep.sum()} matched pairs written")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--shift-check", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--blocking-only", action="store_true", help="build/cached candidates for both splits, then exit")
    ap.add_argument("--predict-only", action="store_true",
                    help="skip training: reuse cache/model_stage*.txt and thresholds from output/run_summary.json")
    ap.add_argument("--reuse-test-prob", action="store_true", help="reuse cache/test_prob.npy if present")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    summary = {"config": {k: v for k, v in CFG.items() if not k.startswith("_")}}
    rng = np.random.default_rng(CFG["seed"])
    if args.predict_only:
        with open(os.path.join(args.out, "run_summary.json")) as fh:
            summary = json.load(fh)
        models = {int(f[len("model_stage"):-4]): lgb.Booster(model_file=os.path.join(args.cache, f))
                  for f in os.listdir(args.cache) if f.startswith("model_stage") and f.endswith(".txt")}
        if not summary["cv"].get("stage2"):
            models = {1: models[1]}
        feat_names = models[1].feature_name()
        feat2 = models[2].feature_name() if 2 in models else None
        predict_test(args, summary, models, feat_names, feat2, summary["cv"]["t_first"], summary["cv"]["t_add"])
        with open(os.path.join(args.out, "run_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        return

    # ---------------- train
    tr = load_split(args.data_dir, "train", args.cache)
    gt, pairs = load_ground_truth(args.data_dir)
    summary["diagnostics"] = diagnostics(tr, gt, pairs)
    n_s1 = len(tr["source1"])
    train_mask = rng.random(n_s1) < CFG["train_s1_frac"]
    CFG["_train_s1_mask"] = train_mask
    c = get_candidates("train", tr, args.cache, s1_mask=train_mask)
    if args.blocking_only:
        del tr, c
        te = load_split(args.data_dir, "test", args.cache)
        get_candidates("test", te, args.cache)
        log("blocking-only: done")
        return
    c, true_count = label_candidates(c, tr, pairs)
    summary["blocking_train"] = blocking_report(c, true_count * train_mask, tr, train_mask)

    stats = TextStats(seed=CFG["seed"]).fit([tr["source1"], tr["source2"], tr["source3"]])
    c = base_scores(c, tr, stats)
    log("base scores + context done")
    ct = c
    del c
    log(f"training subset: {train_mask.sum()} S1, {len(ct)} pairs, pos={ct.y.sum()}")
    X = build_features(ct, tr, stats)
    y = ct.y.values
    groups = ct.s1_pos.values
    feat_names = list(X.columns)
    s1_country = tr["source1"].country.values
    del tr  # free the train strings before model training

    s1_subset = np.flatnonzero(train_mask)
    oof, best_iters, _ = cross_validate(X, y, groups)
    iters = {1: int(np.mean(best_iters))}
    summary["cv_stage1_best_iters"] = best_iters
    feat2 = None
    if CFG["stage2"]:
        sc1, _, _, _, _ = evaluate(ct, oof, true_count, s1_subset)
        log(f"stage-1 CV macro F0.5 = {sc1:.5f}")
        summary["cv_stage1_macro_f05"] = sc1
        X = pd.concat([X, stage2_context(ct, oof)], axis=1)
        feat2 = list(X.columns)
        oof, best_iters2, _ = cross_validate(X, y, groups)
        iters[2] = int(np.mean(best_iters2))
        summary["cv_stage2_best_iters"] = best_iters2
    CFG["mean_best_iter"] = iters[max(iters)]
    sc, tf, ta, s1c, rid = evaluate(ct, oof, true_count, s1_subset)
    CFG["_t_first"], CFG["_t_add"] = tf, ta
    ntr = near_tie_rate(rid, oof, min_top=ta)
    log(f"CV macro F0.5 = {sc:.5f}  t_first={tf:.4f} t_add={ta:.4f} near_tie_rate={ntr:.4f}")
    summary["cv"] = {"macro_f05": sc, "t_first": tf, "t_add": ta, "near_tie_rate": ntr,
                     "stage2": CFG["stage2"], "n_train_s1": int(train_mask.sum()), "n_train_pairs": int(len(ct))}
    ct[["s1_pos", "src", "r_pos", "y"]].assign(p=oof).to_parquet(os.path.join(args.cache, "oof_pairs.parquet"))

    if args.shift_check:
        summary["shift_check"] = shift_check(X, y, ct, true_count, s1_country)

    # refit: stage 1 on its own features, stage 2 on stage-1 features + OOF context
    models = {}
    X1 = X[feat_names]
    for st, Xs in ((1, X1), (2, X)):
        if st not in iters:
            continue
        log(f"refit stage {st} on {len(Xs)} pairs, {iters[st]} rounds")
        models[st] = lgb.train(dict(CFG["lgb"], seed=CFG["seed"]), lgb.Dataset(Xs, y), iters[st])
        models[st].save_model(os.path.join(args.cache, f"model_stage{st}.txt"))
    last = models[max(models)]
    imp = pd.Series(last.feature_importance("gain"), index=last.feature_name()).sort_values(ascending=False)
    summary["feature_importance_gain"] = {k: float(v) for k, v in imp.items()}
    log("top features: " + ", ".join(imp.index[:12]))
    del X, X1, ct
    with open(os.path.join(args.out, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if args.skip_test:
        return

    # ---------------- test
    predict_test(args, summary, models, feat_names, feat2, tf, ta)
    with open(os.path.join(args.out, "run_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
