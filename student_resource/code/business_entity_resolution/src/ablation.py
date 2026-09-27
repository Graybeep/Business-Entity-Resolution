"""Ablation of run 10 WITHOUT record-side competition features (train/test shift source, row 20).

Dropped:  stage 1  ctx_rank_rec, ctx_gap_rec, ctx_n_rec
          stage 2  p1_rec_max, p1_rec_gap, p1_rec_rank, p1_rec_n_hi, p1_rec_sum, p1_rec_best_other
Everything else is run 10: same candidates (round 1 + round 2), same training rows / sampling / folds
(run-8 recipe: positives + hard negatives + 25 % easy, v2_folds.npy), same stage-2 rows and thresholds
procedure (retuned on the new OOF).  OOF CV is expected to drop; the paired LB vs run 10 decides.

Steps (each caches cache/abl_*):  python ablation.py stage1 | stage2 | test
  stage1  5-fold stage-1 CV on the cached run-8 feature store minus the 3 columns (fold models saved),
          OOF over the full 32M round-1 pairs + round-2 pairs by fold, refit.   ~1.7 h, ~9 GB
  stage2  stage-2 CV on the union with the new stage-1 OOF minus the 6 columns, thresholds, refit. ~0.5 h
  test    per country: stage-1 refit on pairs with run-8 refit P >= 1e-4 (others cannot reach the
          thresholds), then stage 2; decide; write runs/v4_abl.                  ~1.5 h, ~10 GB
"""
import glob
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
import run10 as R  # noqa: E402
from data import load_split  # noqa: E402
from features import TextStats  # noqa: E402
from postprocess import Decider  # noqa: E402
from training_scaled import hard_negative_mask  # noqa: E402

DROP1 = ["ctx_rank_rec", "ctx_gap_rec", "ctx_n_rec"]
DROP2 = ["p1_rec_max", "p1_rec_gap", "p1_rec_rank", "p1_rec_n_hi", "p1_rec_sum", "p1_rec_best_other"]
T0 = time.time()


def log(msg):
    """Timestamped progress line."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def ap(name):
    """Cache path of an ablation artefact."""
    return os.path.join(R.CACHE, f"abl_{name}")


def store_chunks():
    """Run-8 train feature chunk files in row order."""
    return sorted(glob.glob(os.path.join(R.CACHE, "v2_train_feats", "chunk_*.parquet")))


def stage1():
    """Stage-1 CV without the record-side columns: same sampled rows, folds and seeds as run 8."""
    names = R.stage1_names()
    keep = [n for n in names if n not in DROP1]
    oofp = pd.read_parquet(os.path.join(R.CACHE, "v2_oof_pairs.parquet"), columns=["s1_pos", "y"])
    ctx = pd.concat([pd.read_parquet(p, columns=["ctx_rank_s1", "ctx_rank_rec"]) for p in store_chunks()],
                    ignore_index=True)
    meta = pd.concat([oofp, ctx], axis=1)
    ns = PL.NEG_SAMPLING_DEFAULT
    y = meta.y.values.astype(np.int8)
    hard = hard_negative_mask(meta, ns["hard_rank_s1"], ns["hard_rank_rec"])
    u = np.random.default_rng(PL.CFG["seed"] + 1).random(len(meta))
    rows = np.flatnonzero((y == 1) | hard | (u < ns["easy_rate"]))   # identical to the run-8 sample
    del ctx, hard, u
    X = np.empty((len(rows), len(keep)), np.float32)
    off = w = 0
    for p in store_chunks():
        F = pd.read_parquet(p, columns=keep)
        lo, hi = np.searchsorted(rows, [off, off + len(F)])
        X[w:w + hi - lo] = F.values[rows[lo:hi] - off]
        w += hi - lo
        off += len(F)
    full = lgb.Dataset(X, y[rows], feature_name=keep, free_raw_data=True,
                       params={"max_bin": 255, "verbose": -1}).construct()
    del X
    log(f"binned {len(rows)} sampled rows x {len(keep)} features")
    folds = np.load(os.path.join(R.CACHE, "v2_folds.npy"))
    fr = folds[rows]
    best, models = [], []
    for f in range(5):
        m = lgb.train(dict(PL.CFG["lgb"], seed=PL.CFG["seed"] + f), full.subset(np.flatnonzero(fr != f)),
                      PL.CFG["num_boost_round"], valid_sets=[full.subset(np.flatnonzero(fr == f))],
                      callbacks=[lgb.early_stopping(PL.CFG["early_stopping"], verbose=False)])
        m.save_model(ap(f"stage1_fold{f}.txt"), num_iteration=m.best_iteration)
        best.append(m.best_iteration)
        models.append(lgb.Booster(model_file=ap(f"stage1_fold{f}.txt")))
        log(f"  fold {f}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.5f}")
    # OOF over every round-1 pair (full candidate sets)
    oof = np.empty(len(meta), np.float32)
    off = 0
    for p in store_chunks():
        F = pd.read_parquet(p, columns=keep).values
        fo = folds[off:off + len(F)]
        for f in range(5):
            s = fo == f
            if s.any():
                oof[off:off + len(F)][s] = models[f].predict(F[s])
        off += len(F)
    np.save(ap("train_r1_p1.npy"), oof)
    # round-2 train pairs by the S1's fold
    r2 = pd.read_parquet(R.cp("train_r2_pairs.parquet"))
    s1_fold = pd.Series(folds, index=oofp.s1_pos.values).groupby(level=0).first()
    fo2 = s1_fold.reindex(r2.s1_pos.values).values.astype(np.int64)
    F2 = np.load(R.cp("train_r2_feats.npy"), mmap_mode="r")
    kidx = [names.index(n) for n in keep]
    p2 = np.empty(len(r2), np.float32)
    for b in range(0, len(r2), R.CHUNK):
        Fb = np.asarray(F2[b:b + R.CHUNK])[:, kidx]
        fb = fo2[b:b + R.CHUNK]
        pb = np.empty(len(Fb), np.float32)
        for f in range(5):
            s = fb == f
            if s.any():
                pb[s] = models[f].predict(Fb[s])
        p2[b:b + R.CHUNK] = pb
    np.save(ap("train_r2_p1.npy"), p2)
    log("OOF stage-1 scores saved; refit")
    final = lgb.train(dict(PL.CFG["lgb"], seed=PL.CFG["seed"]), full, int(np.mean(best)))
    final.save_model(ap("model_stage1.txt"))
    with open(ap("stage1.json"), "w") as fh:
        json.dump({"best_iters": best, "features": keep}, fh, indent=2)
    log(f"stage 1 done; best iters {best}")


def stage2():
    """Stage-2 CV on the union with the ablation stage-1 OOF, without the record-side columns."""
    data = load_split(os.path.join(R.ROOT, "dataset"), "train", R.CACHE)
    u, rows, X, names = R.union_inputs("train", data, p1_all=np.load(ap("train_r1_p1.npy")),
                                       p2_all=np.load(ap("train_r2_p1.npy")), drop=DROP1 + DROP2)
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
        log(f"  s2 fold {f}: best_iter={m.best_iteration}")
    p = u.p.values.copy()
    p[rows] = oof
    tca = R.true_counts(data)
    u40 = np.random.default_rng(PL.CFG["seed"]).random(len(data["source1"]))
    cty = data["source1"].country.values
    res = {}
    for name, m in (("full40", u40 < 0.40), ("India", (u40 < 0.40) & (cty == "India")),
                    ("US", (u40 < 0.40) & (cty == "US"))):
        sc, tf, ta = R.fast_score(u, p, tca, m)
        res[name] = {"macro_f05": sc, "t_first": tf, "t_add": ta}
        log(f"RESULT ablation {name}: {sc:.5f} (tf {tf:.3f} ta {ta:.3f})")
    log("  run 10 reference: full40 0.97609 / India 0.96807 / US 0.98127")
    final = lgb.train(dict(PL.CFG["lgb"], seed=PL.CFG["seed"]), lgb.Dataset(X, y, feature_name=names), int(np.mean(best)))
    final.save_model(ap("stage2_model.txt"))
    res["best_iters"] = best
    with open(ap("train_result.json"), "w") as fh:
        json.dump(res, fh, indent=2)


def test(out="runs/v4_abl"):
    """Per country: ablation stage 1 on pairs that can matter, then stage 2; decide; write."""
    data = load_split(os.path.join(R.ROOT, "dataset"), "test", R.CACHE)
    stats = TextStats(seed=PL.CFG["seed"]).fit([data["source1"], data["source2"], data["source3"]])
    names = R.stage1_names()
    keep = json.load(open(ap("stage1.json")))["features"]
    kidx = [names.index(n) for n in keep]
    m1 = lgb.Booster(model_file=ap("model_stage1.txt"))
    m2 = lgb.Booster(model_file=ap("stage2_model.txt"))
    thr = json.load(open(ap("train_result.json")))["full40"]
    c1, p1_run8, _ = R.round1("test", full=True)
    r2 = pd.read_parquet(R.cp("test_r2_pairs.parquet"))
    p2_run8 = np.load(R.cp("test_r2_p1.npy"))
    p1_new, p2_new = p1_run8.copy(), p2_run8.copy()
    s1_cty = data["source1"].country.values
    cty1 = s1_cty[c1.s1_pos.values]
    for c_ in sorted(x for x in np.unique(s1_cty) if x):
        idx = np.flatnonzero(cty1 == c_)
        cc = PL.base_scores(c1.iloc[idx].reset_index(drop=True), data, stats)
        sel = np.flatnonzero(p1_run8[idx] >= 1e-4)
        for b in range(0, len(sel), R.CHUNK):
            s = sel[b:b + R.CHUNK]
            p1_new[idx[s]] = m1.predict(PL.build_features(cc.iloc[s], data, stats)[keep].values)
        del cc
        log(f"  test {c_}: ablation stage 1 on {len(sel)} round-1 pairs")
    F2 = np.load(R.cp("test_r2_feats.npy"), mmap_mode="r")
    sel2 = np.flatnonzero(p2_run8 >= 1e-4)
    for b in range(0, len(sel2), R.CHUNK):
        s = sel2[b:b + R.CHUNK]
        p2_new[s] = m1.predict(np.asarray(F2[s])[:, kidx])
    del c1
    parts = []
    for c_ in sorted(x for x in np.unique(s1_cty) if x):
        u, rows, X, nm = R.union_inputs("test", data, c_, p1_all=p1_new, p2_all=p2_new, drop=DROP1 + DROP2)
        assert m2.feature_name() == nm
        p = u.p.values.copy()
        for b in range(0, len(rows), R.CHUNK):
            p[rows[b:b + R.CHUNK]] = m2.predict(X[b:b + R.CHUNK])
        parts.append(u[["s1_pos", "src", "r_pos"]].assign(p=p))
        log(f"  test {c_}: stage 2 scored")
    u = pd.concat(parts, ignore_index=True).sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True)
    s1_pos, src, r_pos = (u[k].values.astype(np.int64) for k in ("s1_pos", "src", "r_pos"))
    keepm = Decider(s1_pos, (src << 23) | r_pos, u.p.values.astype(np.float64)).keep(thr["t_first"], thr["t_add"])
    ids = {k: np.asarray(data[f"source{k}"].entity_id.values, dtype=object) for k in (2, 3)}
    s1_ids = np.asarray(data["source1"].entity_id.values, dtype=object)
    outd = os.path.join(R.ROOT, out)
    os.makedirs(outd, exist_ok=True)
    PL.write_lists(os.path.join(outd, "candidate_pairs.tsv"), s1_ids, s1_pos, src, r_pos, ids[2], ids[3],
                   "candidate_entity_ids")
    PL.write_lists(os.path.join(outd, "matching_results.tsv"), s1_ids, s1_pos[keepm], src[keepm], r_pos[keepm],
                   ids[2], ids[3], "matched_entity_ids")
    cty = np.asarray(s1_cty, dtype=object)
    kept = s1_pos[keepm]
    summ = {"t_first": thr["t_first"], "t_add": thr["t_add"], "pred_pairs": int(keepm.sum()),
            "pred_singleton_fraction_by_country": {k: float(1 - len(np.unique(kept[cty[kept] == k])) / (cty == k).sum())
                                                   for k in np.unique(cty)}}
    json.dump(summ, open(os.path.join(outd, "test_summary.json"), "w"), indent=2)
    log(f"test done: {summ}")


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2, "test": test}[sys.argv[1]]()
