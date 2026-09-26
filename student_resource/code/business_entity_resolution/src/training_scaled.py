"""Stage 4 (scaled variant): train on more S1 entities than fit in RAM as a dense matrix.

- Pair features are computed in chunks and stored on disk as float32 parquet.
- TRAINING rows keep every positive, every "hard" negative and a seeded random fraction of the
  remaining "easy" negatives.  VALIDATION (out-of-fold) predictions are streamed over the FULL
  candidate sets, so the macro F0.5 used for threshold tuning stays exact.
- LightGBM bins the sampled rows once; each fold uses Dataset.subset (no raw-matrix copies).

This deliberately departs from the "no resampling" line of the decision log (explicit
experiment request, see EXPERIMENTS.md); it is enabled only through CFG["neg_sampling"].
"""
import glob
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def hard_negative_mask(meta, hard_rank_s1, hard_rank_rec):
    """Stage 4 helper.  Input: candidate metadata with ctx_rank_s1 / ctx_rank_rec (ranks of the
    cheap base TF-IDF score within the S1's candidates per source and within the record's
    competing S1s).  Output: bool mask of pairs that count as hard (kept even if negative)."""
    return (meta.ctx_rank_s1.values <= hard_rank_s1) | (meta.ctx_rank_rec.values <= hard_rank_rec)


def write_feature_chunks(c, build_features, data, stats, fdir, log, chunk_pairs=4_000_000):
    """Stage 3 to disk.  Input: candidate frame c (with base/context columns), the feature builder,
    data frames and fitted TextStats.  Writes float32 feature chunks chunk_{k}.parquet to fdir and
    returns the list of feature names.  Existing complete chunk sets are reused."""
    os.makedirs(fdir, exist_ok=True)
    done = os.path.join(fdir, "_complete")
    if os.path.exists(done):
        return open(done).read().split(",")
    names = None
    for k, a in enumerate(range(0, len(c), chunk_pairs)):
        F = build_features(c.iloc[a:a + chunk_pairs], data, stats).astype(np.float32)
        F.to_parquet(os.path.join(fdir, f"chunk_{k:04d}.parquet"), index=False)
        names = list(F.columns)
        log(f"  feature chunk {k} written ({a + len(F)}/{len(c)})")
        del F
    open(done, "w").write(",".join(names))
    return names


def iter_chunks(fdir):
    """Yield (row_offset, float32 DataFrame) for every stored feature chunk, in row order."""
    off = 0
    for p in sorted(glob.glob(os.path.join(fdir, "chunk_*.parquet"))):
        F = pd.read_parquet(p)
        yield off, F
        off += len(F)


def load_rows(fdir, rows, n_feat):
    """Load the given (sorted) global row indices from the chunk store into one float32 array."""
    out = np.empty((len(rows), n_feat), np.float32)
    w = 0
    for off, F in iter_chunks(fdir):
        lo, hi = np.searchsorted(rows, [off, off + len(F)])
        if hi > lo:
            out[w:w + hi - lo] = F.values[rows[lo:hi] - off]
            w += hi - lo
    return out


def predict_rows(fdir, model, row_mask, num_iteration=None):
    """Stream the chunk store and predict every row where row_mask is True (full candidate
    sets).  Returns a float32 array aligned with the global row order (NaN elsewhere)."""
    out = np.full(len(row_mask), np.nan, np.float32)
    for off, F in iter_chunks(fdir):
        m = row_mask[off:off + len(F)]
        if m.any():
            out[off:off + len(F)][m] = model.predict(F.values[m], num_iteration=num_iteration)
    return out


def scaled_cv(fdir, feat_names, meta, cfg_lgb, cfg, log, save_prefix=None, predict_oof=True):
    """Stage 4 grouped CV with negative subsampling in training folds only.
    Input: chunk store, feature names, meta (s1_pos, y, ctx ranks) aligned with the store.
    Output: dict with full-candidate OOF probabilities, best iterations, the sampled-row index
    and the binned LightGBM Dataset over the sampled rows (reused for refit / shift check).
    save_prefix: if set, each fold model is saved as <save_prefix>_fold{f}.txt (for test-time fold
    averaging).  predict_oof=False skips the full-candidate OOF streaming (fold models only)."""
    ns = cfg["neg_sampling"]
    y = meta.y.values.astype(np.int8)
    hard = hard_negative_mask(meta, ns["hard_rank_s1"], ns["hard_rank_rec"])
    u = np.random.default_rng(cfg["seed"] + 1).random(len(meta))
    sampled = (y == 1) | hard | (u < ns["easy_rate"])
    rows = np.flatnonzero(sampled)
    log(f"sampling: {len(meta)} pairs -> {len(rows)} training rows (pos {int(y.sum())}, "
        f"hard neg {int((hard & (y == 0)).sum())}, easy kept {int((~hard & (y == 0) & sampled).sum())})")
    t = time.time()
    X = load_rows(fdir, rows, len(feat_names))
    full = lgb.Dataset(X, y[rows], feature_name=feat_names, free_raw_data=True,
                       params={"max_bin": cfg_lgb.get("max_bin", 255), "verbose": -1}).construct()
    del X
    log(f"binned {len(rows)} rows in {time.time() - t:.0f}s")
    folds = np.empty(len(meta), np.int8)
    for f, (_, va) in enumerate(GroupKFold(cfg["n_folds"]).split(np.zeros(len(meta)), y, meta.s1_pos.values)):
        folds[va] = f
    oof = np.full(len(meta), np.nan, np.float32)
    best = []
    fr = folds[rows]
    for f in range(cfg["n_folds"]):
        tr_idx = np.flatnonzero(fr != f)
        va_idx = np.flatnonzero(fr == f)        # sampled rows of the fold: early stopping only
        m = lgb.train(dict(cfg_lgb, seed=cfg["seed"] + f), full.subset(tr_idx), cfg["num_boost_round"],
                      valid_sets=[full.subset(va_idx)],
                      callbacks=[lgb.early_stopping(cfg["early_stopping"], verbose=False)])
        best.append(m.best_iteration)
        if save_prefix:
            m.save_model(f"{save_prefix}_fold{f}.txt", num_iteration=m.best_iteration)
        if predict_oof:
            p = predict_rows(fdir, m, folds == f, num_iteration=m.best_iteration)   # FULL candidate set
            oof[folds == f] = p[folds == f]
        log(f"  fold {f}: best_iter={m.best_iteration} (early-stop logloss on sampled rows "
            f"{m.best_score['valid_0']['binary_logloss']:.5f})")
    return {"oof": oof, "best_iters": best, "rows": rows, "full": full, "folds": folds}


def scaled_shift_check(fdir, meta, s1_country, res, cfg_lgb, cfg, n_iter, evaluate, score_fixed, c, true_count,
                       train_mask, t_main, log):
    """Country-shift check for the scaled model: train on the sampled rows of one country,
    predict the other country's FULL candidate sets, report retuned / main / other-country scores."""
    cty = s1_country[meta.s1_pos.values]
    rows_cty = cty[res["rows"]]
    out, tuned, cache = {}, {}, {}
    labels = [l for l in np.unique(s1_country[train_mask]) if l]
    for held in labels:
        m = lgb.train(dict(cfg_lgb, seed=cfg["seed"]), res["full"].subset(np.flatnonzero(rows_cty != held)), n_iter)
        va = cty == held
        p = predict_rows(fdir, m, va)[va]
        s1_sub = np.flatnonzero((s1_country == held) & train_mask)
        sc, tf, ta, _, _ = evaluate(c[va], p, true_count, s1_sub)
        out[held] = {"retuned": sc, "t_first": tf, "t_add": ta,
                     "with_main_oof_thresholds": score_fixed(c[va], p, true_count, s1_sub, *t_main)}
        tuned[held] = (tf, ta)
        cache[held] = (p, s1_sub, va)
        log(f"  shift: held-out {held}: retuned={sc:.4f} main={out[held]['with_main_oof_thresholds']:.4f}")
    for held in labels:
        other = [l for l in labels if l != held]
        if other:
            p, s1_sub, va = cache[held]
            out[held]["with_other_country_thresholds"] = score_fixed(c[va], p, true_count, s1_sub, *tuned[other[0]])
            log(f"  shift: held-out {held} with {other[0]} thresholds: {out[held]['with_other_country_thresholds']:.4f}")
    return out
