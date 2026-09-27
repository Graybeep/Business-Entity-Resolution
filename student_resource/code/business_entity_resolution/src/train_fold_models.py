"""Regenerate run 8's five stage-1 fold models for test-time fold averaging.

Run 8 (scaled training) kept only the final refit.  This rebuilds the exact run-8 training inputs
from caches (cached 40 % candidates + cached feature chunks; base scores are recomputed because the
hard-negative mask needs ctx ranks) and re-trains the five folds with the same seeds, folds and
sampling, saving cache/v2_stage1_fold{0..4}.txt.  The log prints each fold's best iteration; they
should match run 8 (1787, 2165, 2569, 2317, 2751) up to multithreaded float-summation noise.

Estimated: ~10 min inputs + ~70 min training, peak RAM ~16-18 GB committed (as run 8's CV).
Used only if run 9's LB gain over run 2 is < +0.003 (see predict_stage2.py --stage1-folds).

python train_fold_models.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL  # noqa: E402
import training_scaled as TS  # noqa: E402
from data import load_ground_truth, load_split  # noqa: E402
from features import TextStats  # noqa: E402


def main():
    """Rebuild run-8 meta, then run the grouped fold training with model saving and no OOF pass."""
    PL.CFG["train_s1_frac"] = 0.4
    PL.CFG["neg_sampling"] = dict(PL.NEG_SAMPLING_DEFAULT)
    cache = os.environ.get("ER_CACHE", "cache")
    data_dir = os.environ.get("ER_DATA", "dataset")
    tr = load_split(data_dir, "train", cache)
    _, pairs = load_ground_truth(data_dir)
    u_s1 = np.random.default_rng(PL.CFG["seed"]).random(len(tr["source1"]))
    train_mask = u_s1 < PL.CFG["train_s1_frac"]
    c = PL.get_candidates("train", tr, cache, s1_mask=train_mask)          # cached train_cands_f0.4
    c, _ = PL.label_candidates(c, tr, pairs)
    stats = TextStats(seed=PL.CFG["seed"]).fit([tr["source1"], tr["source2"], tr["source3"]])
    c = PL.base_scores(c, tr, stats)
    fdir = os.path.join(cache, "v2_train_feats")
    with open(os.path.join(fdir, "_complete")) as fh:
        feat_names = fh.read().split(",")
    meta = c[["s1_pos", "y", "ctx_rank_s1", "ctx_rank_rec"]].copy()
    ref = pd.read_parquet(os.path.join(cache, "v2_oof_pairs.parquet"), columns=["s1_pos", "src", "r_pos", "y"])
    assert (ref.s1_pos.values == c.s1_pos.values).all() and (ref.r_pos.values == c.r_pos.values).all()         and (ref.y.values == c.y.values).all(), "rebuilt candidates are not in run-8 order"
    del c, tr, ref
    res = TS.scaled_cv(fdir, feat_names, meta, PL.CFG["lgb"], PL.CFG, PL.log,
                       save_prefix=os.path.join(cache, "v2_stage1"), predict_oof=False)
    np.save(os.path.join(cache, "v2_folds.npy"), res["folds"])          # aligned with v2_oof_pairs rows
    PL.log(f"fold models saved; best iterations {res['best_iters']} (run 8: [1787, 2165, 2569, 2317, 2751])")


if __name__ == "__main__":
    main()
