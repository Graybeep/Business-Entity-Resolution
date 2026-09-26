"""PREPARED, NOT RUN - test-time density matching for run 10 (one LB probe against run 10).

Why: training blocking runs the forward passes for the 40 % S1 subset only, so every record-side
competition feature was learned with ~40 % of a record's competing S1s present; at test all S1s are
present (adversarial AUC 0.911; candidate S1s per record 3.9 train vs 6.2-6.4 test; EXPERIMENTS.md
row 20).  This recomputes exactly those features on test against a random 40 % of each country's
S1s (the pair's own S1 always counts), K times, and averages the final probabilities.

What changes (test only; models, features definitions and thresholds are untouched - NO retraining):
  stage 1  ctx_rank_rec, ctx_gap_rec, ctx_n_rec    (record-side rank / gap / count by base score)
  stage 2  p1_rec_max, p1_rec_gap, p1_rec_rank, p1_rec_n_hi, p1_rec_sum, p1_rec_best_other
           (record-side context of the stage-1 P, which itself changes with the stage-1 inputs)
  sibling features follow the new stage-1 P (siblings = the S1's candidates with P >= 0.5).
Unchanged: every S1-side feature, rank_rev / sim_rev (the reverse pass already ranks against ALL
S1s in training too), name-frequency (whole-split statistics on both sides), the one-owner rule and
the (t_first, t_add) decision, which use the full competition as before.
Round-1 pairs are matched against round-1 competitors, round-2 pairs against the union (same rule as
their unmatched features).

Cost (estimate): one-off stage-1 feature store for the 64M round-1 test pairs (~80 min, ~17 GB on
disk, memmap), then per draw ~50 min (context rewrite + stage-1 scoring of ~92M pairs + stage 2).
K = 3 -> ~3.5-4 h; K = 2 -> ~3 h.  Peak RAM ~10 GB (one country at a time, features on disk).

python density_match.py --draws 3 --out runs/v5
"""
import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL  # noqa: E402
import run10 as R  # noqa: E402
from data import load_split  # noqa: E402
from features import TextStats  # noqa: E402
from postprocess import Decider  # noqa: E402
from stage2_sibling import P_MIN, sibling_features  # noqa: E402

KEEP_FRAC = 0.40
T0 = time.time()


def log(msg):
    """Timestamped progress line."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def matched_record_context(rec_key, s1_pos, score, in_draw, fill=-1.0):
    """Record-side competition statistics where the competitors of pair i are the OTHER pairs of
    the same record whose S1 is in the draw; pair i itself always counts.
    Returns (rank [1 = best, ties -> min], max_other [NaN if none], n_competitors, sum_other,
    n_other_above_half).  Inputs are aligned 1-D arrays; score NaN -> `fill` (as add_context)."""
    s = np.where(np.isnan(score), fill, score).astype(np.float64)
    m = in_draw[s1_pos]
    order = np.lexsort((-s, rec_key))
    k, sv, mv = rec_key[order], s[order], m[order]
    start = np.r_[True, k[1:] != k[:-1]]
    grp = np.cumsum(start) - 1
    # strictly-greater count of in-draw members: cumulative count of in-draw members in earlier tie groups
    tie_start = start | np.r_[True, sv[1:] != sv[:-1]]
    tie = np.cumsum(tie_start) - 1
    tie_m = np.bincount(tie, weights=mv)
    tie_grp = grp[tie_start]
    cum = np.cumsum(tie_m) - tie_m                       # in-draw members before this tie group (global)
    grp_first_tie = np.r_[0, np.flatnonzero(tie_grp[1:] != tie_grp[:-1]) + 1]
    grp_base = np.repeat(cum[grp_first_tie], np.diff(np.r_[grp_first_tie, len(tie_m)]))
    greater = (cum - grp_base)[tie]
    # in-draw max / second max / count / sum / n>0.5 per record
    n_m = np.bincount(grp, weights=mv)
    sm = np.where(mv, sv, -np.inf)
    first_m = pd.Series(sm).groupby(grp).max().values
    cnt_first = np.bincount(grp, weights=mv & (sv == first_m[grp]))
    second_m = pd.Series(np.where(sv == first_m[grp], -np.inf, sm)).groupby(grp).max().values
    self_is_unique_top = mv & (sv == first_m[grp]) & (cnt_first[grp] == 1)
    max_other = np.where(self_is_unique_top, second_m[grp], first_m[grp])
    max_other = np.where(np.isinf(max_other), np.nan, max_other)
    n_other = n_m[grp] - mv
    sum_other = np.bincount(grp, weights=np.where(mv, sv, 0.0))[grp] - np.where(mv, sv, 0.0)
    hi_other = np.bincount(grp, weights=mv & (sv > 0.5))[grp] - (mv & (sv > 0.5))
    out = [None] * 5
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    for j, a in enumerate((greater + 1, max_other, n_other, sum_other, hi_other)):
        out[j] = np.asarray(a, np.float64)[inv]
    return out


def stage1_ctx(rec_key, s1_pos, base, in_draw):
    """Density-matched ctx_rank_rec / ctx_gap_rec / ctx_n_rec (definitions of features.add_context)."""
    rank, mx, n, _, _ = matched_record_context(rec_key, s1_pos, base, in_draw)
    own = np.where(np.isnan(base), -1.0, base)
    best = np.fmax(own, np.where(np.isnan(mx), -np.inf, mx))
    return rank.astype(np.float32), (best - own).astype(np.float32), (n + 1).astype(np.float32)


def stage2_rec_ctx(rec_key, s1_pos, p, in_draw):
    """Density-matched record-side columns of pipeline.stage2_context."""
    rank, mx, _, sm, hi = matched_record_context(rec_key, s1_pos, p, in_draw, fill=0.0)
    mx0 = np.where(np.isnan(mx), 0.0, mx)
    best = np.maximum(p, mx0)
    return {"p1_rec_max": best, "p1_rec_gap": best - p, "p1_rec_rank": rank,
            "p1_rec_n_hi": (p > 0.5) + hi, "p1_rec_sum": p + sm, "p1_rec_best_other": mx0}


def build_round1_store(data, stats):
    """One-off: stage-1 features + base score of every round-1 test pair (sorted order), per country,
    written to cache/dm_test_r1_feats.npy (memmap) and dm_test_r1_base.npy."""
    names = R.stage1_names()
    c1, _, _ = R.round1("test", full=True)
    X = np.lib.format.open_memmap(os.path.join(R.CACHE, "dm_test_r1_feats.npy"), mode="w+", dtype=np.float32,
                                  shape=(len(c1), len(names)))
    base = np.empty(len(c1), np.float32)
    cty = data["source1"].country.values[c1.s1_pos.values]
    for c_ in sorted(np.unique(cty)):
        idx = np.flatnonzero(cty == c_)
        cc = PL.base_scores(c1.iloc[idx].reset_index(drop=True), data, stats)
        base[idx] = cc.base.values
        for b in range(0, len(idx), R.CHUNK):
            X[idx[b:b + R.CHUNK]] = PL.build_features(cc.iloc[b:b + R.CHUNK], data, stats)[names].values
        del cc
        log(f"  round-1 store {c_}: {len(idx)} pairs")
    X.flush()
    np.save(os.path.join(R.CACHE, "dm_test_r1_base.npy"), base)


def main():
    """Build the round-1 store (once), then K density-matched draws, average, decide, write."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", default="runs/v5")
    args = ap.parse_args()
    data = load_split(os.path.join(R.ROOT, "dataset"), "test", R.CACHE)
    stats = TextStats(seed=PL.CFG["seed"]).fit([data["source1"], data["source2"], data["source3"]])
    if not os.path.exists(os.path.join(R.CACHE, "dm_test_r1_base.npy")):
        build_round1_store(data, stats)
    raise NotImplementedError(
        "draw loop: per draw j and country - (1) rewrite ctx_*_rec in the round-1 store rows with "
        "stage1_ctx over round-1 pairs and in the round-2 rows over the union (base of round-2 pairs "
        "needs saving in run10.r2feats); (2) stage-1 predict (v2 refit) -> p1_j; (3) run10.union_inputs "
        "with p1_j and stage2_rec_ctx replacing the record-side context columns; (4) r10 stage-2 "
        "predict -> p_j; average p over draws; decide with r10 thresholds; write TSVs. "
        "To be completed and smoke-tested before the probe (not run tonight).")


if __name__ == "__main__":
    main()
