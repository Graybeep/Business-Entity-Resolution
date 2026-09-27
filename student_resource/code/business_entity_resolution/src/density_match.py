"""Test-time density matching for run 10 (one paired LB probe against run 10; no retraining).

Why: training blocking runs the forward passes for the 40 % S1 subset only, so every record-side
competition feature was learned with ~40 % of a record's competing S1s present; at test all S1s are
present (adversarial AUC 0.911; candidate S1s per record 3.9 train vs 6.2-6.4 test; EXPERIMENTS.md
row 20).  This recomputes exactly those features on test against a FIXED hash-sampled fraction of
test S1s (the pair's own S1 always counts).  The fraction matches the record-level competitor count
(test 6.41 -> train 3.86 S1s per record: f = 0.529).  Blocking, candidates and the one-owner
decision stay on the full pool.

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

Stage 1 is re-scored only for pairs whose original stage-1 P >= 1e-4 (pairs below cannot reach the
decision thresholds); context statistics still use every pair.  One country at a time (~10 GB).

python density_match.py --frac 0.529 --out runs/v4_dm
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

KEEP_FRAC = 0.529        # record-level match: test S1s/record 6.41 -> train 3.86 (f = 2.86 / 5.41)
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


def in_draw_mask(entity_ids, frac):
    """Fixed hash-sampled competitor universe: S1 is in the draw iff hash(entity_id) < frac
    (deterministic, country-agnostic)."""
    h = pd.util.hash_array(np.asarray(entity_ids, dtype=object)).astype(np.float64) / 2.0 ** 64
    return h < frac


def country_pass(data, stats, c_, in_draw, s1_names, s2_model, s1_model):
    """Density-matched stage 1 -> stage 2 for one country.  Returns (union frame with final P,
    competitor-count arrays before/after for round-1 pairs)."""
    s1_cty = data["source1"].country.values
    # ---- round 1 (full columns, original order) and round 2 of this country
    c1, p1_old, _ = R.round1("test", full=True)
    i1 = np.flatnonzero(s1_cty[c1.s1_pos.values] == c_)
    c1 = c1.iloc[i1].reset_index(drop=True)
    p1_old = p1_old[i1]
    r2 = pd.read_parquet(R.cp("test_r2_pairs.parquet"))
    i2 = np.flatnonzero(s1_cty[r2.s1_pos.values] == c_)
    r2 = r2.iloc[i2].reset_index(drop=True)
    p2_old = np.load(R.cp("test_r2_p1.npy"))[i2]
    base2 = np.load(R.cp("test_r2_base.npy"))[i2]
    # ---- stage-1 context: base scores over the full round-1 set; matched record-side columns
    cc = PL.base_scores(c1, data, stats)
    key1 = (cc.src.values.astype(np.int64) << 23) | cc.r_pos.values
    n_before = cc.ctx_n_rec.values.copy()
    rk, gp, nn = stage1_ctx(key1, cc.s1_pos.values, cc.base.values.astype(np.float64), in_draw)
    cc["ctx_rank_rec"], cc["ctx_gap_rec"], cc["ctx_n_rec"] = rk, gp, nn
    key2 = (r2.src.values.astype(np.int64) << 23) | r2.r_pos.values
    ukey = np.r_[key1, key2]
    us1 = np.r_[cc.s1_pos.values, r2.s1_pos.values]
    ubase = np.r_[cc.base.values, base2].astype(np.float64)
    rk_u, gp_u, nn_u = stage1_ctx(ukey, us1, ubase, in_draw)            # round 2 competes with the union
    n1 = len(cc)
    # ---- stage-1 re-scoring of pairs that can matter (original stage-1 P >= 1e-4)
    sel1 = np.flatnonzero(p1_old >= 1e-4)
    X1a = np.empty((len(sel1), len(s1_names)), np.float32)
    for b in range(0, len(sel1), R.CHUNK):
        X1a[b:b + R.CHUNK] = PL.build_features(cc.iloc[sel1[b:b + R.CHUNK]], data, stats)[s1_names].values
    del cc
    sel2 = np.flatnonzero(p2_old >= 1e-4)
    X2a = np.asarray(np.load(R.cp("test_r2_feats.npy"), mmap_mode="r")[i2[sel2]])
    col = {n: s1_names.index(n) for n in ("ctx_rank_rec", "ctx_gap_rec", "ctx_n_rec")}
    X2a[:, col["ctx_rank_rec"]] = rk_u[n1 + sel2]
    X2a[:, col["ctx_gap_rec"]] = gp_u[n1 + sel2]
    X2a[:, col["ctx_n_rec"]] = nn_u[n1 + sel2]
    p1 = p1_old.copy()
    p2 = p2_old.copy()
    for X, sel, tgt in ((X1a, sel1, p1), (X2a, sel2, p2)):
        for b in range(0, len(sel), R.CHUNK):
            tgt[sel[b:b + R.CHUNK]] = s1_model.predict(X[b:b + R.CHUNK])
    log(f"  {c_}: stage 1 re-scored {len(sel1)} round-1 + {len(sel2)} round-2 pairs")
    # ---- stage 2 on the union with matched record-side context
    u = pd.DataFrame({"s1_pos": us1, "src": np.r_[c1.src.values, r2.src.values],
                      "r_pos": np.r_[c1.r_pos.values, r2.r_pos.values], "p": np.r_[p1, p2].astype(np.float32),
                      "is_round2": np.r_[np.zeros(n1, np.float32), np.ones(len(r2), np.float32)]})
    rows = np.flatnonzero(np.r_[p1 >= P_MIN, p2 >= R.R2_MIN])
    q = np.flatnonzero(u.p.values >= 1e-4)
    cq = PL.stage2_context(u.iloc[q][["s1_pos", "src", "r_pos"]].reset_index(drop=True), u.p.values[q])
    m = stage2_rec_ctx(ukey[q], us1[q], u.p.values[q].astype(np.float64), in_draw)
    for k, v in m.items():
        cq[k] = v.astype(np.float32)
    ctxv = np.full((len(rows), cq.shape[1]), np.nan, np.float32)
    hit = np.isin(rows, q)
    ctxv[hit] = cq.values[np.searchsorted(q, rows[hit])]
    ctx_cols = list(cq.columns)
    del cq
    d = u.iloc[rows].reset_index(drop=True)
    sibf = sibling_features(d, {k: data[f"source{k}"][["core", "addr", "nums"]] for k in (2, 3)},
                            data["source1"][["core", "addr", "nums"]])
    nf = R.name_freq(data, d)
    # stage-1 feature rows of the stage-2 rows (every stage-2 row was re-scored: P >= 0.01 needs P_old >= 1e-4
    # or a matched-context jump from below 1e-4, which the selection excludes by construction)
    pos1 = np.full(n1, -1, np.int64)
    pos1[sel1] = np.arange(len(sel1))
    pos2 = np.full(len(r2), -1, np.int64)
    pos2[sel2] = np.arange(len(sel2))
    r1r, r2r = rows[rows < n1], rows[rows >= n1] - n1
    assert (pos1[r1r] >= 0).all() and (pos2[r2r] >= 0).all()
    Xs1 = np.vstack([X1a[pos1[r1r]], X2a[pos2[r2r]]])
    del X1a, X2a
    X = np.hstack([Xs1, ctxv, sibf.values, nf.values, d[["is_round2"]].values]).astype(np.float32)
    names = s1_names + ctx_cols + list(sibf.columns) + list(nf.columns) + ["is_round2"]
    assert s2_model.feature_name() == names, "stage-2 feature layout differs from the trained model"
    p = u.p.values.copy()
    for b in range(0, len(rows), R.CHUNK):
        p[rows[b:b + R.CHUNK]] = s2_model.predict(X[b:b + R.CHUNK])
    log(f"  {c_}: stage 2 re-scored {len(rows)} rows")
    return u[["s1_pos", "src", "r_pos"]].assign(p=p), n_before, nn[:n1] if len(nn) > n1 else nn


def main():
    """Single fixed hash draw (fraction --frac), per country, decide with run-10 thresholds, write."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=KEEP_FRAC)
    ap.add_argument("--out", default="runs/v4_dm")
    ap.add_argument("--check", default="", help="country: run only it and compare P with run 10 (use --frac 1)")
    args = ap.parse_args()
    data = load_split(os.path.join(R.ROOT, "dataset"), "test", R.CACHE)
    stats = TextStats(seed=PL.CFG["seed"]).fit([data["source1"], data["source2"], data["source3"]])
    in_draw = in_draw_mask(data["source1"].entity_id.values, args.frac)
    log(f"hash draw: {in_draw.mean():.4f} of test S1s (target fraction {args.frac})")
    s1_names = R.stage1_names()
    s1_model = lgb.Booster(model_file=os.path.join(R.CACHE, "v2_model_stage1.txt"))
    s2_model = lgb.Booster(model_file=R.cp("stage2_model.txt"))
    with open(R.cp("train_result.json")) as fh:
        thr = json.load(fh)["full40"]
    parts, comp = [], {}
    if args.check:
        u, _, _ = country_pass(data, stats, args.check, in_draw, s1_names, s2_model, s1_model)
        ref = pd.DataFrame({"s1_pos": 0, "src": 0, "r_pos": 0}, index=[])
        r1, _, _ = R.round1("test")
        r2 = pd.read_parquet(R.cp("test_r2_pairs.parquet"))
        ref = pd.concat([r1[["s1_pos", "src", "r_pos"]], r2], ignore_index=True)
        ref = ref.sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True).assign(p=np.load(R.cp("test_prob.npy")))
        j = u.merge(ref, on=["s1_pos", "src", "r_pos"], suffixes=("", "_r10"))
        d = np.abs(j.p.values - j.p_r10.values)
        log(f"CHECK {args.check}: {len(u)} pairs, matched {len(j)}; |P - P_run10| max {d.max():.2e}, "
            f"mean {d.mean():.2e}; pairs crossing t_add {int(((j.p >= thr['t_add']) != (j.p_r10 >= thr['t_add'])).sum())}")
        return
    for c_ in sorted(x for x in data["source1"].country.unique() if x):
        u, nb, na = country_pass(data, stats, c_, in_draw, s1_names, s2_model, s1_model)
        parts.append(u)
        comp[c_] = {"pair_mean_before": float(nb.mean()), "pair_mean_after": float(na.mean()),
                    "pair_median_before": float(np.median(nb)), "pair_median_after": float(np.median(na)),
                    "p90_before": float(np.quantile(nb, 0.9)), "p90_after": float(np.quantile(na, 0.9))}
        log(f"  {c_} competitor count (round-1 pairs, ctx_n_rec): {comp[c_]}")
    u = pd.concat(parts, ignore_index=True).sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True)
    np.save(os.path.join(R.CACHE, "dm_test_prob.npy"), u.p.values)
    s1_pos, src, r_pos = (u[k].values.astype(np.int64) for k in ("s1_pos", "src", "r_pos"))
    keep = Decider(s1_pos, (src << 23) | r_pos, u.p.values.astype(np.float64)).keep(thr["t_first"], thr["t_add"])
    s1 = data["source1"]
    ids = {k: np.asarray(data[f"source{k}"].entity_id.values, dtype=object) for k in (2, 3)}
    s1_ids = np.asarray(s1.entity_id.values, dtype=object)
    out = os.path.join(R.ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    PL.write_lists(os.path.join(out, "candidate_pairs.tsv"), s1_ids, s1_pos, src, r_pos, ids[2], ids[3],
                   "candidate_entity_ids")
    PL.write_lists(os.path.join(out, "matching_results.tsv"), s1_ids, s1_pos[keep], src[keep], r_pos[keep],
                   ids[2], ids[3], "matched_entity_ids")
    cty = np.asarray(s1.country.values, dtype=object)
    kept = s1_pos[keep]
    summ = {"frac": args.frac, "t_first": thr["t_first"], "t_add": thr["t_add"], "pred_pairs": int(keep.sum()),
            "competitor_counts": comp,
            "pred_singleton_fraction_by_country": {k: float(1 - len(np.unique(kept[cty[kept] == k])) / (cty == k).sum())
                                                   for k in np.unique(cty)}}
    with open(os.path.join(out, "test_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2)
    log(f"done: {summ}")


if __name__ == "__main__":
    main()
