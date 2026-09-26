"""Test inference for the stage-2 sibling-agreement model (run 9 = run 8 stage 1 + stage 2).

Reuses the run-8 stage-1 test probabilities (cache/v2_test_prob.npy, aligned with the sorted
test candidate set), recomputes stage-1 features only for pairs with stage-1 P >= P_MIN, adds the
rank/gap context and the sibling features, re-scores those pairs with cache/v3_stage2_model.txt,
and writes both output files to --out.  Thresholds come from the stage-2 OOF tuning.

Context approximation: stage2_context is computed over pairs with stage-1 P >= 1e-4 instead of
the full 64M-pair set (memory).  Max / gap / rank / n_hi are unchanged by dropping lower-P pairs;
the P sums and best-other values shift by at most the dropped mass (< 1e-4 per pair).

python predict_stage2.py --out runs/v3 --t-first 0.533 --t-add 0.794

Optional (prepared, NOT run): --stage1-folds replaces the run-8 refit stage-1 probabilities with the
average of the five fold models (cache/v2_stage1_fold{0..4}.txt, see train_fold_models.py) on every
pair whose refit P >= FOLD_MIN, so stage 2 sees stage-1 scores distributed like its OOF training
input.  Pairs below FOLD_MIN keep the refit P (they are far below any threshold).  Stage-2
thresholds are unchanged.  Use only if run 9's LB gain over run 2 is < +0.003.
python predict_stage2.py --stage1-folds --out runs/v4 --prob-out v4_test_prob.npy --t-first 0.533 --t-add 0.794
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
from pipeline import CFG, base_scores, build_features, load_split, stage2_context, write_lists  # noqa: E402
from features import TextStats  # noqa: E402
from postprocess import Decider  # noqa: E402
from stage2_sibling import P_MIN, sibling_features  # noqa: E402

FOLD_MIN = 1e-3
T0 = time.time()


def log(msg):
    """Timestamped progress line."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def main():
    """Stage-1 features for live test pairs -> stage-2 features -> stage-2 P -> decision -> TSVs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="runs/v3")
    ap.add_argument("--t-first", type=float, required=True)
    ap.add_argument("--t-add", type=float, required=True)
    ap.add_argument("--stage1-folds", action="store_true", help="average the 5 stage-1 fold models (prepared option)")
    ap.add_argument("--prob-out", default="v3_test_prob.npy")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    te = load_split(args.data_dir, "test", args.cache)
    ct = pd.read_parquet(os.path.join(args.cache, "test_cands.parquet"))
    ct = ct.sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True)
    p1 = np.load(os.path.join(args.cache, "v2_test_prob.npy")).astype(np.float32)
    assert len(p1) == len(ct), "run-8 test probabilities do not match the candidate set"
    stats = TextStats(seed=CFG["seed"]).fit([te["source1"], te["source2"], te["source3"]])
    ct = base_scores(ct, te, stats)                       # context columns need the full candidate set
    model = lgb.Booster(model_file=os.path.join(args.cache, "v3_stage2_model.txt"))
    names = model.feature_name()
    stage1_names = lgb.Booster(model_file=os.path.join(args.cache, "v2_model_stage1.txt")).feature_name()
    feat_rows = np.flatnonzero(p1 >= (FOLD_MIN if args.stage1_folds else P_MIN))
    parts = []
    for a in range(0, len(feat_rows), 1_000_000):
        r = feat_rows[a:a + 1_000_000]
        parts.append(build_features(ct.iloc[r], te, stats)[stage1_names].values.astype(np.float32))
        log(f"  stage-1 features {a + len(r)}/{len(feat_rows)}")
    F = np.vstack(parts)
    del parts
    if args.stage1_folds:
        folds = [lgb.Booster(model_file=os.path.join(args.cache, f"v2_stage1_fold{f}.txt")) for f in range(5)]
        p1[feat_rows] = np.mean([m.predict(F) for m in folds], axis=0).astype(np.float32)
        log(f"stage-1 fold average on {len(feat_rows)} pairs with refit P >= {FOLD_MIN}")
    rows = np.flatnonzero(p1 >= P_MIN)
    if args.stage1_folds:
        rows = rows[np.isin(rows, feat_rows)]            # pairs below FOLD_MIN keep refit P < P_MIN
    X1 = F[np.searchsorted(feat_rows, rows)]
    del F
    log(f"{len(ct)} test pairs, {len(rows)} with stage-1 P >= {P_MIN}")
    keys = ct[["s1_pos", "src", "r_pos"]].copy()
    del ct

    q = np.flatnonzero(p1 >= 1e-4)
    ctx_q = stage2_context(keys.iloc[q].reset_index(drop=True), p1[q])
    pos_in_q = np.searchsorted(q, rows)
    ctx = ctx_q.iloc[pos_in_q].reset_index(drop=True)
    del ctx_q
    log("context done")

    d = keys.iloc[rows].reset_index(drop=True).assign(p=p1[rows])
    rec = {2: te["source2"][["core", "addr", "nums"]], 3: te["source3"][["core", "addr", "nums"]]}
    sibf = sibling_features(d, rec, te["source1"][["core", "addr", "nums"]])
    log("sibling features done")

    X = np.hstack([X1, ctx.values, sibf.values]).astype(np.float32)
    assert X.shape[1] == len(names) and list(ctx.columns) + list(sibf.columns) == names[len(stage1_names):], \
        "stage-2 feature layout differs from the trained model"
    p = p1.copy()
    p[rows] = model.predict(X)
    np.save(os.path.join(args.cache, args.prob_out), p)
    log("stage-2 predicted")

    s1_ids = np.asarray(te["source1"].entity_id.values, dtype=object)
    cty = np.asarray(te["source1"].country.values, dtype=object)
    ids2 = np.asarray(te["source2"].entity_id.values, dtype=object)
    ids3 = np.asarray(te["source3"].entity_id.values, dtype=object)
    s1_pos = keys.s1_pos.values.astype(np.int64)
    src = keys.src.values.astype(np.int64)
    r_pos = keys.r_pos.values.astype(np.int64)
    keep = Decider(s1_pos, (src << 23) | r_pos, p).keep(args.t_first, args.t_add)
    write_lists(os.path.join(args.out, "candidate_pairs.tsv"), s1_ids, s1_pos, src, r_pos, ids2, ids3,
                "candidate_entity_ids")
    write_lists(os.path.join(args.out, "matching_results.tsv"), s1_ids, s1_pos[keep], src[keep], r_pos[keep],
                ids2, ids3, "matched_entity_ids")
    kept = s1_pos[keep]
    n1 = len(s1_ids)
    summary = {"t_first": args.t_first, "t_add": args.t_add, "pred_pairs": int(keep.sum()),
               "pred_singleton_fraction": float(1 - len(np.unique(kept)) / n1),
               "pred_singleton_fraction_by_country": {k: float(1 - len(np.unique(kept[cty[kept] == k])) / (cty == k).sum())
                                                      for k in np.unique(cty)},
               "mean_pred_matches_by_country": {k: float((cty[kept] == k).sum() / (cty == k).sum()) for k in np.unique(cty)}}
    with open(os.path.join(args.out, "test_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    log(f"test: {keep.sum()} matched pairs written; {summary}")


if __name__ == "__main__":
    main()
