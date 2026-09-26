"""Train-vs-test shift diagnostics for US + India (no test labels are used or exist).

1. Adversarial validation: ~200k random candidate pairs each from train (run-8 feature store, 40 %
   S1 subset) and test (stage-1 features recomputed with full-country context), LightGBM predicting
   test-vs-train from the stage-1 features; 5-fold AUC + top-10 features by gain.
2. Per country, train (OOF) vs test (run-8 refit): candidates per S1, candidate S1s per record,
   top-1 P, top1-top2 gap, near-tie rate, distribution of the record-side context rank.

python shift_diag.py   (log: runs/shift_diag.log)
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL  # noqa: E402
from data import load_split  # noqa: E402
from features import TextStats  # noqa: E402
from training_scaled import load_rows  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE = os.path.join(ROOT, "cache")
N = 100_000          # pairs per country per split
T0 = time.time()


def log(msg):
    """Timestamped progress line."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def per_s1_stats(s1_pos, src, r_pos, p):
    """Per-S1 top-1 P and top1-top2 gap; per-record candidate-S1 count and near-tie flag."""
    d = pd.DataFrame({"s": s1_pos, "r": (src.astype(np.int64) << 23) | r_pos, "p": p})
    s = d.sort_values(["s", "p"], ascending=[True, False])
    first = np.r_[True, s.s.values[1:] != s.s.values[:-1]]
    second = np.r_[False, first[:-1]] & ~first
    t1 = pd.Series(s.p.values[first], index=s.s.values[first])
    t2 = pd.Series(s.p.values[second], index=s.s.values[second]).reindex(t1.index).fillna(0)
    r = d.sort_values(["r", "p"], ascending=[True, False])
    rf = np.r_[True, r.r.values[1:] != r.r.values[:-1]]
    rs = np.r_[False, rf[:-1]] & ~rf
    r1 = pd.Series(r.p.values[rf], index=r.r.values[rf])
    r2 = pd.Series(r.p.values[rs], index=r.r.values[rs]).reindex(r1.index).fillna(0)
    live = r1.values >= 0.5
    return {"cands_per_s1": len(d) / d.s.nunique(), "cand_s1_per_record": len(d) / d.r.nunique(),
            "top1_p_median": float(np.median(t1.values)), "top1_p<0.5": float((t1.values < 0.5).mean()),
            "top1_top2_gap_median": float(np.median(t1.values - t2.values)),
            "near_tie_rate(rec top>=0.5, gap<0.05)": float(((r1.values - r2.values) < 0.05)[live].mean())}


def main():
    """Sample pairs, rebuild test features, adversarial AUC, per-country stats."""
    names = lgb.Booster(model_file=os.path.join(CACHE, "v2_model_stage1.txt")).feature_name()
    rng = np.random.default_rng(0)
    # ---- train side (run-8 store, row order = v2_oof_pairs)
    oof = pd.read_parquet(os.path.join(CACHE, "v2_oof_pairs.parquet"))
    s1_tr = pd.read_parquet(os.path.join(CACHE, "train_source1.parquet"), columns=["country"]).country.values
    cty_tr = s1_tr[oof.s1_pos.values]
    tr_rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(cty_tr == c), N, replace=False) for c in ("India", "US")]))
    Xtr = load_rows(os.path.join(CACHE, "v2_train_feats"), tr_rows, len(names))
    stats_rows = {"train": {c: per_s1_stats(*(oof[k].values[cty_tr == c] for k in ("s1_pos", "src", "r_pos")),
                                            oof.p.values[cty_tr == c]) for c in ("India", "US")}}
    ctx_tr = pd.DataFrame(Xtr, columns=names)
    del oof
    log("train sample + stats done")
    # ---- test side: full-country context, features only for the sample
    te = load_split(os.path.join(ROOT, "dataset"), "test", CACHE)
    stats = TextStats(seed=PL.CFG["seed"]).fit([te["source1"], te["source2"], te["source3"]])
    ct = pd.read_parquet(os.path.join(CACHE, "test_cands.parquet")).sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True)
    pte = np.load(os.path.join(CACHE, "v2_test_prob.npy"))
    cty_te = te["source1"].country.values[ct.s1_pos.values]
    stats_rows["test"] = {}
    Xte = []
    for c in ("India", "US"):
        idx = np.flatnonzero(cty_te == c)
        stats_rows["test"][c] = per_s1_stats(ct.s1_pos.values[idx], ct.src.values[idx], ct.r_pos.values[idx], pte[idx])
        cc = PL.base_scores(ct.iloc[idx].reset_index(drop=True), te, stats)
        pick = np.sort(rng.choice(len(idx), N, replace=False))
        Xte.append(PL.build_features(cc.iloc[pick], te, stats)[names].values.astype(np.float32))
        del cc
        log(f"test {c}: sample features done")
    Xte = np.vstack(Xte)
    # ---- adversarial validation
    X = np.vstack([Xtr, Xte])
    y = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    auc, gain = [], np.zeros(len(names))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "verbose": -1, "num_threads": 14,
                       "seed": 0}, lgb.Dataset(X[tr], y[tr], feature_name=names), 200)
        auc.append(roc_auc_score(y[va], m.predict(X[va])))
        gain += m.feature_importance("gain")
    log(f"ADVERSARIAL AUC (US+India, 5-fold) = {np.mean(auc):.4f} +- {np.std(auc):.4f}")
    top = pd.Series(gain, index=names).sort_values(ascending=False)
    print("top-10 features by gain (share):")
    print((top / top.sum()).head(10).round(3).to_string())
    te_df = pd.DataFrame(Xte, columns=names)
    print("\nmedian train vs test for the top-10:")
    for f in top.index[:10]:
        print(f"  {f:22s} train {np.nanmedian(ctx_tr[f]):9.4f}  test {np.nanmedian(te_df[f]):9.4f}  "
              f"NaN-rate train {ctx_tr[f].isna().mean():.3f} test {te_df[f].isna().mean():.3f}")
    print("\nper-country candidate / score statistics (train = 40 % S1 OOF, test = run-8 refit):")
    for c in ("India", "US"):
        for split in ("train", "test"):
            print(f"  {c:6s} {split:5s} " + "  ".join(f"{k}={v:.4f}" for k, v in stats_rows[split][c].items()))
    for f in ("ctx_rank_rec", "ctx_n_rec", "ctx_gap_rec"):
        if f in names:
            print(f"  {f}: train quantiles {np.nanquantile(ctx_tr[f], [.5, .9, .99]).round(3)} "
                  f"test {np.nanquantile(te_df[f], [.5, .9, .99]).round(3)}")


if __name__ == "__main__":
    main()
