"""Leaderboard probe (Stage 5 variant, no retraining): rewrite the matching file from the cached
test probabilities, shifting t_first for one country label (or ALL).

Used to test the hypothesis that the unseen country (France) needs a stricter first-match bar.
It reads only cached artefacts (cache/test_cands.parquet, cache/test_prob.npy) and the thresholds
in output/run_summary.json; it never retrains and never reads test labels (there are none).

python probe_france.py --country France --delta 0.05 --out submissions/sub-probe-france.tsv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import write_lists  # noqa: E402
from postprocess import Decider  # noqa: E402


def load_scored_pairs(cache, tag=""):
    """Return (s1_pos, src, r_pos, prob) in the exact order predict_test scored them
    (candidates sorted by s1_pos, src, r_pos; pairs are unique so the order is deterministic)."""
    c = pd.read_parquet(os.path.join(cache, "test_cands.parquet"), columns=["s1_pos", "src", "r_pos"])
    c = c.sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True)
    prob = np.load(os.path.join(cache, f"{tag + '_' if tag else ''}test_prob.npy"))
    assert len(prob) == len(c), "cached probabilities do not match the candidate set"
    return c.s1_pos.values.astype(np.int64), c.src.values.astype(np.int64), c.r_pos.values.astype(np.int64), prob


def country_probe_keep(s1_pos, src, r_pos, prob, s1_country, country, tf, ta, delta):
    """Keep mask: global (t_first, t_add) everywhere, except S1s whose country == `country`
    use (t_first + delta, t_add).  One-owner assignment is identical in both variants."""
    dec = Decider(s1_pos, (src << 23) | r_pos, prob)
    keep_main = dec.keep(tf, ta)
    keep_probe = dec.keep(tf + delta, ta)
    in_country = np.ones(len(s1_pos), bool) if country == "ALL" else s1_country[s1_pos] == country
    return np.where(in_country, keep_probe, keep_main), keep_main


def main():
    """CLI entry point: write the probe matching file and print what changed."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--summary", default="output/run_summary.json")
    ap.add_argument("--country", default="France", help="country label, or ALL for every S1")
    ap.add_argument("--tag", default="", help="use cache/<tag>_test_prob.npy (e.g. v2 = run 8)")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--out", default="submissions/sub-probe-france.tsv")
    args = ap.parse_args()
    cv = json.load(open(args.summary))["cv"]
    tf, ta = cv["t_first"], cv["t_add"]
    s1 = pd.read_parquet(os.path.join(args.cache, "test_source1.parquet"), columns=["entity_id", "country"])
    s1_ids = np.asarray(s1.entity_id.values, dtype=object)
    s1_country = np.asarray(s1.country.values, dtype=object)
    ids2 = np.asarray(pd.read_parquet(os.path.join(args.cache, "test_source2.parquet"), columns=["entity_id"]).entity_id.values, dtype=object)
    ids3 = np.asarray(pd.read_parquet(os.path.join(args.cache, "test_source3.parquet"), columns=["entity_id"]).entity_id.values, dtype=object)
    s1_pos, src, r_pos, prob = load_scored_pairs(args.cache, args.tag)
    keep, keep_main = country_probe_keep(s1_pos, src, r_pos, prob, s1_country, args.country, tf, ta, args.delta)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_lists(args.out, s1_ids, s1_pos[keep], src[keep], r_pos[keep], ids2, ids3, "matched_entity_ids")
    n_main = np.bincount(s1_pos, weights=keep_main, minlength=len(s1_ids)) > 0
    n_probe = np.bincount(s1_pos, weights=keep, minlength=len(s1_ids)) > 0
    in_c = np.ones(len(s1_ids), bool) if args.country == "ALL" else s1_country == args.country
    print(f"t_first {tf:.4f} -> {tf + args.delta:.4f} for {args.country} ({in_c.sum()} S1 rows); t_add {ta:.4f} unchanged")
    print(f"{args.country} rows non-empty -> empty: {(n_main & ~n_probe & in_c).sum()}")
    print(f"{args.country} rows otherwise changed: {((n_main != n_probe) & in_c).sum() - (n_main & ~n_probe & in_c).sum()}")
    print(f"non-{args.country} rows changed: {((n_main != n_probe) & ~in_c).sum()}; pairs removed: {int(keep_main.sum() - keep.sum())}")


if __name__ == "__main__":
    main()
