"""Average the final probabilities of run 10 and its density-matched variant (same candidate union, same
sorted order), decide with the run-10 thresholds, write runs/v4_blend.  Test-time only, no retraining.

python blend.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as PL  # noqa: E402
import run10 as R  # noqa: E402
from postprocess import Decider  # noqa: E402


def main():
    """Rebuild the sorted union keys, check the alignment with run 10's decision, blend, write."""
    c1, _, _ = R.round1("test")
    r2 = pd.read_parquet(R.cp("test_r2_pairs.parquet"))
    u = pd.concat([c1[["s1_pos", "src", "r_pos"]], r2], ignore_index=True)
    u = u.sort_values(["s1_pos", "src", "r_pos"]).reset_index(drop=True)
    p10 = np.load(R.cp("test_prob.npy"))
    pdm = np.load(os.path.join(R.CACHE, "dm_test_prob.npy"))
    assert len(u) == len(p10) == len(pdm)
    with open(R.cp("train_result.json")) as fh:
        thr = json.load(fh)["full40"]
    s1_pos, src, r_pos = (u[k].values.astype(np.int64) for k in ("s1_pos", "src", "r_pos"))
    rid = (src << 23) | r_pos
    # alignment check: run 10's own probabilities must reproduce run 10's pair count exactly
    k10 = Decider(s1_pos, rid, p10.astype(np.float64)).keep(thr["t_first"], thr["t_add"])
    print(f"run 10 re-decided: {int(k10.sum())} pairs (runs/v4 has 5554870)", flush=True)
    assert int(k10.sum()) == 5554870, "union order does not match run 10"
    p = (p10.astype(np.float64) + pdm.astype(np.float64)) / 2
    keep = Decider(s1_pos, rid, p).keep(thr["t_first"], thr["t_add"])
    s1 = pd.read_parquet(os.path.join(R.CACHE, "test_source1.parquet"), columns=["entity_id"])
    ids = {k: np.asarray(pd.read_parquet(os.path.join(R.CACHE, f"test_source{k}.parquet"), columns=["entity_id"])
                         .entity_id.values, dtype=object) for k in (2, 3)}
    s1_ids = np.asarray(s1.entity_id.values, dtype=object)
    out = os.path.join(R.OUT_ROOT, "runs", "v4_blend")
    os.makedirs(out, exist_ok=True)
    PL.write_lists(os.path.join(out, "matching_results.tsv"), s1_ids, s1_pos[keep], src[keep], r_pos[keep],
                   ids[2], ids[3], "matched_entity_ids")
    print(f"blend: {int(keep.sum())} pairs written to {out}", flush=True)


if __name__ == "__main__":
    main()
