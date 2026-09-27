"""Build a small copy of dataset/ for the end-to-end smoke test (same TSV format and file names).

Train: a hash-sampled FRAC of Source-1 entities, all their true S2/S3 matches, plus the same
fraction of the remaining S2/S3 records as distractors; ground truth restricted to the sample.
Test: a hash-sampled FRAC of Source-1 entities and the same fraction of S2/S3 records (no labels
exist, so this only exercises the code path).  Every country is kept (hash sampling is
country-agnostic).

python smoke_dataset.py --src dataset --dst dataset_smoke5 --frac 0.05
"""
import argparse
import os

import numpy as np
import pandas as pd


def read(path):
    """Competition TSV reader (all strings, empty stays empty)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write(df, path):
    """TSV writer matching the provided files (single tab, no quoting)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=3)


def sample(ids, frac):
    """Deterministic hash sample of entity ids."""
    h = pd.util.hash_array(np.asarray(ids, dtype=object)).astype(np.float64) / 2.0 ** 64
    return h < frac


def main():
    """Write the sampled train and test splits."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--dst", default="dataset_smoke5")
    ap.add_argument("--frac", type=float, default=0.05)
    a = ap.parse_args()
    # ---- train
    s1 = read(os.path.join(a.src, "train", "train_source1.tsv"))
    gt = read(os.path.join(a.src, "train", "train_ground_truth.tsv"))
    keep1 = sample(s1.entity_id.values, a.frac)
    s1 = s1[keep1]
    gt = gt[gt.source1_entity_id.isin(set(s1.entity_id))]
    true_ids = {i for l in gt.matched_entity_ids.values for i in l.split(",") if i}
    for k in (2, 3):
        r = read(os.path.join(a.src, "train", f"train_source{k}.tsv"))
        r = r[r.entity_id.isin(true_ids).values | sample(r.entity_id.values, a.frac)]
        write(r, os.path.join(a.dst, "train", f"train_source{k}.tsv"))
    write(s1, os.path.join(a.dst, "train", "train_source1.tsv"))
    write(gt, os.path.join(a.dst, "train", "train_ground_truth.tsv"))
    print(f"train: {len(s1)} S1, {len(true_ids)} true records")
    # ---- test
    for k in (1, 2, 3):
        t = read(os.path.join(a.src, "test", f"test_source{k}.tsv"))
        t = t[sample(t.entity_id.values, a.frac)]
        write(t, os.path.join(a.dst, "test", f"test_source{k}.tsv"))
        print(f"test source{k}: {len(t)} rows")


if __name__ == "__main__":
    main()
