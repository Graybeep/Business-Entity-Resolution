"""I/O helpers and cached, parallel normalisation of the three sources."""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from normalize import normalize_record

NORM_COLS = ["name", "core", "variants", "addr", "nums", "postal", "phon"]


def read_tsv(path):
    # quoting=3 (QUOTE_NONE): fields are never quoted and may contain stray '"'
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def _norm_chunk(args):
    names, addrs = args
    return [normalize_record(n, a) for n, a in zip(names, addrs)]


def normalize_frame(df, workers):
    n = len(df)
    step = 20000
    chunks = [(df.business_name.values[i:i + step].tolist(), df.business_address.values[i:i + step].tolist())
              for i in range(0, n, step)]
    with Pool(workers) as pool:
        res = pool.map(_norm_chunk, chunks, chunksize=1)
    rows = [r for c in res for r in c]
    out = pd.DataFrame(rows, columns=NORM_COLS, index=df.index)
    return pd.concat([df, out], axis=1)


def load_split(data_dir, split, cache_dir, workers=14):
    """Returns dict source -> normalised DataFrame (cached as parquet)."""
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    for src in ("source1", "source2", "source3"):
        cp = os.path.join(cache_dir, f"{split}_{src}.parquet")
        if os.path.exists(cp):
            out[src] = pd.read_parquet(cp)
            continue
        df = read_tsv(os.path.join(data_dir, split, f"{split}_{src}.tsv"))
        df = normalize_frame(df, workers)
        df.to_parquet(cp, index=False)
        out[src] = df
        print(f"  normalised {split}/{src}: {len(df)} rows", flush=True)
    return out


def load_ground_truth(data_dir):
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    lists = gt.matched_entity_ids.str.split(",").map(lambda x: [i for i in x if i])
    pairs = pd.DataFrame({"s1": np.repeat(gt.source1_entity_id.values, lists.map(len).values),
                          "rid": [i for l in lists for i in l]})
    return gt, pairs


if __name__ == "__main__":
    import sys
    import time
    data_dir, cache_dir = sys.argv[1], sys.argv[2]
    for split in ("train", "test"):
        t = time.time()
        load_split(data_dir, split, cache_dir)
        print(split, "done", round(time.time() - t), "s", flush=True)
