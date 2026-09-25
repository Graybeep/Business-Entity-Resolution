"""Stage 2 - candidate generation.

Several exact sparse TF-IDF cosine top-k passes (sparse_dot_topn, multi-threaded C++),
each vectoriser fitted on a sample of the split being processed (provided data only).
Very frequent terms are dropped (max_df) - they carry no identity signal and dominate cost.
Per S1 and per source (S2, S3 separately), the union of:
  na    word tokens of core name + address          top-20   (main pass)
  c4    char_wb 4-grams of core name + address      top-10   (typo backstop)
  ph    phonetic name skeleton + address words      top-10   (transliterated Indic names)
  nm    word tokens of core name only               top-10   (records with empty address)
  rev   each S2/S3 record -> its top-3 S1s by `na`  (reverse pass)
Blocking is within country label (0 cross-country true pairs in train); a record with an
empty country joins every pool.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

PASSES = {
    "na": dict(text=lambda d: d["core"] + " " + d["addr"], analyzer="word", ngram=(1, 1), max_df=0.01),
    "c4": dict(text=lambda d: d["core"] + " " + d["addr"], analyzer="char_wb", ngram=(4, 4), max_df=0.002),
    "ph": dict(text=lambda d: d["phon"] + " " + d["addr"], analyzer="word", ngram=(1, 1), max_df=0.01),
    "nm": dict(text=lambda d: d["core"], analyzer="word", ngram=(1, 1), max_df=0.01),
}


def _vectorizer(spec, seed, texts, fit_n=1_500_000):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), min(fit_n, len(texts)), replace=False)
    vec = TfidfVectorizer(analyzer=spec["analyzer"], ngram_range=spec["ngram"], min_df=2, max_df=spec["max_df"],
                          sublinear_tf=True, dtype=np.float32, token_pattern=r"[a-z0-9]+")
    return vec.fit(texts[idx])


def par_transform(vec, texts, n_jobs=14, step=200_000):
    """vec.transform in parallel chunks (sklearn's transform is single-threaded)."""
    if len(texts) <= step:
        return vec.transform(texts)
    parts = Parallel(n_jobs=n_jobs)(delayed(vec.transform)(texts[i:i + step]) for i in range(0, len(texts), step))
    return sp.vstack(parts).tocsr()


def topk(Q, T, k, threads):
    """Exact top-k cosine (rows are L2-normalised).  Returns COO (row, col, sim)."""
    if Q.shape[0] == 0 or T.shape[0] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    C = sp_matmul_topn(Q, T.T.tocsr(), top_n=min(k, T.shape[0]), threshold=1e-6, n_threads=threads).tocoo()
    return C.row, C.col, C.data.astype(np.float32)


def pools(s1, other):
    """Yield (country, s1 positions, other positions); empty country joins every pool."""
    c1 = s1.country.values
    c2 = other.country.values
    labels = sorted(set(c1[c1 != ""]) | set(c2[c2 != ""]))
    e1 = np.flatnonzero(c1 == "")
    e2 = np.flatnonzero(c2 == "")
    if not labels:
        yield "", np.arange(len(s1)), np.arange(len(other))
        return
    for lab in labels:
        yield lab, np.union1d(np.flatnonzero(c1 == lab), e1), np.union1d(np.flatnonzero(c2 == lab), e2)


def pair_key(s1_pos, src, r_pos):
    return (np.asarray(s1_pos, np.int64) << 24) | (np.int64(src == 3) << 23) | np.asarray(r_pos, np.int64)


def split_key(key):
    return ((key >> 24).astype(np.int32), np.where((key >> 23) & 1, 3, 2).astype(np.int8),
            (key & ((1 << 23) - 1)).astype(np.int32))


def _rank_within(row, sim):
    """Rank (0 = best) of each hit within its query row."""
    order = np.lexsort((-sim, row))
    r = row[order]
    start = np.r_[0, np.flatnonzero(r[1:] != r[:-1]) + 1]
    rank = np.arange(len(r)) - np.repeat(start, np.diff(np.r_[start, len(r)]))
    out = np.empty(len(r), np.int64)
    out[order] = rank
    return np.minimum(out, 254).astype(np.uint8)


class _Acc:
    def __init__(self):
        self.parts = []

    def add(self, key, rank, sim):
        self.parts.append((key, rank, sim))

    def finish(self):
        key = np.concatenate([p[0] for p in self.parts])
        rank = np.concatenate([p[1] for p in self.parts])
        sim = np.concatenate([p[2] for p in self.parts])
        order = np.lexsort((rank, key))
        key, rank, sim = key[order], rank[order], sim[order]
        first = np.r_[True, key[1:] != key[:-1]]
        return key[first], rank[first], sim[first]


def generate_candidates(s1, s2, s3, cfg, log=print, s1_mask=None):
    """Returns DataFrame of unique pairs (s1_pos, src 2/3, r_pos) with, per pass, the rank of
    the pair in that pass (255 = not retrieved) and its cosine (NaN = not retrieved).
    s1_mask (optional, training only): forward passes run for these S1s only, and only their
    pairs are returned; the reverse pass still ranks against ALL S1s so that its rank reflects
    the real competition."""
    if s1_mask is None:
        s1_mask = np.ones(len(s1), bool)
    assert len(s2) < (1 << 23) and len(s3) < (1 << 23)
    k, rev_k, threads, seed = cfg["k"], cfg["rev_k"], cfg.get("threads", 14), cfg.get("seed", 0)
    passes = {}
    for name, spec in PASSES.items():
        if k.get(name, 0) <= 0:
            continue
        t = time.time()
        txt = {1: spec["text"](s1).values, 2: spec["text"](s2).values, 3: spec["text"](s3).values}
        vec = _vectorizer(spec, seed, np.concatenate([txt[1], txt[2], txt[3]]))
        X = {i: par_transform(vec, v) for i, v in txt.items()}
        del txt
        log(f"  [{name}] vectorised in {time.time() - t:.0f}s (vocab {len(vec.vocabulary_)})")
        acc, racc = _Acc(), _Acc()
        for src, other in ((2, s2), (3, s3)):
            for lab, p1, p2 in pools(s1, other):
                t = time.time()
                q1 = p1[s1_mask[p1]]
                r, c, v = topk(X[1][q1], X[src][p2], k[name], threads)
                acc.add(pair_key(q1[r], src, p2[c]), _rank_within(r, v), v)
                if name == "na" and rev_k > 0:
                    r, c, v = topk(X[src][p2], X[1][p1], rev_k, threads)
                    rank = _rank_within(r, v)
                    ok = s1_mask[p1[c]]
                    racc.add(pair_key(p1[c[ok]], src, p2[r[ok]]), rank[ok], v[ok])
                log(f"    [{name}] src{src} {lab!r}: {len(p1)} x {len(p2)} in {time.time() - t:.0f}s")
        passes[name] = acc.finish()
        if racc.parts:
            passes["rev"] = racc.finish()
        del X
    keys = np.unique(np.concatenate([v[0] for v in passes.values()]))
    s1_pos, src, r_pos = split_key(keys)
    out = pd.DataFrame({"s1_pos": s1_pos, "src": src, "r_pos": r_pos})
    for name, (pk, pr, ps) in passes.items():
        rank = np.full(len(keys), 255, np.uint8)
        sim = np.full(len(keys), np.nan, np.float32)
        loc = np.searchsorted(keys, pk)
        rank[loc] = pr
        sim[loc] = ps
        out[f"rank_{name}"] = rank
        out[f"sim_{name}"] = sim
    return out
