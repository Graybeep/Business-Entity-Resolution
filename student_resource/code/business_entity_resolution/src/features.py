"""Stage 3 - pair features.  Missing input field -> NaN (never 0 / -1).  No country one-hot."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class TextStats:
    """Vectorisers fitted on (a sample of) the records of one split - provided data only."""

    def __init__(self, seed=0, fit_n=1_500_000):
        self.seed, self.fit_n = seed, fit_n

    def fit(self, frames):
        rng = np.random.default_rng(self.seed)
        allf = pd.concat([f[["core", "addr", "nums"]] for f in frames], ignore_index=True)
        idx = rng.choice(len(allf), min(self.fit_n, len(allf)), replace=False)
        smp = allf.iloc[idx]
        char = dict(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_features=300_000,
                    sublinear_tf=True, dtype=np.float32)
        self.name_char = TfidfVectorizer(**char).fit(smp.core.values)
        self.addr_char = TfidfVectorizer(**char).fit(smp.addr.values)
        word = dict(token_pattern=r"[a-z0-9]+", min_df=1, dtype=np.float32)
        self.name_word = TfidfVectorizer(**word).fit(smp.core.values)
        self.addr_word = TfidfVectorizer(**word).fit(smp.addr.values)
        # idf vectors for weighted Jaccard (unknown words get the max idf via binary matrices)
        self.num_vec = CountVectorizer(token_pattern=r"\d+", binary=True, dtype=np.float32)
        self.num_vec.fit(np.array(["0 1 2 3 4 5 6 7 8 9"] + list(smp.nums.values)))
        return self


def _rowdot(A, B):
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


def _enc(vec, texts):
    """Transform each distinct text once, then gather rows per pair."""
    uniq, inv = np.unique(texts.astype(object), return_inverse=True)
    return vec.transform(uniq)[inv]


def _weighted_jaccard(A, B, idf):
    """IDF-weighted token Jaccard from two row-aligned TF-IDF matrices (presence only)."""
    A = A.copy(); A.data[:] = 1
    B = B.copy(); B.data[:] = 1
    inter = np.asarray(A.multiply(B) @ idf).ravel()
    union = np.asarray(A @ idf).ravel() + np.asarray(B @ idf).ravel() - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        return (inter / union).astype(np.float32)


CP_WORKERS = -1  # set to 1 inside worker processes


def _cp(scorer, a, b):
    return cpdist(a, b, scorer=scorer, workers=CP_WORKERS, dtype=np.float32) / (100.0 if scorer is not JaroWinkler.normalized_similarity else 1.0)


def _variant_best(va, vb):
    """Best token_set_ratio over DBA/aka variant combinations (only computed where needed)."""
    out = np.empty(len(va), np.float32)
    for i, (x, y) in enumerate(zip(va, vb)):
        xs, ys = x.split("|"), y.split("|")
        out[i] = max(fuzz.token_set_ratio(p, q) for p in xs for q in ys) / 100.0
    return out


def _number_feats(m1, m2):
    """House-number style comparisons between the number sets of two addresses.
    Targets 'twin' distractors whose house number is shifted / truncated."""
    n = len(m1)
    out = np.full((n, 5), np.nan, np.float32)
    for i in range(n):
        a, b = m1[i], m2[i]
        if not a or not b:
            continue
        A = [x for x in a.split() if len(x) >= 2] or a.split()
        B = [x for x in b.split() if len(x) >= 2] or b.split()
        la = max(A, key=lambda x: (len(x), x))
        lb = max(B, key=lambda x: (len(x), x))
        best_ratio, min_rel, prefix = 0.0, 1.0, 0.0
        for x in A:
            xi = int(x[:12])
            for y in B:
                if x == y:
                    best_ratio, min_rel = 1.0, 0.0
                    continue
                best_ratio = max(best_ratio, fuzz.ratio(x, y) / 100.0)
                yi = int(y[:12])
                min_rel = min(min_rel, abs(xi - yi) / max(xi, yi, 1))
                if x.startswith(y) or y.startswith(x):
                    prefix = 1.0
        out[i] = (float(la == lb), float(la in B), best_ratio, min_rel, prefix)
    return out


def _coverage_feats(c1, c2, idf, idf_max):
    """Asymmetric IDF-weighted token coverage between two core names (a token counts as
    covered if an exact or near (ratio >= 80) token exists on the other side).  Twins
    substitute a distinctive token; true matches mostly add / drop generic ones."""
    n = len(c1)
    out = np.full((n, 6), np.nan, np.float32)
    for i in range(n):
        t1, t2 = c1[i].split(), c2[i].split()
        if not t1 or not t2:
            continue
        s2 = set(t2)
        s1 = set(t1)
        res = []
        for src, other, oset in ((t1, t2, s2), (t2, t1, s1)):
            tot = cov = 0.0
            miss_max = 0.0
            n_miss = 0
            for t in src:
                w = idf.get(t, idf_max)
                tot += w
                if t in oset or any(fuzz.ratio(t, o) >= 80 for o in other):
                    cov += w
                else:
                    n_miss += 1
                    miss_max = max(miss_max, w)
            res.extend((cov / tot if tot else np.nan, miss_max, n_miss))
        out[i] = res
    return out


def pair_features(L, R, stats):
    """L, R: DataFrames aligned row-by-row (S1 side, S2/S3 side) with normalised columns.
    Returns a float32 DataFrame of pair features (no context features)."""
    f = {}
    n1, n2 = L.core.values, R.core.values
    a1, a2 = L.addr.values, R.addr.values
    name_miss = (L.core.values == "") | (R.core.values == "")
    addr_miss = (a1 == "") | (a2 == "")

    # --- name
    f["name_tfidf"] = _rowdot(_enc(stats.name_char, n1), _enc(stats.name_char, n2))
    W1, W2 = _enc(stats.name_word, n1), _enc(stats.name_word, n2)
    f["name_word_tfidf"] = _rowdot(W1, W2)
    f["name_ratio"] = _cp(fuzz.ratio, n1, n2)
    f["name_token_sort"] = _cp(fuzz.token_sort_ratio, n1, n2)
    f["name_token_set"] = _cp(fuzz.token_set_ratio, n1, n2)
    f["name_partial"] = _cp(fuzz.partial_ratio, n1, n2)
    f["name_jw"] = _cp(JaroWinkler.normalized_similarity, n1, n2)
    f["name_wjacc"] = _weighted_jaccard(W1, W2, stats.name_word.idf_.astype(np.float32))
    del W1, W2
    ns1 = pd.Series(n1).str.replace(" ", "", regex=False).values
    ns2 = pd.Series(n2).str.replace(" ", "", regex=False).values
    f["name_nospace_ratio"] = _cp(fuzz.ratio, ns1, ns2)
    f["name_nospace_partial"] = _cp(fuzz.partial_ratio, ns1, ns2)
    f["fullname_token_set"] = _cp(fuzz.token_set_ratio, L.name.values, R.name.values)
    f["name_phon_ratio"] = _cp(fuzz.ratio, L.phon.values, R.phon.values)
    f["name_phon_token_set"] = _cp(fuzz.token_set_ratio, L.phon.values, R.phon.values)
    multi = (pd.Series(L.variants.values).str.contains("|", regex=False).values
             | pd.Series(R.variants.values).str.contains("|", regex=False).values)
    vb = f["name_token_set"].copy()
    if multi.any():
        vb[multi] = _variant_best(L.variants.values[multi], R.variants.values[multi])
    f["name_variant_best"] = vb
    f["dba_either"] = multi.astype(np.float32)
    if not hasattr(stats, "_idf_map"):
        stats._idf_map = dict(zip(stats.name_word.get_feature_names_out(), stats.name_word.idf_.astype(float)))
        stats._idf_max = float(stats.name_word.idf_.max())
    cov = _coverage_feats(n1, n2, stats._idf_map, stats._idf_max)
    for j, c in enumerate(("name_s1_cov", "name_s1_miss_idf", "name_s1_n_miss",
                           "name_rec_cov", "name_rec_miss_idf", "name_rec_n_miss")):
        f[c] = cov[:, j]
    t1 = pd.Series(n1).str.split(n=1).str[0].fillna("").values
    t2 = pd.Series(n2).str.split(n=1).str[0].fillna("").values
    f["name_first_tok_eq"] = (t1 == t2).astype(np.float32)
    len1 = pd.Series(n1).str.len().values
    len2 = pd.Series(n2).str.len().values
    f["name_len_diff"] = np.abs(len1 - len2).astype(np.float32)
    f["name_len_ratio"] = (np.minimum(len1, len2) / np.maximum(np.maximum(len1, len2), 1)).astype(np.float32)
    for c in list(f):
        if c.startswith("name_") and c not in ("name_len_diff", "name_len_ratio"):
            f[c] = np.where(name_miss, np.nan, f[c]).astype(np.float32)

    # --- address
    f["addr_tfidf"] = _rowdot(_enc(stats.addr_char, a1), _enc(stats.addr_char, a2))
    W1, W2 = _enc(stats.addr_word, a1), _enc(stats.addr_word, a2)
    f["addr_word_tfidf"] = _rowdot(W1, W2)
    f["addr_token_set"] = _cp(fuzz.token_set_ratio, a1, a2)
    f["addr_token_sort"] = _cp(fuzz.token_sort_ratio, a1, a2)
    f["addr_partial"] = _cp(fuzz.partial_ratio, a1, a2)
    f["addr_wjacc"] = _weighted_jaccard(W1, W2, stats.addr_word.idf_.astype(np.float32))
    del W1, W2
    for c in ("addr_tfidf", "addr_word_tfidf", "addr_token_set", "addr_token_sort", "addr_partial", "addr_wjacc"):
        f[c] = np.where(addr_miss, np.nan, f[c]).astype(np.float32)

    # --- numeric
    m1, m2 = L.nums.values, R.nums.values
    num_miss = (m1 == "") | (m2 == "")
    B1 = _enc(stats.num_vec, m1)
    B2 = _enc(stats.num_vec, m2)
    inter = _rowdot(B1, B2)
    c1 = np.asarray(B1.sum(1)).ravel()
    c2 = np.asarray(B2.sum(1)).ravel()
    # numbers outside the fitted vocabulary are counted by hand for the union
    k1 = pd.Series(m1).str.split().str.len().fillna(0).values
    k2 = pd.Series(m2).str.split().str.len().fillna(0).values
    union = np.maximum(k1 + k2 - inter, 1)
    f["num_jacc"] = np.where(num_miss, np.nan, inter / union).astype(np.float32)
    f["num_any"] = np.where(num_miss, np.nan, (inter > 0).astype(np.float32))
    f["num_contained"] = np.where(num_miss, np.nan, inter / np.maximum(np.minimum(k1, k2), 1)).astype(np.float32)
    del c1, c2
    nf = _number_feats(m1, m2)
    for j, c in enumerate(("num_longest_eq", "num_longest_in", "num_best_ratio", "num_min_reldiff", "num_prefix")):
        f[c] = nf[:, j]
    p1, p2 = L.postal.values, R.postal.values
    f["postal_eq"] = np.where((p1 == "") | (p2 == ""), np.nan, (p1 == p2).astype(np.float32))

    # --- meta & missingness
    f["is_s3"] = (R.src.values == 3).astype(np.float32)
    c1v, c2v = L.country.values, R.country.values
    f["same_country"] = np.where((c1v == "") | (c2v == ""), np.nan, (c1v == c2v).astype(np.float32))
    f["name_missing"] = name_miss.astype(np.float32)
    f["addr_missing_l"] = (a1 == "").astype(np.float32)
    f["addr_missing_r"] = (a2 == "").astype(np.float32)
    f["postal_missing"] = ((p1 == "") | (p2 == "")).astype(np.float32)
    f["nums_missing"] = num_miss.astype(np.float32)
    f["addr_ntok_l"] = pd.Series(a1).str.split().str.len().fillna(0).values.astype(np.float32)
    f["addr_ntok_r"] = pd.Series(a2).str.split().str.len().fillna(0).values.astype(np.float32)
    return pd.DataFrame(f).astype(np.float32)


def add_context(df, score_col, prefix):
    """Rank / gap-to-best of each pair (by `score_col`) among its S1's candidates per source,
    and among the record's competing S1s.  df needs s1_pos, src, r_pos."""
    s = df[score_col].fillna(-1.0)
    g1 = s.groupby([df.s1_pos, df.src])
    df[f"{prefix}_rank_s1"] = g1.rank(ascending=False, method="min").astype(np.float32)
    df[f"{prefix}_gap_s1"] = (g1.transform("max") - s).astype(np.float32)
    g2 = s.groupby([df.src, df.r_pos])
    df[f"{prefix}_rank_rec"] = g2.rank(ascending=False, method="min").astype(np.float32)
    df[f"{prefix}_gap_rec"] = (g2.transform("max") - s).astype(np.float32)
    df[f"{prefix}_n_rec"] = g2.transform("size").astype(np.float32)
    df["n_cand_s1"] = df.groupby([df.s1_pos, df.src]).s1_pos.transform("size").astype(np.float32)
    return df
