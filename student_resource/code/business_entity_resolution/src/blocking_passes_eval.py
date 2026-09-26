"""Blocking-miss analysis + candidate-pass evaluation on the cached 15 % train candidates (train only).

1. Categorise true pairs missed by the current blocking (cache/train_cands_f0.15.parquet):
   empty address, transliteration (non-Latin record name), name divergence / DBA, postal mismatch,
   other (priority in that order).
2. For each candidate extra pass, measure on the 15 % S1 subset:
   recall gain (missed true pairs the pass would add) and cost (new pairs per S1, deduplicated
   against the current candidates).  Every pass is within country, with the pool = ALL S1s /
   records of that country (as at inference); only hits on subset S1s are counted.
   a) exact key (country, postal, house number)
   b) the record's rarest core-name token (lowest S1 document frequency), df cap sweep
   c) sibling expansion: word TF-IDF (core+addr) top-5 record neighbours of each S1's top-3
      candidates by run-2 OOF P (sampled S1s; needs a model score, i.e. a second scoring round)
   d) name-only (core word TF-IDF) record -> S1 top-k for records with an empty address, k sweep
3. Union of the chosen settings.

python blocking_passes_eval.py   (log: runs/blocking_passes_eval.log)
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import PASSES, _vectorizer, pair_key, par_transform, topk  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CACHE = os.path.join(ROOT, "cache")
T0 = time.time()


def log(msg):
    """Timestamped progress line."""
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def load():
    """S1 (all, with 15 % subset mask), records per source, true pairs of subset S1s, current keys."""
    cols = ["entity_id", "country", "core", "addr", "nums", "postal", "business_name"]
    s1 = pd.read_parquet(os.path.join(CACHE, "train_source1.parquet"), columns=cols)
    u = np.random.default_rng(42).random(len(s1))
    m15 = u < 0.15
    rec = {k: pd.read_parquet(os.path.join(CACHE, f"train_source{k}.parquet"), columns=cols) for k in (2, 3)}
    gt = pd.read_csv(os.path.join(ROOT, "dataset", "train", "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    lists = gt.matched_entity_ids.str.split(",")
    pairs = pd.DataFrame({"s1": np.repeat(gt.source1_entity_id.values, lists.map(len).values),
                          "rid": [i for l in lists for i in l]})
    pairs = pairs[pairs.rid != ""]
    pos1 = pd.Series(np.arange(len(s1)), index=s1.entity_id.values)
    pairs["s1_pos"] = pos1.reindex(pairs.s1).values
    pairs = pairs[m15[pairs.s1_pos.values]].reset_index(drop=True)
    pairs["src"] = np.where(pairs.rid.str.startswith("S3-"), 3, 2).astype(np.int8)
    pairs["r_pos"] = 0
    for k in (2, 3):
        idx = pd.Series(np.arange(len(rec[k])), index=rec[k].entity_id.values)
        mk = pairs.src.values == k
        pairs.loc[mk, "r_pos"] = idx.reindex(pairs.rid[mk]).values
    pairs["key"] = pair_key(pairs.s1_pos.values, pairs.src.values, pairs.r_pos.values)
    c = pd.read_parquet(os.path.join(CACHE, "train_cands_f0.15.parquet"), columns=["s1_pos", "src", "r_pos"])
    cur = np.unique(pair_key(c.s1_pos.values, c.src.values, c.r_pos.values))
    pairs["found"] = np.isin(pairs.key.values, cur)
    return s1, m15, rec, pairs, cur


def rget(rec, pairs, col):
    """Column value of each pair's record."""
    out = np.empty(len(pairs), object)
    for k in (2, 3):
        mk = pairs.src.values == k
        out[mk] = rec[k][col].values[pairs.r_pos.values[mk]]
    return out


def categorise(s1, rec, pairs):
    """Mutually exclusive miss categories (priority order) for the missed pairs."""
    m = pairs[~pairs.found]
    L = s1.iloc[m.s1_pos.values]
    r_addr, r_core, r_name, r_post = (rget(rec, m, c) for c in ("addr", "core", "business_name", "postal"))
    empty = (L.addr.values == "") | (r_addr == "")
    nonlatin = np.array([any(ord(ch) > 0x24F for ch in s) for s in r_name])
    tset = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(L.core.values, r_core)])
    postal_mm = (L.postal.values != "") & (r_post != "") & (L.postal.values != r_post)
    cat = np.select([empty, nonlatin, tset < 70, postal_mm],
                    ["empty address", "transliteration (non-Latin name)", "name divergence / DBA (tset < 70)",
                     "postal mismatch"], "other (similar name, has address)")
    s = pd.Series(cat)
    ctry = L.country.values
    log(f"recall {pairs.found.mean():.4f}; missed {len(m)} of {len(pairs)} true pairs")
    print(pd.crosstab(s, ctry, normalize="columns").round(3).assign(all=s.value_counts(normalize=True).round(3)).to_string())
    return m


def gain_cost(name, new_keys, pairs, cur, n_sub):
    """Recall gain on the missed pairs and new (deduplicated) pairs per subset S1."""
    new_keys = np.setdiff1d(np.unique(new_keys), cur)
    hit = np.isin(pairs.key.values[~pairs.found.values], new_keys)
    g = hit.sum() / len(pairs)
    log(f"  {name:45s} recall +{g:.4f} -> {pairs.found.mean() + g:.4f} | +{len(new_keys) / n_sub:.1f} cands/S1")
    return new_keys


def pass_postal_num(s1, m15, rec, cap=50):
    """a) (country, postal, house number) exact key; keys with > cap S1s are skipped."""
    def explode(df, pos):
        d = df.iloc[pos][["country", "postal", "nums"]].assign(pos=pos)
        d = d[(d.postal != "") & (d.nums != "")]
        d = d.assign(num=d.nums.str.split()).explode("num")
        return d.assign(k=d.country + "|" + d.postal + "|" + d.num)[["k", "pos"]]
    a = explode(s1, np.arange(len(s1)))
    size = a.groupby("k").size()
    a = a[m15[a.pos.values] & a.k.map(size).le(cap).values]
    out = []
    for k in (2, 3):
        b = explode(rec[k], np.arange(len(rec[k])))
        j = a.merge(b, on="k", suffixes=("_1", "_r"))
        out.append(pair_key(j.pos_1.values, np.full(len(j), k), j.pos_r.values))
    return np.concatenate(out)


def rare_token_index(s1, m15, rec):
    """Per country: S1 core-token document frequency, token -> subset S1 positions, and every
    record's rarest core token with its df.  Computed once, reused for every df cap."""
    idx = {}
    for cty in [c for c in s1.country.unique() if c]:
        ms = np.flatnonzero(s1.country.values == cty)
        toks = s1.core.values[ms]
        df = Counter(t for s in toks for t in set(s.split()))
        inv = {}
        for p, s in zip(ms, toks):
            if m15[p]:
                for t in set(s.split()):
                    inv.setdefault(t, []).append(p)
        recs = {}
        for k in (2, 3):
            mr = np.flatnonzero(rec[k].country.values == cty)
            best_t, best_df = [], []
            for s in rec[k].core.values[mr]:
                ts = [t for t in set(s.split()) if df.get(t, 0) > 0]
                t = min(ts, key=lambda x: df[x]) if ts else ""
                best_t.append(t)
                best_df.append(df[t] if t else 10**9)
            recs[k] = (mr, best_t, np.array(best_df))
        idx[cty] = (inv, recs)
        log(f"    b) index {cty} done")
    return idx


def pass_rare_token(idx, cap):
    """b) each record's rarest core token -> subset S1s holding it, if its S1 df <= cap."""
    out = []
    for inv, recs in idx.values():
        for k, (mr, best_t, best_df) in recs.items():
            rr, ss = [], []
            for i in np.flatnonzero(best_df <= cap):
                for p in inv.get(best_t[i], ()):
                    ss.append(p)
                    rr.append(mr[i])
            out.append(pair_key(np.array(ss, np.int64), np.full(len(ss), k), np.array(rr, np.int64)))
    return np.concatenate(out)


def pass_empty_addr_name(s1, m15, rec, kmax, threads=14):
    """d) records with empty address -> top-kmax S1s by core-name word TF-IDF (pool: all S1 of the
    country).  Returns (keys, rank) so smaller k can be read off."""
    spec = PASSES["nm"]
    keys, ranks = [], []
    for cty in [c for c in s1.country.unique() if c]:
        ms = np.flatnonzero(s1.country.values == cty)
        vec = _vectorizer(spec, 0, s1.core.values[ms])
        T = par_transform(vec, s1.core.values[ms])
        for k in (2, 3):
            mr = np.flatnonzero((rec[k].country.values == cty) & (rec[k].addr.values == ""))
            Q = par_transform(vec, rec[k].core.values[mr])
            row, col, sim = topk(Q, T, kmax, threads)
            order = np.lexsort((-sim, row))
            row, col = row[order], col[order]
            start = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1]
            rank = np.arange(len(row)) - np.repeat(start, np.diff(np.r_[start, len(row)]))
            s1p = ms[col]
            keep = m15[s1p]
            keys.append(pair_key(s1p[keep], np.full(keep.sum(), k), mr[row[keep]]))
            ranks.append(rank[keep])
        log(f"    d) {cty} done")
    return np.concatenate(keys), np.concatenate(ranks)


def pass_sibling(s1, m15, rec, pairs, frac=0.3, top_c=3, nb=5, threads=14, anchor="oof"):
    """c) for a random `frac` of subset S1s: each S1's top-`top_c` candidates by run-2 OOF P, then
    each anchor's top-`nb` record neighbours (word TF-IDF core+addr, both sources, same country).
    Returns (new keys, sampled S1 mask) - gains must be scaled to the sample."""
    if anchor == "oof":
        oof = pd.read_parquet(os.path.join(CACHE, "oof_pairs.parquet"))
    else:   # blocking-time score: best sim over the passes (no model needed)
        oof = pd.read_parquet(os.path.join(CACHE, "train_cands_f0.15.parquet"))
        sims = oof[[c for c in oof.columns if c.startswith("sim_")]].fillna(0).values
        oof = oof[["s1_pos", "src", "r_pos"]].assign(p=sims.max(axis=1))
    rng = np.random.default_rng(1)
    samp = m15 & (rng.random(len(m15)) < frac)
    oof = oof[samp[oof.s1_pos.values]].sort_values(["s1_pos", "p"], ascending=[True, False])
    anc = oof.groupby("s1_pos").head(top_c)
    spec = PASSES["na"]
    allrec = pd.concat([rec[2][["country", "core", "addr"]].assign(src=2, pos=np.arange(len(rec[2]))),
                        rec[3][["country", "core", "addr"]].assign(src=3, pos=np.arange(len(rec[3])))],
                       ignore_index=True)
    base = np.r_[0, len(rec[2])]
    anc_row = base[anc.src.values - 2] + anc.r_pos.values
    out = []
    for cty in [c for c in s1.country.unique() if c]:
        mr = np.flatnonzero(allrec.country.values == cty)
        text = (allrec.core.values[mr] + " " + allrec.addr.values[mr]).astype(object)
        vec = _vectorizer(spec, 0, text)
        T = par_transform(vec, text)
        loc = pd.Series(np.arange(len(mr)), index=mr)
        a = anc[np.isin(anc_row, mr)]
        a_row = loc.reindex(base[a.src.values - 2] + a.r_pos.values).values.astype(np.int64)
        Q = T[a_row]
        row, col, _ = topk(Q, T, nb + 1, threads)
        nbr = mr[col]
        s1p = a.s1_pos.values[row]
        src = allrec.src.values[nbr]
        pos = allrec.pos.values[nbr]
        out.append(pair_key(s1p, src, pos))
        log(f"    c) {cty}: {len(a)} anchors")
    return np.concatenate(out), samp


def main_sibling_variants():
    """Sweep sibling-expansion settings (same 30 % sample), unioned with d) top-50."""
    s1, m15, rec, pairs, cur = load()
    n_sub = int(m15.sum())
    kd, rd = pass_empty_addr_name(s1, m15, rec, 50)
    d50 = np.setdiff1d(np.unique(kd), cur)
    for anchor, top_c, nb in (("oof", 5, 10), ("block", 3, 5), ("block", 5, 10), ("oof", 10, 10)):
        kc, samp = pass_sibling(s1, m15, rec, pairs, top_c=top_c, nb=nb, anchor=anchor)
        ps = pairs[samp[pairs.s1_pos.values]]
        miss = ps.key.values[~ps.found.values]
        new = np.setdiff1d(np.unique(kc), cur)
        dd = d50[samp[(d50 >> 24).astype(np.int64)]]
        g = np.isin(miss, new).sum() / len(ps)
        gu = np.isin(miss, np.union1d(new, dd)).sum() / len(ps)
        base = ps.found.mean()
        log(f"  c) anchors={anchor:5s} top{top_c} x {nb} nbrs: recall {base:.4f} -> {base + g:.4f} (+{g:.4f}), "
            f"+{len(new) / samp.sum():.1f} cands/S1 | with d50: {base + gu:.4f}, "
            f"+{len(np.union1d(new, dd)) / samp.sum():.1f} cands/S1")


def main():
    """Run the categorisation and the pass sweeps; print a summary."""
    s1, m15, rec, pairs, cur = load()
    n_sub = int(m15.sum())
    log(f"15 % subset: {n_sub} S1, {len(pairs)} true pairs, current candidates {len(cur) / n_sub:.1f}/S1")
    categorise(s1, rec, pairs)
    chosen = {}
    log("a) exact (country, postal, house number)")
    for cap in (20, 100):
        chosen[f"a_cap{cap}"] = gain_cost(f"a) postal+number, S1 group cap {cap}", pass_postal_num(s1, m15, rec, cap), pairs, cur, n_sub)
    log("b) rarest core token")
    idx = rare_token_index(s1, m15, rec)
    for cap in (5, 20, 50):
        chosen[f"b_cap{cap}"] = gain_cost(f"b) rarest token, df cap {cap}", pass_rare_token(idx, cap), pairs, cur, n_sub)
    del idx
    log("d) empty-address records -> name-only top-k")
    kd, rd = pass_empty_addr_name(s1, m15, rec, 50)
    for k in (10, 20, 50):
        chosen[f"d_k{k}"] = gain_cost(f"d) empty-address name-only top-{k}", kd[rd < k], pairs, cur, n_sub)
    log("c) sibling expansion (30 % sample of subset S1s)")
    kc, samp = pass_sibling(s1, m15, rec, pairs)
    ps = pairs[samp[pairs.s1_pos.values]]
    new = np.setdiff1d(np.unique(kc), cur)
    g = np.isin(ps.key.values[~ps.found.values], new).sum() / len(ps)
    per = len(new) / samp.sum()
    chosen["c"] = new
    log(f"  c) sibling expansion top-3 x 5 neighbours (sample)   recall +{g:.4f} (sample-based) | +{per:.1f} cands/S1")
    log("unions")
    for combo in (["a_cap20", "b_cap20", "d_k20"], ["a_cap100", "b_cap50", "d_k50"], ["a_cap20", "b_cap20", "d_k20", "c"]):
        keys = np.unique(np.concatenate([chosen[c] for c in combo]))
        hit = np.isin(pairs.key.values[~pairs.found.values], keys)
        note = " (c counted only on its sample: recall gain is a lower bound)" if "c" in combo else ""
        log(f"  UNION {'+'.join(combo):35s} recall {pairs.found.mean() + hit.sum() / len(pairs):.4f} | "
            f"+{len(keys) / n_sub:.1f} cands/S1 (upper bound for c){note}")


if __name__ == "__main__":
    main_sibling_variants() if "--sibling-variants" in sys.argv else main()
