"""Stage 5 - entity-level decision + the single implementation of the competition metric."""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric: per-S1 F0.5, macro-averaged over ALL S1 entities (singletons included)
# ---------------------------------------------------------------------------
def entity_f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_sets, true_sets, s1_ids):
    """pred_sets / true_sets: dict s1 -> set of ids; every id in s1_ids is scored."""
    empty = frozenset()
    return float(np.mean([entity_f05(pred_sets.get(k, empty), true_sets.get(k, empty)) for k in s1_ids]))


def macro_f05_arrays(s1_codes, is_match, keep, true_count, n_s1):
    """Vectorised metric. Pair arrays (s1 integer code, label, kept flag) plus true match
    count per S1 (length n_s1).  Equivalent to macro_f05."""
    pred_n = np.bincount(s1_codes, weights=keep, minlength=n_s1)
    tp = np.bincount(s1_codes, weights=keep & is_match, minlength=n_s1)
    f = np.zeros(n_s1)
    empty_true = true_count == 0
    f[empty_true & (pred_n == 0)] = 1.0
    ok = (~empty_true) & (tp > 0)
    p = tp[ok] / pred_n[ok]
    r = tp[ok] / true_count[ok]
    f[ok] = 1.25 * p * r / (0.25 * p + r)
    return float(f.mean())


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
def one_owner(s1_codes, rid_codes, prob):
    """Each S2/S3 record keeps only its highest-P S1 (ties -> smallest S1 code, which is
    the smallest S1 id because codes are assigned on sorted ids).  Returns a bool mask."""
    order = np.lexsort((s1_codes, -prob, rid_codes))
    first = np.ones(len(order), dtype=bool)
    r = rid_codes[order]
    first[1:] = r[1:] != r[:-1]
    mask = np.zeros(len(prob), dtype=bool)
    mask[order[first]] = True
    return mask


def near_tie_rate(rid_codes, prob, gap=0.05, min_top=0.0):
    """Among records whose best P >= min_top (i.e. records that would be assigned), the
    fraction whose top-2 competing S1 probabilities differ by < gap."""
    df = pd.DataFrame({"r": rid_codes, "p": prob}).sort_values(["r", "p"], ascending=[True, False])
    first = np.r_[True, df.r.values[1:] != df.r.values[:-1]]
    top1 = df.p.values[first]
    rec = df.r.values[first]
    second_mask = np.r_[False, first[:-1]] & ~first
    top2 = pd.Series(df.p.values[second_mask], index=df.r.values[second_mask]).reindex(rec).fillna(-1).values
    live = top1 >= min_top
    return float(((top1 - top2) < gap)[live].sum() / max(live.sum(), 1))


class Decider:
    """Pre-sorts pairs once so that many (t_first, t_add) settings can be evaluated fast.
    An S1 keeps its top (owned) candidate only if P >= t_first, and each further owned
    candidate only if P >= t_add."""

    def __init__(self, s1_codes, rid_codes, prob, use_owner=True):
        owner = one_owner(s1_codes, rid_codes, prob) if use_owner else np.ones(len(prob), bool)
        p = np.where(owner, prob, -1.0)
        self.order = np.lexsort((-p, s1_codes))
        s = s1_codes[self.order]
        self.p = p[self.order]
        is_top = np.ones(len(s), dtype=bool)
        is_top[1:] = s[1:] != s[:-1]
        self.is_top = is_top
        # best owned probability of each pair's S1, broadcast to its pairs
        grp = np.cumsum(is_top) - 1
        self.top_p = self.p[is_top][grp]
        self.n = len(prob)

    def keep(self, t_first, t_add):
        ks = (self.top_p >= t_first) & np.where(self.is_top, self.p >= t_first, self.p >= t_add)
        keep = np.zeros(self.n, dtype=bool)
        keep[self.order] = ks
        return keep


def decide(s1_codes, rid_codes, prob, t_first, t_add, use_owner=True):
    return Decider(s1_codes, rid_codes, prob, use_owner).keep(t_first, t_add)


def tune_thresholds(s1_codes, rid_codes, prob, is_match, true_count, n_s1, n_q=50, use_owner=True):
    """Joint search of (t_first, t_add) over quantiles of the OOF score distribution,
    then a fine local refinement.  Returns (score, t_first, t_add)."""
    dec = Decider(s1_codes, rid_codes, prob, use_owner)
    score = lambda tf, ta: macro_f05_arrays(s1_codes, is_match, dec.keep(tf, ta), true_count, n_s1)
    # quantiles of the scores that can be kept: over all pairs, ~97 % sit near 0 and a coarse grid
    # jumps straight from ~0.02 to ~0.99 (n_q=30 tuned 0.95 instead of ~0.6 on run-2 OOF)
    live = prob[prob >= 0.01]
    qs = np.unique(np.quantile(live if len(live) else prob, np.linspace(0.0, 0.99, n_q)))
    best = (-1.0, 0.5, 0.5)
    for tf in qs:
        for ta in qs:
            sc = score(tf, ta)
            if sc > best[0]:
                best = (sc, float(tf), float(ta))
    _, tf0, ta0 = best
    for tf in np.unique(np.clip(tf0 + np.linspace(-0.04, 0.04, 17), 0, 1)):
        for ta in np.unique(np.clip(ta0 + np.linspace(-0.04, 0.04, 17), 0, 1)):
            sc = score(tf, ta)
            if sc > best[0]:
                best = (sc, float(tf), float(ta))
    return best
