# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** grain
**Team Members:** [List all team members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

A *blocking → pairwise classifier → entity-level decision* pipeline built only on the provided data, in two rounds.
**Round 1** retrieves candidates with four exact sparse TF-IDF top-k passes (word, char 4-gram, a script-agnostic
*phonetic skeleton*, name-only) plus a reverse pass, within country. A **LightGBM** stage-1 model scores each pair from
67 string / number / context features. **Round 2** then adds candidates the passes cannot reach: records with an empty
address matched by name only, and **sibling expansion** (records that closely resemble an S1's best-scoring
candidates). A **stacked stage-2 LightGBM** re-scores the plausible pairs with competitor context,
**sibling-agreement** features (does this record agree with the S1's other confident matches on name, address and
house number?) and name-frequency features. The output is decided per Source-1 entity with **one-owner assignment**
and **two thresholds tuned on the exact macro F0.5** using grouped out-of-fold predictions. Grouped CV macro F0.5:
**0.9761** (candidate recall 0.984 at 53 candidates per S1).

---

## 2. Methodology

### 2.1 Problem Analysis

Stage-0 diagnostics on the training set (2,206,821 S1; 5,034,616 S2; 5,285,603 S3):

| Finding | Value | Consequence |
|---|---|---|
| S1 singletons (no true match) | 5.6 % | most S1s have 2–5 matches → recall matters too |
| Match-list size | mode 3, max 11 | multiple matches per source per S1 |
| S2/S3 ids under more than one S1 | **0** | each record has one owner → per-record argmax is valid |
| True matches whose country labels differ | **0** | blocking within country loses nothing |
| S2/S3 records matching nothing | 26 % | many distractors |
| Empty address (S2 / S3) | 3.4 % / 3.3 % | name-only evidence for these |
| Addresses with a 5–6 digit postal code | ~7 % | postal code is a weak signal |
| Names in an Indic script (S2 / S3) | 5.4 % / 3.0 % | need transliteration |

Noise patterns seen when reading matched groups:

- **Names:** typos, leetspeak (`C0rp`, `Chemica1s`, `5ervices`), junk prefixes (`--`, `***`, `>>`), shuffled word order,
  added accents, legal-form changes, appended generic words (`Services`, `Center`, `Holdings`), honorifics (`Sri`, `Dr`),
  domain style (`kempgloba.com`, `#bellmodern`), DBA / F/K/A with an unrelated trade name, and names written in
  Devanagari, Telugu, Tamil, Gujarati, Bengali, Gurmukhi, Kannada or Oriya script.
- **Addresses:** component re-ordering, abbreviations, state name vs code (`Texas`/`TX`), state names in Indic script,
  `null`, missing components, house-number prefixes (`H.No`, `Door No`, `#`), leading zeros, and perturbed or
  truncated house numbers.
- **"Twin" distractors:** unmatched records with the same (or nearly the same) name as an S1 and a slightly different
  house number (4150 vs 415, 6732 vs 6734). True matches also carry number noise, which makes these the hardest errors.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pair classifier (two-stage stack) + entity-level decision optimised for macro F0.5
**Core Innovation:** multi-view exact sparse blocking with a script-agnostic phonetic key (so Indic-script and Latin spellings of
the same name collide), and a decision stage tuned directly on the competition metric with one-owner assignment.

S2 and S3 go through one pipeline; the source is a feature (`is_s3`), not a separate branch. Nothing branches on the
value of `country`; it is only used as an open-set blocking pool label and in `same_country`.

---

## 3. Candidate Generation (Blocking)

**Normalisation (country-agnostic, `normalize.py`).** NFKD accent folding; transliteration of every Brahmic script with one
offset table (these Unicode blocks share the ISCII layout), with virama/matra handling and schwa deletion; lowercase;
`&`→`and`; punctuation removal; legal-form canonicalisation (corporation→corp, private→pvt, limited→ltd, …, including
French forms sarl/sas/eurl/sci/cie and romanised Indic forms such as `pra li`→`pvt ltd`); `core_name` without legal tokens and
honorifics; DBA / aka / F/K/A split into name variants; domain and hashtag unwrapping; leetspeak repair on mixed
tokens; address abbreviations (road→rd, rue→r, avenue/av→ave, …, ordinal words→`9th`), house-number prefixes dropped,
leading zeros stripped; numbers and postal code extracted. A **phonetic skeleton** of the core name (voiced/unvoiced
merged, sibilants merged, vowels/y/h dropped, repeats collapsed) makes, e.g., `gud treding` ≡ `good trading` (`kt trtnk`)
and `purotyuchar` ≡ `producer` (`prtsr`).

**Round 1 passes** (exact cosine top-k with `sparse_dot_topn`; vectorisers fitted on a sample of the split being processed;
terms in more than 1 % (0.2 % for char) of documents dropped):

| Pass | View | k per source |
|---|---|---|
| `na` | word TF-IDF of core name + address | 10 |
| `c4` | char_wb 4-gram TF-IDF of core name + address | 5 |
| `ph` | phonetic name skeleton + address words | 5 |
| `nm` | word TF-IDF of core name only | 5 |
| `rev` | each S2/S3 record → its top-2 S1s by `na` | 2 |

**Round 2 (after stage-1 scoring):**

| Pass | View | Cost |
|---|---|---|
| empty-address name pass | each record with an empty address → top-50 S1s by core-name word TF-IDF | +5.5 per S1 |
| sibling expansion | each S1's top-5 candidates by stage-1 P → their top-10 record neighbours (word TF-IDF of core name + address, both sources) | +11.4 per S1 |

Sibling expansion works because duplicates of one business resemble each other more than they resemble the S1: a
transliterated or empty-address record is reached through an English-name or full-address sibling. It needs model
scores to choose the anchors; with blocking-score anchors the gain halves (+0.0053 vs +0.0115 recall on a 15 % study).

Every pass runs within each country label, separately for S2 and S3 (0 cross-country true pairs in training; a record
with an empty country would join every pool).

- **Blocking keys used:** exact sparse TF-IDF cosine over four text views + reverse pass + the two round-2 passes (no hard
  key, so no single corrupted field can drop a pair).
- **Candidate pairs generated:** train (40 % of S1 queried): 32.3M round-1 + 14.5M round-2 pairs = 53.0 per S1.
  Test (all 1,732,544 S1): 63.8M round-1 + 26.3M round-2 = 90.2M pairs, **52.0 per S1**; no test S1 is left without
  candidates.
- **Recall / reduction ratio (train):** pair recall **0.9717 → 0.9841** with round 2 (India 0.9486 → 0.9697,
  US 0.9871 → 0.9937); reduction ratio **0.99999** vs the within-country cross product.
- **How we ensured true matches were not lost:** every missed true pair was categorised (transliteration 32 %, empty
  address 28 %, similar name with address 22 %, name divergence / DBA 18 %, postal mismatch 0.1 %), and each candidate
  pass was measured for recall gain against candidates per S1 on a 15 % study before adoption. Discarded: an exact
  (country, postal, house number) key (+0.0001 recall), a rarest-name-token pass (+0.0006 at +5.3 per S1), and
  SVD-compressed char-TF-IDF embeddings with GPU search (recall 0.94 at k = 30, because SVD drops the rare n-grams that
  identify a business).

---

## 4. Matching Model

**Features used** (missing input → NaN, never 0; no country one-hot):
- Name features: char-3 and word TF-IDF cosine; RapidFuzz ratio / token_sort / token_set / partial; Jaro-Winkler; IDF-weighted
  token Jaccard; no-space ratio / partial (domain-style names); full-name token_set; phonetic-skeleton ratio and token_set;
  best token_set over DBA variants; first-token equality; length difference / ratio; asymmetric IDF-weighted token
  coverage (S1→record and record→S1) with the IDF of the most distinctive missing token.
- Address features: char-3 and word TF-IDF cosine; token_set / token_sort / partial; IDF-weighted Jaccard; token counts.
- Number features: number-set Jaccard, any-overlap, containment; longest-number equality / membership; best digit-string
  similarity; minimum relative numeric difference; prefix (truncation) flag; postal equality.
- Context: per-pass retrieval rank and cosine from blocking; rank and gap-to-best of a base TF-IDF score among the S1's
  candidates and among the record's competing S1s; candidate counts.
- Meta / missingness: `is_s3`, `same_country`, name/address/postal/number missing indicators.
- **Stage 2** adds: the stage-1 out-of-fold probability and its max / gap / rank / count-above-0.5 / sum / best
  competitor within the S1 group and within the record group; **10 sibling-agreement features** (count and P-mass of
  the S1's other candidates with P ≥ 0.5; max and P-weighted mean agreement of this record with them on core name,
  address and number set; share of siblings with an identical number set; whether the S1's own numbers back the
  record); **2 name-frequency features** (how many S1s and how many records of the country carry the record's exact
  core name); `is_round2`.

**Model type:** two LightGBM binary classifiers (logloss, fixed seeds, no class weights, no calibration), MIT licence,
trained from scratch. **Stage 1** is trained on 40 % of the training S1 entities (882,771 S1, 32.3M pairs) with
5-fold GroupKFold by S1. To fit in 16 GB RAM, features are stored on disk and the *training* folds keep every positive,
every hard negative (base-score rank ≤ 3 in its S1 group, or best S1 of its record) and 25 % of the other negatives
(16.1M rows); validation folds and threshold tuning always use the full candidate sets. **Stage 2** is trained on the
stage-1 OOF probabilities (round-2 training pairs are scored by the fold model that did not see their S1) over the
5.2M pairs with stage-1 P ≥ 0.01 (round 1) or ≥ 0.001 (round 2), with grouped folds. Final models are refit on all rows
with the mean best iteration.

**Threshold selection method:** decision per S1 entity. (1) One-owner: each record is kept only by its
highest-probability S1 (0 shared owners in train). (2) An S1 keeps its best candidate only if P ≥ `t_first`, and each
further candidate only if P ≥ `t_add`. Both are tuned jointly on the grouped OOF probabilities over a quantile grid of
the scores ≥ 0.01 (a grid over all pair scores is degenerate, since ~97 % of pairs score near 0), maximising the exact
macro F0.5 implemented once in `postprocess.py`. Final: `t_first` = 0.478, `t_add` = 0.760.

---

## 5. Results & Error Analysis

- **F_0.5 score (macro, grouped 5-fold OOF over the 40 % training S1 subset, all 882,771 entities):** **0.9761**
  (India 0.9681, US 0.9813; `t_first` = 0.478, `t_add` = 0.760).

- **Progression** (same metric throughout; public LB where submitted):

  | Run | Change | CV macro F0.5 | Public LB |
  |---|---|---|---|
  | run 2 | base + number / name-coverage features, 15 % of S1 | 0.9654 | 0.952 |
  | run 8 | 40 % of S1, disk-backed features, easy-negative sampling in training folds | 0.9677 | — |
  | run 9 | + stacked stage 2 with sibling-agreement features | 0.9723 | 0.959 |
  | **run 10** | + round-2 candidates (empty-address name pass, sibling expansion) + name frequency | **0.9761** | **0.964** |

  The CV → LB gap stayed at −0.012 to −0.013 (runs 2, 9 and 10), so CV gains carried over to the leaderboard.

- **Feature and component ablations (grouped CV):** number + name-coverage features +0.0128; stage-2 stack with
  competitor context +0.0026 and sibling-agreement features a further +0.0020; round 2 + name frequency +0.0038.
  Expected-F0.5 subset selection per entity instead of two thresholds: −0.0015 (dropped).

- **Country-shift check (proxy for France):** stage 1 trained on one country scores held-out India 0.863 and US 0.959
  (0.859 / 0.925 with the other country's thresholds). A stage-2 layer trained on the other country still adds
  +0.0028 (India) / +0.0017 (US), so it is not country-specific. A diagnostic submission with every France row empty
  (never a final) implies a France score of about 0.95 (0.944–0.954), in line with India.

- **Known train/test shift.** Training blocking queries only the 40 % S1 subset, so in training a record meets
  about 40 % of the S1s it competes with at test time (3.9 vs 6.2–6.4 candidate S1s per record). Adversarial validation
  (US + India pairs, train vs test) reaches AUC 0.911, driven by the record-side competition features (`ctx_gap_rec`
  median 0.14 train vs 0.43 test; `ctx_n_rec` 7 vs 11) and an IDF-based coverage feature (IDF fitted per split). Two
  fixes that need no extra RAM were measured on the leaderboard against run 10: *density matching* (record-side
  features recomputed against a hash-sampled 52.9 % of test S1s, which restores the training competitor count, no
  retraining): public LB **0.963** vs 0.964, so not adopted; and an *ablation* without record-side competition
  features (CV 0.9752, −0.0009), not submitted. So the record-side shift, although large, is not what separates CV
  from the leaderboard. The principled fix, blocking and training on every S1, needs ≥ 24 GB RAM; it is future work,
  together with IDF fitted jointly on train and test.

- **Remaining loss (run 10 OOF):** India 0.0318 = false positives on true singletons 9 %, false positives on matched S1s
  14 %, model misses 36 %, blocking misses 42 %; US 0.0186 = 8 / 12 / 63 / 16 %. With perfect pair classification on
  the round-1 candidates the score would be 0.990.
- **Common false positives (wrong merges):** "twin" records with the S1's name and a house number shifted by a few
  units or truncated (4150 → 415); legal-form twins (`… SARL, 92 Rue Drouot` vs `… SA, 89 Rue Drouot`).
- **Common false negatives (missed matches):** exact-name records with an empty address whose name several S1s share;
  transliterated Indic names in crowded generic-name groups (`Ram Products Private Limited`); true matches whose house
  number is perturbed exactly like a twin's.

---

## 6. Conclusion

Multi-view exact sparse blocking with a phonetic key, extended by a model-guided second round (empty-address name pass
and sibling expansion), reaches 0.984 pair recall at ~53 candidates per S1. A LightGBM matcher, stacked with a second
stage that checks each candidate against the S1's other confident matches, and a decision stage tuned on the exact
metric reach 0.976 grouped-CV macro F0.5. The main open issues are India blocking (42 % of India's remaining loss) and
the train/test difference in competitor density caused by training on an S1 subset.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (see its `README.md`). One entry point regenerates both output files from `dataset/`:

```bash
DATA=dataset CACHE=cache OUT=. bash code/business_entity_resolution/run_all.sh
```

| File | Role |
|---|---|
| `src/normalize.py` | Normalisation (no country branching) |
| `src/data.py` | TSV I/O (`sep="\t"`, `keep_default_na=False`), cached parallel normalisation |
| `src/blocking.py` | Round-1 candidate generation |
| `src/features.py` | Pair and context features |
| `src/training_scaled.py` | Disk-backed grouped CV with easy-negative sampling |
| `src/pipeline.py` | Diagnostics, stage-1 model, CV, thresholds, test blocking and stage-1 scoring |
| `src/train_fold_models.py` | Stage-1 fold models (out-of-fold scores for round-2 training pairs) |
| `src/stage2_sibling.py` | Sibling-agreement features |
| `src/run10.py` | Round-2 expansion, stage-2 stack, final test decision and output files |
| `src/postprocess.py` | Metric, one-owner assignment, two thresholds, tuning |
| `src/density_match.py`, `src/ablation.py` | Train/test-shift variants (section 5) |
| `src/smoke_dataset.py` | 5 % sample of `dataset/` for the end-to-end smoke test |

Models: LightGBM (MIT), trained from scratch; no pretrained model, LLM or external data.

### B. Additional Results

`EXPERIMENTS.md` logs every run (change, CV delta, decision). `runs/*/run_summary.json` and
`submissions/run10_train_result.json` hold diagnostics, blocking recall, CV scores, thresholds, shift-check scores and
feature importance.
