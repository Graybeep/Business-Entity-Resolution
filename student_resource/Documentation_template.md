# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** [Date]

---

## 1. Executive Summary

A classic *blocking → pairwise classifier → entity-level decision* pipeline, built entirely on the provided data.
Candidates come from several **exact sparse TF-IDF top-k passes** (word, char 4-gram and a *phonetic skeleton* view)
plus a reverse pass, within country; a **LightGBM** model scores each pair from ~60 string / number / context features;
a **stacked second stage** adds how each pair's probability compares with its competitors; the output is decided per
Source-1 entity with **one-owner assignment** and **two thresholds tuned directly on the exact macro F0.5** using
grouped out-of-fold predictions.

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

**Passes** (exact cosine top-k with `sparse_dot_topn`; vectorisers fitted on a sample of the split being processed;
terms in more than 1 % (0.2 % for char) of documents dropped):

| Pass | View | k per source |
|---|---|---|
| `na` | word TF-IDF of core name + address | 10 |
| `c4` | char_wb 4-gram TF-IDF of core name + address | 5 |
| `ph` | phonetic name skeleton + address words | 5 |
| `nm` | word TF-IDF of core name only | 5 |
| `rev` | each S2/S3 record → its top-2 S1s by `na` | 2 |

The union is taken separately for S2 and S3, within each country label (a record with an empty country would join every
pool). Because Stage 0 found no cross-country true pairs, no global pass is needed.

- **Blocking keys used:** exact sparse TF-IDF cosine over four text views + reverse pass (no hard key, so no single corrupted
  field can drop a pair).
- **Candidate pairs generated:** train (15 % of S1 queried, reverse pass against all S1): 12,091,331 pairs, 36.6 per S1.
  Test (all 1,732,544 S1): 63,846,924 pairs, 36.9 per S1 (France 39.0, India 37.3, US 35.4 — the unseen country blocks like the others); no test S1 is left without candidates.
- **Recall / reduction ratio (train):** pair recall **0.971** (US 0.99 / India 0.98 in the sample study); reduction ratio
  **0.999993** vs the within-country cross product.
- **How we ensured true matches were not lost:** recall was measured per pass and for their union at several k.
  Misses were read one by one, and each pass was added for a failure family: `ph` for transliterated names, `nm` for
  empty addresses, `c4` for typos, `rev` for records whose S1 is crowded out of its own top-k. We tried and discarded
  SVD-compressed char-TF-IDF embeddings with GPU search (recall 0.94 at k = 30, because SVD discards the rare n-grams
  that identify a business).

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
- Stage 2 (stack): the stage-1 out-of-fold probability, and its max / gap / rank / count-above-0.5 / sum / best competitor
  within the S1 group and within the record group.

**Model type:** LightGBM binary classifier (logloss, fixed seed, no class weights, no resampling, no calibration), 5-fold
GroupKFold by S1 id; the final model is refit on all training pairs with the mean best iteration. Stage 2 is a second
LightGBM on stage-1 features + OOF-derived context, trained with the same grouped folds.

**Threshold selection method:** decision per S1 entity. (1) One-owner: each record is kept only by its highest-probability
S1 (0 shared owners in train). (2) An S1 keeps its best candidate only if P ≥ `t_first`, and each further candidate if P ≥
`t_add`. `t_first` and `t_add` are tuned jointly on the grouped OOF probabilities over quantiles of the score
distribution, maximising the exact macro F0.5, which is implemented once in `postprocess.py` and used everywhere.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, grouped 5-fold OOF over all 330,664 training-subset S1 entities):** **0.9648**
  (`t_first` = 0.607, `t_add` = 0.612; near-tie rate among assigned records 0.0001. Run 1 before the number/coverage features: 0.9510.
  Oracle with perfect pair classification on the same candidates: 0.990, so blocking costs ~0.01 and the matcher ~0.025.)
- **Feature ablation (10 % subset of the training S1s, identical folds, lr 0.1):**

  | Variant | CV macro F0.5 | Δ |
  |---|---|---|
  | A: base features | 0.9460 | — |
  | B: + number features and asymmetric name-coverage features | 0.9588 | **+0.0128** (kept) |
  | C: B + stacked stage 2 (competitor context of OOF P) | 0.9594 | +0.0006 — within noise, **not used** |
- **Country-shift check (proxy for France; model trained on one country, scored on the other):**

  | Held-out country | thresholds re-tuned on it | main OOF thresholds | other country's thresholds |
  |---|---|---|---|
  | India (trained on US) | 0.861 | 0.865 | 0.858 |
  | US (trained on India) | 0.955 | 0.948 | 0.914 |

  India-only → US transfers well; US-only → India loses ~0.08 because Indic transliteration and landmark-style
  addresses never occur in US data. France is Latin-script with number + street addresses (structurally closer to US)
  and the final model sees both training countries. The main residual risk for France is **threshold calibration**:
  the "other country's thresholds" column shows a 0.04 loss when thresholds do not match the country.
- **Test-set sanity check (no labels; the thresholds were not tuned on test):** predicted singleton fraction France 5.96 % /
  India 6.27 % / US 5.79 % (train truth 5.6 %); predicted matches per S1 France 3.16 / India 3.15 / US 3.30. The unseen
  country behaves like the training countries, so the thresholds do not look badly miscalibrated for France.
  Test candidate set: 63,846,924 pairs, reduction ratio 0.9999963 vs the full cross product.
- **Common false positives (wrong merges):** "twin" records with the S1's name and a house number shifted by a few units
  or truncated (4150 → 415); names that differ in one distinctive word (`Shrinathji Capital` vs `Shrinathji Systems`).
- **Common false negatives (missed matches):** records with an empty address and a name altered by an appended word; transliterated
  Indic names with short addresses; true matches whose house number is perturbed exactly like a twin's.
- Loss breakdown (run 1): singletons wrongly matched 0.8 %, entities with every match missed 0.7 %, partially correct lists 3.4 %.

---

## 6. Conclusion

Exact sparse multi-view blocking with a phonetic key achieves 0.97 pair recall at ~37 candidates per S1. A
LightGBM matcher with number-aware and coverage features, stacked with competitor context, and a decision stage tuned on
the exact metric reach 0.9648 macro F0.5 in grouped cross-validation. The largest remaining loss is the twin
distractors: records whose name and address match the S1 except for a house number shifted the same way true matches
are.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — see its `README.md`. Entry point:

```bash
python code/business_entity_resolution/src/pipeline.py --data-dir dataset --out output --cache cache --shift-check
```

| File | Role |
|---|---|
| `src/normalize.py` | Stage 1 normalisation (no country branching) |
| `src/data.py` | TSV I/O (`sep="\t"`, `keep_default_na=False`), cached parallel normalisation |
| `src/blocking.py` | Stage 2 candidate generation |
| `src/features.py` | Stage 3 pair and context features |
| `src/postprocess.py` | Stage 5 metric, one-owner, two thresholds, tuning |
| `src/pipeline.py` | Stage 0 diagnostics, Stage 4 model/CV, stage-2 stack, shift check, output |

Models: LightGBM (MIT), trained from scratch; no pretrained model, LLM or external data.

### B. Additional Results

`output/run_summary.json` contains diagnostics, blocking recall per pass and k, reduction ratio, CV score, thresholds,
near-tie rate, shift-check scores and full gain feature importance.
