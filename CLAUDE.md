# CLAUDE.md — Business Entity Resolution Challenge

Read this before touching code. It holds the competition rules, the agreed architecture, and the
decisions already made (with reasons). Do not reopen a decision in the Decision Log without new
evidence from the training data.

---

## 1. Task in one paragraph

Three sources of business records (`entity_id`, `business_name`, `business_address`, `country`).
Source 1 is deduplicated and is the reference. For every S1 entity, output all S2/S3 records that
refer to the same real-world business (zero, one, or many). There is **no phone field and no shared
ID** — name, address, and country are the only signals. Train covers US + India; **test adds France**,
which never appears in training.

---

## 2. Hard rules (violating any = rejection or disqualification)

### Fair play
- **No external data of any kind**: no entity-resolution APIs, business registries, geocoding APIs,
  web lookups, or downloaded gazetteers/address databases.
- Hand-written normalisation maps (abbreviations, legal suffixes) are allowed — they are code, not data.
- TF-IDF / IDF statistics must be fit on the provided data only (fitting within the test split is fine).
- Any model used must be **MIT or Apache-2.0 licensed** and **≤ 8B parameters**. Verify the licence on
  the model card before adding it; record name + licence + size in the README.

### Country handling
- Treat `country` as an **open set of strings**. Never hard-code, filter, or one-hot to `{US, India}`.
- Every test S1 entity — France included — must appear in the output.
- Normalisation must be country-agnostic (the same code path for every country).

### File I/O
- All files are **tab-separated**. Always read with
  `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`.
  `keep_default_na=False` matters: empty match lists must stay `""`, not become `NaN`.
- Write with a single tab between columns, IDs comma-separated, no quoting, no spaces.

### Output: `output/matching_results.tsv`
- Columns: `source1_entity_id<TAB>matched_entity_ids`
- Exactly **one row per test S1 entity**; no duplicate S1 rows.
- `matched_entity_ids` empty for predicted singletons.
- Only S2-/S3- IDs that exist in the test set; no S1 IDs; no duplicates within a list.

### Output: `output/candidate_pairs.tsv`
- Columns: `source1_entity_id<TAB>candidate_entity_ids`
- Same row rules as above.
- Must be **exactly the set the model scores at inference** (the last filtering stage, not an early
  blocking pass).
- Every matched ID must also appear in that S1 row's candidates.

### Before any submission
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
It must print PASS.

### Final package layout
```
<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt (pinned)}
└── Documentation_template.md   (filled in)
```
The pipeline must regenerate both outputs from `dataset/` using only what is in `code/`.

---

## 3. Metric (exact definition — optimise this, nothing else)

For each S1 entity, compute F0.5 between the predicted and true match sets, then take the
**macro average over all S1 entities**, singletons included.

| Truth | Prediction | Entity score |
|---|---|---|
| empty | empty | **1.0** |
| empty | any match | **0.0** |
| non-empty | empty | 0.0 |
| non-empty | non-empty | F0.5 = 1.25·P·R / (0.25·P + R) |

Consequences for design:
- The **first** match on an entity is a 1↔0 flip if that entity is truly a singleton, so it needs a
  high bar. **Additional** matches only dilute precision, so their bar can be lower.
- As a weighted harmonic mean, the weights are 0.8 on 1/P and 0.2 on 1/R, so at P = R precision's
  marginal effect is 4× recall's. Tune against the exact formula, not intuition.
- Pairwise AUC, pairwise F1, and log-loss are diagnostics only. Never select a model or threshold on them.

Implement the metric once in `src/postprocess.py` and use that single implementation everywhere.

---

## 4. Architecture

```
S1, S2, S3 ──► 0 Diagnostics ──► 1 Normalise ──► 2 Blocking ──► 3 Features
                                                      │
                                         candidate_pairs.tsv
                                                      ▼
                        5 Entity-level decision ◄── 4 LightGBM P(match)
                                  │
                         matching_results.tsv
```

S2 and S3 go through **one** pipeline; the source is a feature, not a separate branch.

### Stage 0 — Diagnostics (train; rerun whenever data handling changes)
Report and log to `output/run_summary.json`:
- The singleton fraction and the distribution of match-list sizes.
- The number of S2/S3 IDs appearing under more than one S1. This validates the one-owner rule
  (expected 0).
- The number of true matches whose country labels differ. This decides blocking mode in Stage 2.
- Missingness per field, per source (empty name / address / country, address with no digits,
  address with no postal code).

### Stage 1 — Normalisation (`src/normalize.py`)
- Unicode NFKD plus accent stripping, lowercase, `&` → `and`, dots removed, other punctuation →
  space, whitespace collapsed.
- Legal and business words canonicalised (corporation→corp, private→pvt, limited→ltd,
  incorporated→inc, company→co, …), plus generic French forms (sarl, sas, sa, eurl, cie).
- A `core_name` with legal tokens stripped.
- Name variants split on DBA / "trading as" / aka.
- Address abbreviations canonicalised (road→rd, street→st, avenue/av→ave, boulevard/bd→blvd,
  near→nr, opposite→opp, …).
- Numbers extracted from the address; postal code = the last 5–6-digit token.
- Nothing in this module may branch on `country`.

### Stage 2 — Candidate generation (`src/blocking.py`)
- Vectorise with character n-gram TF-IDF (`char_wb`, 2–4) on core name, normalised address, and
  name + address.
- For each S1, take the union of:
  - the top-k by name (k≈20)
  - the top-k by address (k≈10)
  - the top-k by name + address (k≈10)
  - the reverse top-3 (each S2/S3 record → its nearest S1s)
- Run this separately for S2 and S3.
- **Blocking mode depends on the Stage 0 cross-country count:**
  - ≈ 0 → block within country label (records with an empty country join every pool).
  - Non-trivial → add a **global** pass (all countries, smaller k) to the union, and rely on
    `same_country` plus numeric agreement for precision.
- **Gate:** training-set blocking recall must be ≥ 0.95 before any model work. If it is lower,
  raise k or add a pass. Report recall and the reduction ratio (both go in the documentation).

### Stage 3 — Features (`src/features.py`)
| Group | Features |
|---|---|
| Name | TF-IDF cosine; RapidFuzz ratio, token_sort, token_set, partial; Jaro-Winkler; IDF-weighted token Jaccard; best token_set over DBA variants; first-token equality; length difference |
| Address | TF-IDF cosine; token_set; partial; IDF-weighted Jaccard |
| Numeric | number-set Jaccard; any-number overlap; postal equality |
| Context | rank and gap-to-best of the pair among the S1's candidates (per source); rank and gap-to-best among the candidate's competing S1s; number of candidates |
| Meta | `is_s3`; `same_country` |
| Missing | explicit indicators: name / address / postal / numbers missing on either side; address token count |

**Missing values must be `NaN`, never 0 or -1**, for every similarity whose input field is
missing (including numeric/postal). A missing field is not the same as a low similarity; LightGBM
handles NaN natively. **No country one-hot, ever.**

### Stage 4 — Model (`src/pipeline.py`)
- A LightGBM binary classifier with logloss objective and a fixed seed.
- **No `scale_pos_weight`, no class weights, no resampling, no post-hoc calibration** (see the
  Decision Log).
- Training uses 5-fold **GroupKFold grouped by S1 entity_id**. The out-of-fold probabilities feed
  Stage 5 tuning.
- The final model is refit on all training pairs with the mean best iteration from CV.
- Log feature importance (gain) on every run.

### Stage 5 — Entity-level decision (`src/postprocess.py`)
1. **One-owner assignment:** each S2/S3 record goes to its single highest-P(match) S1 (per-record
   argmax, deterministic tie-break by S1 id). This step is only active if Stage 0 found 0 shared IDs
   in training.
2. **Two thresholds:**
   - An S1 keeps its top candidate only if P ≥ `t_first`. Otherwise its output is empty.
   - It keeps each further candidate only if P ≥ `t_add`.
3. `t_first` and `t_add` are tuned **jointly** on the out-of-fold probabilities to maximise the exact
   macro F0.5. Search over **quantiles of the OOF score distribution**, not a fixed grid.
4. Log the **near-tie rate**: the fraction of records whose top two S1 probabilities differ by less
   than 0.05. If it is high, treat it as a feature gap, not an assignment problem.
5. Possible upgrade if the two-threshold version plateaus: per entity, choose the candidate subset
   that maximises expected F0.5 given P.

---

## 5. Validation protocol

- The main estimate is the macro F0.5 on grouped out-of-fold predictions, over **all** train S1
  entities (entities with no candidates count, and score 1.0 only if they are true singletons).
- **Country-shift check** (proxy for France): train on US, evaluate on India, and the reverse.
  Retune the thresholds within the held-out country and also report the score using the
  thresholds from the other country. A large gap means the features or thresholds do not transfer.
- The CV score is mildly optimistic, because early stopping and threshold tuning share the
  out-of-fold predictions. Do not treat small CV gains (< ~0.003) as real.
- Do not tune on the public leaderboard. It is a subset, and the private leaderboard decides the
  ranking.
- Every run writes `output/run_summary.json` containing: diagnostics, blocking recall, reduction
  ratio, CV score, the chosen `t_first`/`t_add`, the near-tie rate, shift-check scores, and the top
  features.

---

## 6. Decision Log (settled — reopen only with new data evidence)

| Decision | Reason |
|---|---|
| One pipeline for S2 + S3 with an `is_s3` feature | Separate branches halve the training data; the trees learn source-specific splits themselves. Missing-indicators cover differing missingness between sources. |
| Per-record argmax for contested records, not linear assignment | One-to-one assignment is wrong, because an S1 can have many matches. Min-cost flow with unlimited S1 capacity reduces to argmax, so it adds nothing. |
| Two thresholds (`t_first`, `t_add`) instead of a single threshold plus relative cutoff | The singleton 1↔0 flip makes the first match costlier than additional ones. |
| No `scale_pos_weight` / class weighting | It changes the loss and the tree splits, and de-calibrates the output. It does not help a threshold that is tuned anyway. |
| No isotonic/Platt calibration | Both are monotonic, so a threshold tuned on out-of-fold predictions selects the same pairs either way. |
| NaN for missing, not 0 | Keeps "field absent" distinct from "genuinely dissimilar". |
| Within-country blocking is conditional | It excludes cross-country pairs structurally, and no downstream stage can recover them. |
| No LLM or cross-encoder in the baseline | Add one only if error analysis shows semantic misses (trade names unrelated to the legal name), and only if it is MIT/Apache and ≤ 8B. |

---

## 7. Repository layout

```
student_resource/
├── dataset/{train,test}/             # read-only, never modify
├── utils/validate_submission.py      # provided, never modify
├── output/                           # generated
└── code/business_entity_resolution/
    ├── src/
    │   ├── normalize.py      # Stage 1
    │   ├── blocking.py       # Stage 2
    │   ├── features.py       # Stage 3
    │   ├── postprocess.py    # Stage 5 + metric
    │   └── pipeline.py       # Stages 0, 4, orchestration, I/O
    ├── README.md
    └── requirements.txt      # pinned versions
```

## 8. Commands

```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/pipeline.py --data-dir dataset --out output --shift-check
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 9. Working rules for Claude

- Change one thing at a time and report the delta in CV macro F0.5, blocking recall, and the
  shift-check score. No unmeasured "improvements".
- Every change to blocking must re-report recall and the reduction ratio.
- Before adding a feature, show the error cases it targets (sampled false positives and false
  negatives from the out-of-fold predictions).
- Never read the test files to choose thresholds or features. The test set is used for inference only.
- Keep runs deterministic (fixed seeds, sorted IDs before writing).
- Never add network calls to the pipeline. The only allowed network use is `pip install`.
- When uncertain whether something counts as "external data", don't use it and flag it to the user.
- Keep `Documentation_template.md` in sync. It must cover: methodology, blocking strategy (with
  recall and reduction ratio), model and features, the decision stage, and validation results.
