# CLAUDE.md — Amazon ML Challenge 2026: Business Entity Resolution

Read this before touching code. It holds the competition rules, the time and submission budget,
the agreed architecture, and the decisions already made (with reasons). Do not reopen a decision in
the Decision Log without new evidence from the training data.

---

## 0. Clock and budget (overrides everything else)

- **Window:** 25 Sep 2026 00:00 IST → **27 Sep 2026 23:59 IST**. Three days, no extensions.
- **Submissions:** max **5 per day**, 15 in total; the button is disabled afterwards. Unused
  submissions do not carry over to the next day.
- Shortlisting is based on the **submitted solutions**, and evaluation uses **both public and
  private leaderboards**. Final ranking weight sits on the private leaderboard.
- Consequence: a working, validated end-to-end submission on **day 1** beats a better design on
  day 3. Scope cuts are made in favour of shipping.

### Day plan
| Day | Goal | Submissions |
|---|---|---|
| 25 Sep | Diagnostics → blocking (recall gate) → baseline LightGBM + Stage 5 → **first valid submission** to measure the CV↔LB gap | 2–3 |
| 26 Sep | Error analysis on OOF false positives/negatives, feature additions, shift check, threshold retune | 3–5 |
| 27 Sep | **Feature freeze by 18:00 IST**. Final run, package zip, docs. Final submission no later than **22:00 IST** | 1–3, keep 1 in reserve |

### When to spend a submission
- Spend one when CV macro F0.5 improves by more than ~0.003 over the last submitted version.
- Also spend one to test a specific hypothesis about the CV↔LB gap. Write the hypothesis down first.
- Never spend one without a PASS from `validate_submission.py`.
- Never submit two variants that differ only by noise-level threshold changes, unless it is a
  planned single-parameter LB probe (§6, "Public LB"); write its hypothesis down first.

---

## 1. Task in one paragraph

There are three sources of business records (`entity_id`, `business_name`, `business_address`,
`country`). Source 1 is deduplicated and is the reference. For every S1 entity, output all S2/S3
records that refer to the same real-world business (zero, one, or many). There is **no phone field
and no shared ID** — name, address and country are the only signals. Train covers US + India;
**test adds France**, which never appears in training.

---

## 2. Hard rules (violating any = rejection or disqualification)

### Fair play
- **No external data of any kind.** That includes entity-resolution APIs, business registries,
  geocoding APIs, web lookups, and downloaded gazetteers or address databases.
- Hand-written normalisation maps (abbreviations, legal suffixes) are allowed — they are code, not
  data.
- TF-IDF / IDF statistics may be fit on the provided data only. Fitting within the test split is
  fine.
- Any model must be **MIT or Apache-2.0 licensed** and **≤ 8B parameters**. Verify the licence on
  the model card, and record name, licence and size in the README.
- One registration per participant; no multiple IDs.

### Country handling
- Treat `country` as an **open set of strings**. Never hard-code, filter, or one-hot it to
  `{US, India}`.
- Every test S1 entity — France included — must appear in the output.
- Normalisation must be country-agnostic: the same code path for every country.

### File I/O
- All files are **tab-separated**. Always read with
  `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`.
  `keep_default_na=False` matters: empty match lists must stay `""`, not become `NaN`.
- Write with a single tab between columns, IDs comma-separated, no quoting, no spaces.

### Output: `output/matching_results.tsv`
- Columns: `source1_entity_id<TAB>matched_entity_ids`
- Exactly **one row per test S1 entity**, with no duplicate S1 rows.
- `matched_entity_ids` is empty for predicted singletons.
- Lists contain only S2-/S3- IDs that exist in the test set: no S1 IDs, no duplicates within a list.

### Output: `output/candidate_pairs.tsv`
- Columns: `source1_entity_id<TAB>candidate_entity_ids`
- Same row rules as above.
- Must be **exactly the set the model scores at inference**: the last filtering stage, not an early
  blocking pass.
- Every matched ID must also appear in that S1 row's candidates.

### Before any submission
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
It must print PASS.

---

## 3. Deliverables (two different documents — do both)

| Artefact | Requirement | Source |
|---|---|---|
| **1–2 page write-up** | ML approach, models used, **experiments**, conclusion. Required for the best submitted solution. | Guidelines |
| **Documentation_template.md** | Methodology, blocking strategy, model architecture + features, other details. No page limit. Goes in the zip; top-100 teams are asked for these details. | Problem statement |
| **Source code** | Experiments, training and inference, **with comments describing every function**. | Guidelines |
| **Submission zip** | Layout below. | Problem statement |

```
<team_name>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/, README.md, requirements.txt (pinned)}
└── Documentation_template.md   (filled in)
```
The pipeline must regenerate both outputs from `dataset/` using only what is in `code/`.

- The 1–2 page write-up is a condensed version of the template; write the template first.
- Both documents are built from `EXPERIMENTS.md` (§7). If it is kept current, the docs take 30
  minutes on day 3, not 3 hours.

---

## 4. Metric (exact definition — optimise this, nothing else)

For each S1 entity, compute F0.5 between the predicted and true match sets. Then take the **macro
average over all S1 entities**, singletons included.

| Truth | Prediction | Entity score |
|---|---|---|
| empty | empty | **1.0** |
| empty | any match | **0.0** |
| non-empty | empty | 0.0 |
| non-empty | non-empty | F0.5 = 1.25·P·R / (0.25·P + R) |

Consequences for the design:
- The **first** match on an entity is a 1↔0 flip if that entity is truly a singleton, so it needs a
  high bar. **Additional** matches only dilute precision, so their bar can be lower.
- As a weighted harmonic mean, the weights are 0.8 on 1/P and 0.2 on 1/R. At P = R, precision's
  marginal effect is 4× recall's. (The rules' "2×" is van Rijsbergen's β convention.) Tune against
  the exact formula, not intuition.
- Pairwise AUC, pairwise F1 and log-loss are diagnostics only. Never select a model or threshold on
  them.

Implement the metric once, in `src/postprocess.py`, and use that implementation everywhere.

---

## 5. Architecture

```
S1, S2, S3 ──► 0 Diagnostics ──► 1 Normalise ──► 2 Blocking ──► 3 Features
                                                      │
                                         candidate_pairs.tsv
                                                      ▼
                        5 Entity-level decision ◄── 4 LightGBM P(match)
                                  │
                         matching_results.tsv
```

S2 and S3 go through **one** pipeline. The source is a feature, not a separate branch.

### Stage 0 — Diagnostics (train; rerun whenever data handling changes)
Report the following and log them to `output/run_summary.json`:
- The singleton fraction and the distribution of match-list sizes.
- The number of S2/S3 IDs appearing under more than one S1. This validates the one-owner rule
  (expected 0).
- The number of true matches whose country labels differ. This decides the blocking mode in
  Stage 2.
- Missingness per field, per source: empty name / address / country, addresses with no digits, and
  addresses with no postal code.

### Stage 1 — Normalisation (`src/normalize.py`)
- Unicode NFKD, accent stripping, lowercase, `&` → `and`, dots removed, other punctuation → space,
  whitespace collapsed.
- Legal and business words canonicalised (corporation→corp, private→pvt, limited→ltd,
  incorporated→inc, company→co, …), plus generic French forms (sarl, sas, sa, eurl, cie).
- `core_name` is the name with legal tokens stripped.
- Name variants are split on DBA / "trading as" / aka.
- Address abbreviations canonicalised (road→rd, street→st, avenue/av→ave, boulevard/bd→blvd,
  near→nr, opposite→opp, …).
- Numbers are extracted from the address; the postal code is the last 5–6-digit token.
- Nothing in this module may branch on `country`.

### Stage 2 — Candidate generation (`src/blocking.py`)
- Character n-gram TF-IDF (`char_wb`, 2–4) on core name, normalised address, and name + address.
- For each S1, take the union of:
  - top-k by name (k≈20)
  - top-k by address (k≈10)
  - top-k by name + address (k≈10)
  - reverse top-3 (each S2/S3 record → its nearest S1s)
- Run this separately for S2 and S3.
- **The blocking mode depends on the Stage 0 cross-country count:**
  - ≈ 0 → block within the country label; records with an empty country join every pool.
  - Non-trivial → add a **global** pass (all countries, smaller k) to the union, and rely on
    `same_country` plus numeric agreement for precision.
- **Gate:** training blocking recall must be ≥ 0.95 before any model work. If it falls short, raise
  k or add a pass.
- Report recall and the reduction ratio; both go in the docs.
- Budget: blocking plus features on train must finish in a few minutes. If it doesn't, lower k
  before optimising code.

### Stage 3 — Features (`src/features.py`)
| Group | Features |
|---|---|
| Name | TF-IDF cosine; RapidFuzz ratio, token_sort, token_set, partial; Jaro-Winkler; IDF-weighted token Jaccard; best token_set over DBA variants; first-token equality; length difference |
| Address | TF-IDF cosine; token_set; partial; IDF-weighted Jaccard |
| Numeric | number-set Jaccard; any-number overlap; postal equality |
| Context | rank and gap-to-best among the S1's candidates (per source); rank and gap-to-best among the candidate's competing S1s; number of candidates |
| Meta | `is_s3`; `same_country` |
| Missing | explicit indicators for name / address / postal / numbers missing on either side; address token count |

- **Missing values must be `NaN`, never 0 or -1.** This applies to every similarity whose input
  field is missing, including the numeric and postal features. "Missing" is not "low similarity",
  and LightGBM handles NaN natively.
- **No country one-hot, ever.**

### Stage 4 — Model (`src/pipeline.py`)
- LightGBM binary classifier, logloss objective, fixed seed.
- **No `scale_pos_weight`, no class weights, no resampling, no post-hoc calibration.** See the
  Decision Log.
- 5-fold **GroupKFold grouped by S1 entity_id**. The out-of-fold probabilities feed Stage 5 tuning.
- The final model is refit on all training pairs using the mean best iteration from CV.
- Log feature importance (gain) on every run.

### Stage 5 — Entity-level decision (`src/postprocess.py`)
1. **One-owner:** each S2/S3 record goes to its single highest-P S1 (per-record argmax,
   deterministic tie-break by S1 id). Active only if Stage 0 found 0 shared IDs in train.
2. **Two thresholds:**
   - An S1 keeps its top candidate only if P ≥ `t_first`; otherwise its output is empty.
   - It keeps each further candidate only if P ≥ `t_add`.
3. `t_first` and `t_add` are tuned **jointly** on the out-of-fold probabilities to maximise the
   exact macro F0.5. Search over **quantiles of the out-of-fold score distribution**, not a fixed
   grid.
4. Log the **near-tie rate**: the fraction of records whose top-2 S1 probabilities differ by less
   than 0.05. A high rate points to a feature gap, not an assignment problem.
5. Optional, only if time remains on day 2: per entity, pick the candidate subset that maximises
   expected F0.5 given P.

---

## 6. Validation protocol

- **Primary estimate:** macro F0.5 on grouped out-of-fold predictions over **all** train S1
  entities. Entities with no candidates count, and score 1.0 only if they are true singletons.
- **Country-shift check (proxy for France):** train on US and evaluate on India, then the reverse.
  Report the score with thresholds retuned on the held-out country and with the other country's
  thresholds. A large gap means features or thresholds don't transfer.
- **CV is mildly optimistic,** because early stopping and threshold tuning share the out-of-fold
  predictions. Treat gains below ~0.003 as noise.
- **Public LB:**
  - Use it to measure the CV↔LB gap and to sanity-check the direction of improvements.
  - Test has ~1.73M S1s, so the public LB is precise to ~±0.0005, and differences between
    submissions are paired and even more precise. Use it to decide between configs whose CV
    differs by < 0.003, and for single-parameter probes (e.g. France thresholds). Do not
    grid-search many parameters on it.
  - If LB and CV disagree on direction twice in a row, stop and investigate before submitting
    again. The usual cause is the France share of the test set or a blocking gap.
- Every run writes `output/run_summary.json` with:
  - diagnostics
  - blocking recall and reduction ratio
  - CV score
  - `t_first` / `t_add`
  - near-tie rate
  - shift-check scores
  - top features

---

## 7. Version history (required — shortlisting uses submitted solutions)

- **Git commit before every submission.** Tag it `sub-DD-N`, e.g. `sub-25-1`.
- **Archive the exact uploaded file** as `submissions/sub-DD-N_matching_results.tsv`, along with its
  `run_summary.json`.
- Keep `submissions/log.tsv` with one row per submission: `tag, timestamp_IST, commit, cv_macro_f05,
  public_lb, t_first, t_add, blocking_recall, change_summary, hypothesis`.
- Keep `EXPERIMENTS.md` current. Every run, submitted or not, gets a single line: what changed, CV
  delta, keep/revert. This is the raw material for both docs.

---

## 8. Decision Log (settled — reopen only with new data evidence)

| Decision | Reason |
|---|---|
| One pipeline for S2 + S3, with an `is_s3` feature | Separate branches halve the training data; the trees learn source-specific splits themselves. Missing-indicators cover differing missingness. |
| Per-record argmax for contested records, not linear assignment | One-to-one assignment is wrong, because an S1 can have many matches. Min-cost flow with unlimited S1 capacity reduces to argmax. |
| Two thresholds (`t_first`, `t_add`) instead of a single threshold + relative cutoff | The singleton 1↔0 flip makes the first match costlier than additional ones. |
| No `scale_pos_weight` / class weighting | It changes the loss and tree splits and de-calibrates the output. It doesn't help a threshold that is tuned anyway. |
| No isotonic/Platt calibration | Monotonic, so a threshold tuned on out-of-fold predictions selects the same pairs. |
| NaN for missing, not 0 | Keeps "field absent" distinct from "genuinely dissimilar". |
| Within-country blocking is conditional | It excludes cross-country pairs structurally, and nothing downstream can recover them. |
| No LLM / cross-encoder | Not enough time in a 3-day window to justify the compute and risk. Revisit only if day-2 error analysis shows semantic misses (trade names unrelated to the legal name) *and* there is time; it must be MIT/Apache and ≤ 8B. |
| Public LB may decide between near-tied configs and single-parameter probes (26 Sep) | With ~1.73M test S1s, the public LB (even a 10 % subset ≈ 173k S1s) resolves ~±0.0005, and probe-vs-base differences are paired, so they are finer than CV's ~0.003 noise floor. Grid-searching many parameters on it is still banned (overfits the public subset; private LB decides the ranking). |

---

## 9. Repository layout

```
student_resource/
├── CLAUDE.md
├── EXPERIMENTS.md                    # one line per run
├── dataset/{train,test}/             # read-only, never modify
├── utils/validate_submission.py      # provided, never modify
├── output/                           # generated
├── submissions/                      # archived uploads + log.tsv
├── docs/{writeup_1-2pages.md}        # guidelines artefact
├── Documentation_template.md         # problem-statement artefact
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

## 10. Commands

```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/pipeline.py --data-dir dataset --out output --shift-check
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 11. Working rules for Claude

- **Check the clock first.** Before starting any task, state the time left until 27 Sep 23:59 IST
  and whether the task fits the day plan (§0). After the day-3 feature freeze, only bug fixes,
  packaging and docs.
- **Measure every change.** Change one thing at a time and report the delta in CV macro F0.5,
  blocking recall and the shift-check score. No unmeasured "improvements".
- **Blocking changes** must re-report recall and the reduction ratio.
- **Feature additions** need evidence first: show the error cases they target, as sampled false
  positives and false negatives from the out-of-fold predictions.
- **Every function gets a docstring** saying what it does, its inputs and outputs, and which stage
  it belongs to. This is a guidelines requirement, not style.
- **Test data is for inference only.** Test inputs may be inspected to find normalisation failures
  on unseen formats (no test labels exist). Thresholds and model choices are never tuned on test.
- **Deterministic runs:** fixed seeds, IDs sorted before writing.
- **No network calls** in the pipeline. `pip install` is the only allowed network use.
- **When unsure whether something counts as "external data",** don't use it, and flag it to the
  user.
- **After every run,** append to `EXPERIMENTS.md`.
- **Before every submission,** complete §7 (commit, tag, archive, log).
- **Rule questions** go to the organisers' Google Form. Don't guess on anything that could
  disqualify.
