# Business Entity Resolution — reproducible pipeline

Blocking (exact sparse TF-IDF top-k) → pair features → LightGBM P(match) → entity-level decision
(one-owner assignment + two thresholds tuned on grouped out-of-fold predictions for macro F0.5).

## Environment

- Python 3.11, CPU only (16 threads / 16 GB RAM machine used for development).
- `pip install -r code/business_entity_resolution/requirements.txt` (versions pinned).
- No network access is used by the pipeline. No external data of any kind.

## Reproduce

Run from the `student_resource/` directory (the one that contains `dataset/`):

```bash
python code/business_entity_resolution/src/pipeline.py --data-dir dataset --out output --cache cache --shift-check
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Outputs:

| File | Content |
|---|---|
| `output/matching_results.tsv` | final matches, one row per test S1 entity |
| `output/candidate_pairs.tsv` | exactly the pairs the model scores at inference |
| `output/run_summary.json` | diagnostics, blocking recall / reduction ratio, CV score, thresholds, near-tie rate, shift check, feature importance |

`cache/` holds normalised parquet files, candidate sets, the trained model and test probabilities;
delete it to recompute everything from scratch.

Approximate runtime on the machine above (16 threads, 16 GB RAM): normalisation 6 min; train blocking
70 min; test blocking 2.5 h; train features + 5-fold CV + threshold tuning 70 min; shift check 25 min;
refit 7 min; test features + prediction 50 min.

Optional flags (all stages are cached, so an interrupted run can be resumed):

| Flag | Effect |
|---|---|
| `--shift-check` | train on one country, evaluate on the other (proxy for the unseen test country) |
| `--blocking-only` | build and cache the train and test candidate sets, then exit |
| `--skip-test` | stop after training (CV, thresholds, refit) |
| `--predict-only` | skip training; reuse `cache/model_stage1.txt` and the thresholds in `output/run_summary.json` |
| `--reuse-test-prob` | with `--predict-only`: reuse cached test probabilities, only re-run the decision and writing |

The training set is a fixed random 15 % of the train S1 entities (seed 42) with all their candidates
(~12M pairs); this is a memory bound, not a modelling choice.

## Source layout (`src/`)

| File | Stage |
|---|---|
| `normalize.py` | Stage 1 — country-agnostic normalisation (accent folding, Brahmic-script transliteration, legal-form and address-abbreviation canonicalisation, DBA split, leetspeak repair, phonetic skeleton) |
| `data.py` | TSV I/O and cached parallel normalisation |
| `blocking.py` | Stage 2 — candidate generation (exact sparse TF-IDF top-k passes + reverse pass, within country) |
| `features.py` | Stage 3 — pair features (NaN for missing fields) and context features |
| `postprocess.py` | Stage 5 — metric (single implementation), one-owner assignment, two-threshold decision, threshold tuning |
| `pipeline.py` | Stage 0 diagnostics, Stage 4 LightGBM + GroupKFold CV, shift check, orchestration, output writing |

## Models and licences

| Component | Licence | Size |
|---|---|---|
| LightGBM (gradient-boosted trees, trained from scratch on the provided data) | MIT | ≪ 8B parameters (tree ensemble) |

No pretrained model, LLM or embedding model is used. Libraries: scikit-learn (BSD-3), RapidFuzz (MIT),
sparse_dot_topn (MIT), pandas / numpy / scipy (BSD).
