# Business Entity Resolution — reproducible pipeline

Round-1 blocking (exact sparse TF-IDF top-k) → stage-1 LightGBM P(match) → round-2 candidates (empty-address name
pass + sibling expansion) → stacked stage-2 LightGBM (competitor context, sibling agreement, name frequency) →
entity-level decision (one-owner assignment + two thresholds tuned on grouped out-of-fold predictions for macro F0.5).

## Environment

- Python 3.11, CPU only. Development machine: 16 threads, 16 GB RAM.
- `pip install -r code/business_entity_resolution/requirements.txt` (versions pinned).
- The pipeline makes no network calls and uses no external data.

## Reproduce

Run from `student_resource/` (the directory that contains `dataset/` and `utils/`):

```bash
DATA=dataset CACHE=cache OUT=. bash code/business_entity_resolution/run_all.sh
cp runs/v4/matching_results.tsv runs/v4/candidate_pairs.tsv output/
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

`run_all.sh` runs, in order (each step caches its artefacts in `$CACHE`, so an interrupted run resumes):

| Step | Command | What it does |
|---|---|---|
| 1 | `pipeline.py --train-frac 0.4 --neg-sampling --tag v2 --skip-test` | diagnostics, normalisation, round-1 blocking of 40 % of the training S1s, disk-backed features, 5-fold grouped stage-1 CV with easy-negative sampling, thresholds, refit |
| 2 | `pipeline.py --tag v2 --predict-only` | round-1 test blocking and stage-1 scoring of every test candidate |
| 3 | `run10.py folds` | stage-1 fold models (used to score round-2 training pairs out of fold) |
| 4 | `run10.py expand train` / `r2feats train` | round-2 training candidates and their stage-1 features and scores |
| 5 | `run10.py s2train` | stage-2 grouped CV, thresholds, refit |
| 6 | `run10.py expand test` / `r2feats test` / `s2test` | round-2 test candidates, stage 2 per country, decision, output files |

Outputs of the final step (`runs/v4/`): `matching_results.tsv` (one row per test S1 entity), `candidate_pairs.tsv`
(exactly the pairs the model scores: round 1 + round 2), `test_summary.json`. CV scores and thresholds are in
`$CACHE/r10_train_result.json`; stage-1 diagnostics, blocking recall and shift check in `runs/v2/run_summary.json`.

## Runtime and memory

Measured on the development machine (16 threads, 16 GB RAM):

| Step | Time |
|---|---|
| normalisation (train + test) | 7 min |
| round-1 train blocking (40 % of S1) | 1 h 52 min |
| train features + stage-1 CV + thresholds + refit | 2 h 50 min |
| round-1 test blocking | 2 h 33 min |
| test stage-1 scoring | 1 h 25 min |
| stage-1 fold models | 1 h 15 min |
| round-2 train expansion + features | 1 h 20 min |
| stage 2 CV + refit | 30 min |
| round-2 test expansion + features | 2 h 30 min |
| test stage 2 + output | 1 h 05 min |
| **total from an empty cache** | **≈ 15 h** |

**RAM: at least 24 GB is recommended for a from-scratch run (32 GB comfortable).** Stage-1 CV binning peaks at
≈ 18 GB committed; it completed on 16 GB only with a large page file and other applications closed. Every later step
streams per country or from on-disk feature stores and stays at or below ≈ 12 GB. Disk: ≈ 40 GB for `$CACHE`.

**Smoke test (end to end from an empty cache, ~5 % of the data):**

```bash
python code/business_entity_resolution/src/smoke_dataset.py --src dataset --dst dataset_smoke5 --frac 0.05
DATA=dataset_smoke5 CACHE=cache_smoke OUT=runs/smoke bash code/business_entity_resolution/run_all.sh
```

## Source layout (`src/`)

| File | Role |
|---|---|
| `normalize.py` | country-agnostic normalisation (accent folding, Brahmic-script transliteration, legal-form and address canonicalisation, DBA split, leetspeak repair, phonetic skeleton) |
| `data.py` | TSV I/O and cached parallel normalisation |
| `blocking.py` | round-1 candidate generation (exact sparse TF-IDF top-k passes + reverse pass, within country) |
| `features.py` | pair features (NaN for missing fields) and context features |
| `training_scaled.py` | disk-backed grouped CV with easy-negative subsampling in training folds only |
| `pipeline.py` | diagnostics, stage-1 model / CV / thresholds, shift check, test blocking and stage-1 scoring |
| `train_fold_models.py` | regenerates the stage-1 fold models |
| `stage2_sibling.py` | sibling-agreement features (and the stage-2 experiment that introduced them) |
| `run10.py` | round-2 expansion, stage-2 stack, final decision and output writing |
| `postprocess.py` | metric (single implementation), one-owner assignment, two-threshold decision, threshold tuning |
| `density_match.py`, `ablation.py` | train/test competitor-density variants (see the documentation) |
| `blocking_passes_eval.py`, `shift_diag.py` | blocking-miss analysis and adversarial validation (diagnostics only) |
| `smoke_dataset.py` | 5 % sample of the dataset for the smoke test |

## Models and licences

| Component | Licence | Size |
|---|---|---|
| LightGBM (gradient-boosted trees, two stages, trained from scratch on the provided data) | MIT | ≪ 8B parameters (tree ensembles) |

No pretrained model, LLM or embedding model is used. Libraries: scikit-learn (BSD-3), RapidFuzz (MIT),
sparse_dot_topn (MIT), pandas / numpy / scipy (BSD), pyarrow (Apache-2.0), joblib (BSD-3).
