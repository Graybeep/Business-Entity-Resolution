# Business Entity Resolution

Match every Source-1 business record to all Source-2 / Source-3 records of the same real-world business
(name, address and country only; train = US + India, test adds France). Scored by per-entity F0.5, macro-averaged.

**Approach:** country-agnostic normalisation (incl. Brahmic-script transliteration and a phonetic key) →
exact sparse TF-IDF multi-pass blocking → LightGBM pair classifier → one-owner assignment + two thresholds
tuned on the exact metric.

| | |
|---|---|
| CV macro F0.5 (grouped 5-fold OOF, 330k S1) | **0.9648** |
| Blocking recall / reduction ratio (train) | 0.971 / 0.999993 |
| Test candidates | 63.8M pairs (36.9 per S1) |

## Layout

```
CLAUDE.md                                  # rules, architecture and decision log
rules_problem statement.txt                # competition statement
student_resource/
├── code/business_entity_resolution/       # pipeline (src/, README.md, requirements.txt)
├── utils/validate_submission.py           # provided validator
├── Documentation_template.md              # methodology write-up
├── make_package.py                        # builds the submission zip
└── output/run_summary.json                # diagnostics, CV, thresholds, shift check, importances
```

The dataset, caches and generated TSVs are not tracked (size). To reproduce, place the dataset in
`student_resource/dataset/` and follow `student_resource/code/business_entity_resolution/README.md`.
