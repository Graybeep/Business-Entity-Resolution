#!/usr/bin/env bash
# End-to-end pipeline (run 10 configuration): regenerates both output files from the dataset directory.
#
#   DATA=dataset CACHE=cache OUT=. bash code/business_entity_resolution/run_all.sh
#
# Run from student_resource/.  Every step caches its artefacts in $CACHE and can be re-run alone.
# Full data: ~15 h and >= 24 GB RAM from an empty cache (see README).  Smoke test: DATA=dataset_smoke5.
set -euo pipefail
DATA=${DATA:-dataset}
CACHE=${CACHE:-cache}
OUT=${OUT:-.}
SRC=code/business_entity_resolution/src

# Stages 0-4 (stage 1): diagnostics, normalisation, blocking (40 % of training S1s), features,
# grouped 5-fold CV with easy-negative subsampling, thresholds, refit.
python -u $SRC/pipeline.py --data-dir "$DATA" --cache "$CACHE" --out "$OUT/runs/v2" \
    --train-frac 0.4 --neg-sampling --tag v2 --skip-test
# Test blocking + stage-1 scoring of every test candidate.
python -u $SRC/pipeline.py --data-dir "$DATA" --cache "$CACHE" --out "$OUT/runs/v2" --tag v2 --predict-only

# Round 2 (candidate expansion) + stage 2 (sibling-agreement stack), train then test.
export ER_DATA="$DATA" ER_CACHE="$CACHE" ER_OUT="$OUT"
for step in "folds" "expand train" "r2feats train" "s2train" "expand test" "r2feats test" "s2test"; do
    python -u $SRC/run10.py $step
done

python utils/validate_submission.py --matching "$OUT/runs/v4/matching_results.tsv" \
    --candidate "$OUT/runs/v4/candidate_pairs.tsv" --test-dir "$DATA/test"
echo "final outputs: $OUT/runs/v4/{matching_results,candidate_pairs}.tsv"
