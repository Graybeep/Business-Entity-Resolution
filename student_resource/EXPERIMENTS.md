# EXPERIMENTS

One line per run (submitted or not): what changed, CV delta, keep/revert. Newest last.
CV = grouped 5-fold OOF macro F0.5 over the training S1 subset, thresholds tuned on OOF.
Gains below ~0.003 are treated as noise.

| # | Date (IST) | Run | Change | Blocking recall | CV macro F0.5 | Δ | Shift check (held-out India / US, retuned) | Decision |
|---|---|---|---|---|---|---|---|---|
| 0 | 25 Sep | blocking v0 | char-TF-IDF → SVD embeddings, GPU top-k (100k S1 sample) | 0.941 @ k=30 | — | — | — | **revert**: SVD drops rare n-grams |
| 1 | 25 Sep | blocking v1 | exact sparse top-k: word na / char4 / phonetic / name + reverse, k=(10,5,5,5), rev 2 | **0.971**, RR 0.999993 | — | — | — | keep |
| 2 | 25 Sep | run1 | baseline LightGBM, 15 % train S1 (12.1M pairs), lr 0.08 | 0.971 | 0.9510 | — | — | superseded |
| 3 | 25 Sep | ablation A/B/C | 10 % subset, identical folds: A base 0.9460, B +number & name-coverage features 0.9588, C +stage-2 stack 0.9594 | 0.971 | B: +0.0128, C: +0.0006 | — | — | keep B; C off (noise) |
| 4 | 25 Sep | run2 (**submitted file**) | B features + leading-zero normalisation, 15 % S1, stage 2 off | 0.971 | **0.9648** | +0.0138 vs run1 | 0.861 / 0.955 (other-country thr: 0.858 / 0.914) | keep — current model |
| 5 | 25 Sep | writer check | regenerate outputs from cached test probs with current writer | — | — | — | — | byte-identical (sha256 7b21…64d2 / d244…4195) |
| 6 | 25 Sep | France probe | `t_first` +0.05 for France S1 only → `submissions/sub-probe-france.tsv` | — | — | — | — | 326 France rows non-empty → empty, 0 other rows changed; validator PASS. Max LB effect ≈ ±0.0002 (likely below LB resolution) |
| 7 | 25 Sep | error breakdown | run2 OOF loss 0.0353 = (a) singleton FP 0.0039 (11 %) + (b) FP on matched 0.0084 (24 %) + (c) misses 0.0229 (65 %; blocking 0.0106, model 0.0124) | — | — | — | — | recall is the main lever |
| 8 | 25–26 Sep | v2 scaled training | 40 % train S1 (882,771 S1, 32.3M pairs), disk-backed float32 features, training folds keep all positives + hard negatives (base-score rank ≤ 3 per S1/source or record's best S1) + 25 % of easy negatives (16.1M rows); validation folds full | 0.9717, RR 0.999993 | re-scored (fixed grid): nested 15 % **0.9677** (tf .764 / ta .709) vs run 2 same tuner 0.9654; full 40 % 0.9677 (tf 0.725 / ta 0.705, reference only). Logged 0.9560 invalid (broken grid) | +0.0024 (< 0.003 bar) | India 0.8629 / US 0.9592 (cross-thr: 0.8593 / 0.9253) vs run 2 0.8607 / 0.9550 (0.8578 / 0.9138) | below 0.003 bar by rule; user chose test inference anyway -> `runs/v2/{matching_results,candidate_pairs}.tsv`, validator PASS (incl. --check-ids), thr .725/.705; 5.48M pairs vs 5.56M, singletons 6.33 % vs 6.04 % (France 6.39 % vs 5.96 %), 92.1 % of S1 lists identical. `output/` still run 2 |
| 9 | 26 Sep | threshold-grid bug | `tune_thresholds` took quantiles over ALL pair scores (~97 % near 0): n_q=30 grid jumps 0.021 -> 0.99, tuned 0.95 -> 0.9536 on run-2 OOF. Fix: quantiles of scores >= 0.01. Run-2 OOF re-tuned: n_q=30 0.9646, n_q=50 **0.9654** (tf .776 / ta .736) vs 0.9648 | — | +0.0006 (noise) | — | keep fix (correctness); submission thresholds unchanged. **Run 8 was launched before the fix and uses n_q=30 -> its logged CV is invalid; re-score `cache/v2_oof_pairs.parquet` offline** |
| 10 | 26 Sep | expected-F0.5 decision | per S1, prefix of owned candidates maximising Monte-Carlo E[F0.5] (calibration power a, unseen-match rate lam) on run-2 OOF | — | 0.9633 (a=.8, best of first 3 settings) | -0.0015 | — | **drop**: below the two-threshold rule; stopped early |
| 11 | 26 Sep | error analysis (run-2 OOF) | FP 14.8k pairs: 53 % same core name + different numbers ("twins"); 39 % are records owned by an S1 outside the 15 % subset (subset artefact). FN (model) 41.8k pairs: 90 % numbers differ, **46 % empty address on one side**. Empty-addr: record core name unique in S1 for 21 % of FN vs 5 % of FP | — | — | — | — | candidate feature: core-name frequency (S1 / S2+S3, per country) |

## Notes

- **Run 8 departs from the decision log** ("no resampling"): explicitly requested experiment. Validation
  folds and threshold tuning use full candidate sets, so OOF-tuned thresholds match the subsampled
  model's calibration. Enabled only via `--neg-sampling`; default config is unchanged.
- **Comparison protocol for run 8:** the 40 % mask is a strict superset of the 15 % mask (same uniform
  draw per S1). The headline comparison against 0.9648 is the score on the nested 15 % S1 set with the
  decision restricted to those S1s (same competitor set as run 2). A larger S1 pool alone raises CV
  through one-owner assignment, so the full-40 % number is reported but not used for the keep decision.
