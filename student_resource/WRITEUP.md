# Business Entity Resolution — solution write-up

**Team:** grain · **Final submission:** run 10 (public LB 0.964)

## 1. Approach

For every Source-1 business we must return the Source-2/3 records of the same business from name, address and
country alone. Our pipeline has five stages, identical for every country (France never appears in training):

1. **Normalisation.** Accent folding; transliteration of all Brahmic scripts through one offset table; legal-form and
   address canonicalisation (including French forms); DBA / aka splitting; leetspeak repair; house numbers and postal
   code extracted; a script-agnostic **phonetic skeleton** of the name (`gud treding` ≡ `good trading`).
2. **Round-1 blocking.** Four exact sparse TF-IDF top-k passes (word, char 4-gram, phonetic, name-only) plus a reverse
   pass, within country: recall 0.972 at 37 candidates per S1.
3. **Stage 1.** A LightGBM classifier over 67 pair features (fuzzy and TF-IDF name/address similarity, IDF-weighted
   name coverage, number agreement, retrieval ranks, competitor context, missingness as NaN). It is trained on 40 % of
   the training entities with disk-backed features and easy-negative subsampling in the training folds only.
4. **Round 2 + stage 2.** Using stage-1 scores we add candidates the passes miss: records with an empty address
   matched by name only, and **sibling expansion**, i.e. the nearest records of each S1's five best candidates
   (duplicates of one business resemble each other more than they resemble the S1). Recall rises to 0.984 at 53
   candidates per S1. A stacked LightGBM then re-scores plausible pairs using competitor context, **sibling
   agreement** (does this record agree with the S1's other confident matches on name, address and house number?)
   and name frequency (is this exact name unique in the country?).
5. **Decision.** Each record goes to its highest-scoring S1 (one owner; no record has two owners in training). An S1
   keeps its best candidate if P ≥ t_first and further ones if P ≥ t_add, both tuned jointly on grouped out-of-fold
   predictions against the exact macro F0.5.

**Models:** LightGBM (MIT), two stages trained from scratch. No external data, no pretrained or language model.

## 2. Experiments

| Step | Grouped CV macro F0.5 | Public LB |
|---|---|---|
| Base features (10 % of entities, ablation) | 0.946 | — |
| + number and asymmetric name-coverage features (15 % of entities) | 0.965 | 0.952 |
| + 40 % of entities (disk-backed, negative sampling) | 0.968 | — |
| + stage 2 with sibling agreement | 0.972 | 0.959 |
| + round-2 candidates and name frequency (**run 10**) | **0.976** | **0.964** |

What did not help: SVD embeddings with GPU search for blocking (recall 0.94); exact postal-plus-number and
rarest-token blocking keys (recall +0.0001 / +0.0006); expected-F0.5 subset selection instead of two thresholds
(−0.0015); a France-specific threshold (only ~1 % of France rows sit near the threshold).

**Validation.** A US ↔ India country-shift check stands in for France: the stage-2 layer still adds +0.002 to +0.003
when trained on the other country. A diagnostic submission with every France row empty measured France at about 0.95,
close to India. The CV → leaderboard gap stayed at −0.012 to −0.013 across three submissions, so offline gains transferred.

**Known shift.** Training blocks only a subset of entities, so in training each record meets ~40 % of the competing S1s
it faces at test time. Adversarial validation (AUC 0.91) traced the train/test difference to the record-side
competition features. Test-time density matching, which restores the training competitor count without retraining,
scored 0.963 against run 10's 0.964, so the shift is real but is not what separates CV from the leaderboard; a model
without those features lost 0.0009 CV and was not submitted.

## 3. Conclusion

Most of the gain came from two ideas: model-guided candidate expansion through **siblings**, and a second stage that
**checks every candidate against the S1's other confident matches**. Together they add +0.008 CV over a strong
single-stage matcher. The remaining loss is India blocking (transliterated names in crowded generic-name groups, 42 %
of India's loss) and the model's handling of empty-address records. Blocking and training on every entity (≥ 24 GB RAM)
would remove the train/test competitor-density shift and is the first thing we would do with more memory.
