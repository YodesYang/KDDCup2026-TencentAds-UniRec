# Solution Report

## Starting Baseline

We started from a HyFormer-style competition baseline for pCVR prediction. The baseline already had the right high-level decomposition for this task:

- sparse user/item features,
- dense user features,
- four user behavior sequence domains,
- sequence encoders,
- non-sequence feature tokens,
- binary conversion prediction with AUC as the target metric.

The early cleaned baseline established a fast, stable backbone, but it also exposed a large local-public validation gap. Most of the later work can be understood as turning this baseline from "locally strong" into "public-consistent."

## Model Family

The main model family is a HyFormer-style pCVR predictor with four behavior-sequence domains and non-sequence user/item tokens.

```text
user/item sparse + dense features -> NS tokenizers
4 behavior sequence domains       -> per-domain sequence encoders
NS tokens + sequence outputs      -> query decoding / token mixing
final representation              -> conversion prediction
```

Key implementation choices:

- RankMixer-style non-sequence tokenization for sparse user/item features.
- Fast sequence encoder as the practical default.
- Time-bucket embeddings for sequence recency.
- Item-conditioned query generation.
- Selected user-event dense handling.
- DIN-style target-aware interest signal.

## Validation

The largest source of error was validation mismatch. Early experiments showed that normal validation AUC could improve while public AUC dropped. We therefore introduced:

- time-ordered row-group validation,
- fixed clean validation windows,
- auxiliary validation windows,
- public-leaderboard correlation tracking,
- post-hoc rejection of features that improved normal validation but failed cleaner windows.

This changed the workflow from "pick the highest local AUC" to "pick the most public-consistent evidence chain."

## Feature And Objective Work

Positive or useful directions:

- corrected per-domain time buckets,
- fresher tail-oriented validation/training setup,
- multi-task learning with click and conversion heads,
- careful checkpoint selection under a limited public-evaluation budget.

Negative or non-transferring directions in our runs:

- naive absolute-time feature stacking,
- history-CVR features,
- several high-auxiliary-score user-event interaction variants,
- exploratory checkpoint averaging,
- broad seed search as a primary strategy.

We keep these negative results in the public write-up because they are part of the technical contribution: they show where local validation was misleading and how the final selection policy became more conservative.

## Public Release Boundary

This public report intentionally omits exact leaderboard scores for each milestone, final submission arguments, final validation windows, and final checkpoint-selection records.
