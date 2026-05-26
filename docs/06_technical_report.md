# Technical Report

## Temporal Validation and Sequence Modeling for Industrial pCVR Prediction

This report summarizes our unofficial solution notes for the **TAAC x KDD Cup 2026 Tencent Advertising Algorithm Competition, Industrial Track**.

| Item | Value |
|---|---:|
| Team rank | 35/689 |
| Percentile | Top 5.1% |
| Best public AUC | 0.851365 |
| Final model family | M148 |
| Task | Industrial advertising pCVR prediction |

Final run arguments, final validation windows, and checkpoint-selection records are intentionally withheld from this public repository.

## Abstract

The competition task was to predict post-click conversion probability from anonymized industrial advertising recommendation data. The data contained sparse user/item features, dense user features, and four domains of historical user behavior sequences.

Our final public score was **0.851365 AUC**, ranking **35/689** in the Industrial Track. The main improvement did not come from a single large architectural change. Instead, it came from progressively aligning the validation setup with the temporally shifted public test distribution, then refining a sequence-based recommendation model under that validation system.

The most important technical lesson was that normal local validation AUC was often misleading. We therefore built a selection process around time-aware splits, fixed clean windows, auxiliary validation windows, and public-leaderboard calibration.

## 1. Task

The objective is binary pCVR prediction:

```text
pCVR = P(conversion = 1 | ad, user, context)
```

The competition metric is AUC.

The input schema contains:

- sparse user features,
- sparse item/ad features,
- dense user features,
- four domains of user behavior sequences,
- per-domain sequence timestamps and side-information features.

The public test set is a future time slice relative to the training data. This temporal shift became the main challenge in model selection.

## 2. Starting Baseline

We started from a HyFormer-style competition baseline. The baseline had the right high-level structure:

```text
user/item sparse + dense features -> non-sequence tokens
behavior sequence domains         -> sequence encoders
sequence outputs + NS tokens      -> query decoding / token mixing
final representation              -> pCVR prediction
```

However, early experiments showed a large local-public gap. Local validation could improve while public AUC dropped, so the baseline was not mainly limited by raw capacity. It was limited by validation mismatch and distribution shift.

## 3. Model

### 3.1 Backbone

The selected family keeps a HyFormer-style architecture:

- non-sequence user/item tokenization,
- per-domain sequence tokenization,
- time-bucket embeddings for sequence recency,
- per-domain sequence encoding,
- query-based sequence aggregation,
- token mixing across sequence-derived query tokens and non-sequence tokens,
- binary or multi-task output heads.

### 3.2 Non-Sequence Tokenization

We use RankMixer-style non-sequence tokens for sparse user/item features. Instead of treating every feature as an independent token, sparse embeddings are grouped into a smaller fixed number of user and item tokens.

The token budget matters because the token mixing path requires the model dimension to be compatible with the total token count.

### 3.3 Sequence Modeling

The practical default sequence encoder is a fast SwiGLU-style encoder. Transformer-like sequence encoders were considered, but the fast backbone offered a better speed/stability tradeoff in our runs.

Correcting per-domain time buckets was one of the first public-positive changes. It improved the model's treatment of sequence recency without relying on brittle absolute-time features.

### 3.4 Target-Aware Signals

The selected family also used target-aware signals, including item-conditioned query generation, selected user-event dense handling, and DIN-style interest modeling from encoded sequences.

### 3.5 Multi-Task Learning

Click/conversion multi-task learning regularizes sparse conversion labels with a related click signal. In our runs, this objective improved the selected public-positive family.

## 4. Validation System

### 4.1 Why Normal Validation Failed

The public test set is a future time period. Standard validation could overlap temporally with training or fail to represent the public distribution. As a result, several directions improved normal validation AUC while degrading public AUC.

This affected:

- absolute-time features,
- history-CVR features,
- some user-event interaction variants,
- seed/logloss-only selection,
- exploratory checkpoint averaging.

### 4.2 Clean Windows And Auxiliary Validation

We introduced multiple validation signals:

- time-ordered row-group splits,
- fixed clean validation windows,
- auxiliary validation windows logged per epoch,
- public leaderboard correlation tracking.

The final decision process was:

```text
normal validation
  -> auxiliary window behavior
  -> consistency with public-positive model family
  -> scarce public evaluation
```

This was not a perfect offline oracle, but it prevented many high-risk public submissions.

## 5. Results

The public repository keeps qualitative milestone information only:

| Family | Readout | Main contribution |
|---|---|---|
| Early baseline | Baseline | Fast sequence backbone |
| Time-bucket correction | Positive | Corrected per-domain time buckets |
| Stronger baseline | Positive | Auxiliary validation logging and stronger temporal diagnostics |
| Fresh-tail family | Positive | Training/selection closer to public-adjacent tail windows |
| MTL family | Positive | Click/conversion multi-task learning |

Exact public AUC by run, final run arguments, and final checkpoint-selection records are withheld.

## 6. Negative Findings

The following negative results were important because they calibrated the validation system:

| Direction | Observation | Lesson |
|---|---|---|
| Absolute-time feature stacking | Public-negative despite local gains | Time features can overfit a known window |
| History-CVR features | Public-negative | Aggregate historical statistics did not transfer in this setup |
| UE + explicit item interaction variants | High auxiliary signals but weak public result | High auxiliary AUC alone is unsafe |
| Exploratory checkpoint averaging | Negative or neutral | Side artifact; not used for final selected checkpoint |
| Seed / low-logloss selection | Public-negative or unreliable | Logloss or familiar seed alone is not a reliable selector |

We consider these negative findings part of the solution because they shaped the final public-evaluation policy.

## 7. Discussion

From a technical perspective, the project produced useful lessons:

1. In temporally shifted industrial recommendation tasks, validation design can matter more than model size.
2. Public-positive experiment families are more reliable than isolated high local AUC runs.
3. Multi-task learning can be a practical regularizer for sparse conversion labels.
4. Failure analysis is not bookkeeping; it is part of model selection.
5. Limited public evaluation quota turns the competition into a decision-making problem, not just a modeling problem.

## 8. Limitations

This public repository does not include:

- official data,
- checkpoints,
- private logs,
- platform runtime files,
- exact leaderboard submission artifacts,
- exact final submission recipes.

The reported conclusions require the official competition environment and evaluation service. This repository is intended as a readable technical artifact and a reference implementation, not a complete standalone reproduction package.

## 9. Future Work

Promising next steps:

- rolling temporal cross-validation,
- stronger delayed-feedback modeling,
- distribution-distance weighting for validation windows,
- better uncertainty estimates for tiny validation tails,
- more principled target-aware sequence interaction,
- calibration-aware objective design beyond AUC-only selection.
