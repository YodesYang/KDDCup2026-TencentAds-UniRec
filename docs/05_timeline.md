# Project Timeline

This is a cleaned timeline for the public repository. It preserves the technical story without exposing private platform logs, exact final recipes, or operational details.

## Phase 1: Baseline And Problem Framing

We first studied the official schema, baseline code, and platform constraints. The initial working model family was a HyFormer-style pCVR predictor with sparse user/item embeddings, dense user features, four behavior sequence domains, non-sequence tokens, sequence encoders, and a conversion probability output.

Key lesson:

```text
The baseline was not weak because it lacked capacity only; it was weak because local validation did not reliably match public evaluation.
```

## Phase 2: Architecture And Tokenization Checks

We tested sequence encoder and tokenization directions, including Transformer-like encoders, LONGER-style ideas, RankMixer-style non-sequence tokens, and target-aware interactions.

The practical conclusion was to keep the fast HyFormer-style backbone and spend more effort on validation and data alignment. Inference speed and platform stability mattered because public evaluation had strict runtime constraints.

## Phase 3: Temporal Validation Redesign

The biggest change in thinking came from temporal distribution shift. Normal validation could improve while public AUC dropped.

We added:

- time-ordered row-group validation,
- clean fixed-window diagnostics,
- auxiliary validation windows,
- public-leaderboard correlation tracking.

This changed model selection from:

```text
highest local AUC wins
```

to:

```text
local AUC + auxiliary windows + public-calibrated family evidence
```

## Phase 4: Corrected Time Buckets And Stronger Baseline

Correcting per-domain sequence time buckets produced a meaningful public improvement. This was the first stable path above the earlier baseline family.

## Phase 5: Fresh-Tail Training And MTL

The next improvement came from training/selecting closer to the public-adjacent tail distribution. Multi-task learning used click and conversion heads to regularize sparse conversion labels. This was one of the few objective changes that transferred positively.

## Phase 6: Final Sprint

The final sprint focused on the public-positive family rather than broad exploration. Exact final run arguments and checkpoint-selection records are withheld from the public repository.

## Negative Routes We Kept

We mention failures because they were important to the final decision process:

- Naive absolute-time features improved some local signals but did not transfer.
- History-CVR features looked attractive offline but were public-negative.
- Some user-event interaction variants produced high auxiliary AUC but weak public transfer.
- Exploratory checkpoint averaging did not improve the selected family and was not used for the final selected checkpoint.
- Seed and logloss-only selection were not reliable enough for final public submissions.

The broader lesson is that in temporally shifted industrial recommendation tasks, **failure analysis is part of the solution**.
