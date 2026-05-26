# Competition Overview

TAAC x KDD Cup 2026 Tencent Advertising Algorithm Competition, Industrial Track, is a large-scale advertising recommendation task. The target is pCVR prediction:

```text
pCVR = P(conversion = 1 | ad, user, context)
```

The industrial track uses anonymized real-world recommendation features, including sparse user/item fields, dense user fields, and four domains of user behavior sequences. The leaderboard metric is AUC.

## Our Result

| Metric | Value |
|---|---:|
| Rank | 35/689 |
| Percentile | Top 5.1% |
| Best public AUC | 0.851365 |
| Best model family | M148 |

Final run arguments and checkpoint-selection records are intentionally withheld from this public repository.

## Main Technical Problem

The core difficulty was not only model architecture. The most important practical challenge was **train/validation/test distribution shift**, especially temporal shift. Standard local validation could look strong while public leaderboard performance degraded.

The project therefore focused on:

- a sequence-aware recommendation backbone,
- robust validation under temporal shift,
- controlled feature and objective ablations,
- cautious checkpoint selection under limited public evaluations.

## Caveat

This repository intentionally omits official data and private platform artifacts. Reported conclusions are cleaned technical notes rather than a standalone reproducibility guarantee without the official environment.
