# Temporal Validation Notes

## Why Temporal Validation Mattered

The public test period was adjacent to, but later than, the training timeline. Standard row-group validation could contain substantial temporal overlap or distribution mismatch, causing local AUC to overstate public performance.

This created two recurring failure modes:

1. **Normal validation false positives**: features improved local validation but hurt public AUC.
2. **Tiny-window overfitting**: very small validation tails could become noisy selectors.

## Validation Signals Used

We used several complementary signals:

- `valid_split_strategy=time`: sort row groups by time before splitting.
- Fresh-tail validation ratios: small tail windows used as public-adjacent diagnostics.
- Fixed clean validation windows: pseudo-public windows used as diagnostics.
- Auxiliary validation logging: evaluate extra windows per epoch while keeping the main best-model selector unchanged.
- Public-LB calibration: maintain a table of which offline signals transferred.

## Practical Lesson

The most useful validation system was not a single perfect offline metric. It was a tiered decision process:

```text
normal validation -> auxiliary windows -> public-calibrated experiment family -> scarce public eval
```

This was especially important under daily evaluation quotas.

## What We Would Improve Next

If continuing the project, the next validation work would be:

- rolling time-window cross-validation,
- distribution-distance weighted validation windows,
- more explicit delayed-feedback modeling,
- public-style sample weighting,
- stronger uncertainty estimates for tiny validation tails.
