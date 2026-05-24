# Experiment Summary

The project ran many controlled experiments. This file records the cleaned public narrative rather than the full private work log.

## Milestone Families

| Family | Public readout | Contribution |
|---|---|---|
| Early baseline | Baseline | Fast and stable sequence backbone; validation gap remained large |
| Time-bucket correction | Positive | Corrected per-domain sequence recency treatment |
| Stronger baseline | Positive | More reliable temporal validation and auxiliary diagnostics |
| Fresh-tail family | Positive | Trained and selected closer to public-adjacent tail windows |
| MTL family | Positive | Click/conversion multi-task regularization improved the selected family |

## Process Timeline

| Phase | Focus | Outcome |
|---|---|---|
| Baseline reading | Understand official data, schema, baseline model, and platform constraints | Established a reproducible cleaned baseline and fast inference path |
| Architecture sanity checks | Compare sequence encoders and tokenization variants | Kept a fast HyFormer-style backbone as the practical default |
| Validation redesign | Investigate local-public gap and temporal distribution shift | Moved from local-AUC chasing to time-aware and auxiliary validation |
| Time/sequence feature work | Correct sequence time buckets and test time-aware features | Corrected per-domain buckets helped; naive absolute-time features did not transfer |
| Fresh-tail family | Train/select closer to public-adjacent tail windows | Became the main public-positive branch |
| MTL regularization | Add click/conversion multi-task objective | Improved the selected family |
| Final sprint | Narrow search around public-positive family | Most same-family tweaks did not change the final selection |

## Important Negative Results

| Direction | Public / offline readout | Lesson |
|---|---|---|
| Naive time feature stacking | Public-negative | Absolute-time features overfit local validation |
| History-CVR features | Public-negative | Historical aggregate features did not transfer in this setup |
| UE + explicit item interaction variants | Public-negative or unstable | High auxiliary AUC alone was not reliable |
| Checkpoint averaging | Negative or neutral | Exploratory side artifact; not used for final selected checkpoint |
| Broad seed search | Unreliable | Strong logloss alone did not transfer reliably |

These negative results are intentionally kept in the public artifact. They should be read as calibration evidence, not as noise: in this task, knowing which offline signals failed was essential for avoiding low-quality public submissions.

## Final Lesson

Most gains came from data/validation alignment and objective regularization, not from large architectural changes. In this industrial pCVR task, reliable model selection under temporal shift was the dominant constraint.
