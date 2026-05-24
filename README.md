# TencentUniRec-TAAC2026

[中文 README](README.zh-CN.md)

Unofficial solution notes and reference implementation for the **TAAC x KDD Cup 2026 Tencent Advertising Algorithm Competition, Industrial Track**.

> This repository is a cleaned public artifact derived from our competition workspace. It contains code and technical notes, but **does not include official data, checkpoints, private logs, platform paths, exact final submission recipes, or any non-public material**.

## Result

| Item | Value |
|---|---:|
| Track | Industrial Track |
| Result | Top 6% public leaderboard finish |
| Exact score / final recipe | Withheld from the public repository |
| Task | Large-scale advertising pCVR prediction |

## Highlights

- Starts from a cleaned competition baseline and documents how the solution evolved.
- Sequence-based pCVR modeling for industrial advertising recommendation data.
- Sparse/dense feature tokenization with RankMixer-style non-sequence tokens.
- Time-aware sequence buckets and public-tail-oriented validation.
- Multi-task click/conversion objective for regularizing sparse conversion labels.
- Auxiliary validation windows and leaderboard-correlation analysis for model selection.
- Controlled experiments covering validation design, feature engineering, objectives, seeds, checkpoint selection, and final-sprint ablations.

## Baseline To Final

| Stage | Main change |
|---|---|
| Early cleaned baseline | Fast sequence encoder on the initial HyFormer-style baseline |
| Time-bucket correction | Per-domain sequence recency treatment |
| Stronger baseline | More reliable temporal validation and auxiliary diagnostics |
| Fresh-tail family | Training/selection closer to public-adjacent tail windows |
| MTL family | Click/conversion multi-task regularization |
| Final selected family | Public-positive family selected by validation evidence and limited leaderboard checks |

The central lesson was that public score improvements came more from **validation alignment and objective calibration** than from simply adding larger or more complex modules.

## Repository Layout

```text
.
├── src/                       # Training, inference, dataset, model, trainer, EDA utilities
├── configs/                   # Public reference configs for key milestones
├── scripts/                   # Local example commands
├── docs/                      # Clean technical report and validation notes
├── experiments/               # Sanitized experiment summary tables
└── examples/                  # Small public placeholders; no official data included
```

## What Is Included

- A cleaned implementation of the competition model stack.
- Redacted reference configs for public study. Exact final run arguments are intentionally withheld while the competition/review context may still matter.
- Technical notes on temporal validation, model selection, and final-sprint lessons.
- A concise timeline and sanitized negative-result summary.

## What Is Not Included

- Official train/test data.
- Checkpoints or model outputs.
- Private platform logs, copied leaderboard screenshots, user IDs, or workspace paths.
- Any credential, account, or platform-specific runtime state.

To run the code, place the official competition data under a local `data/` directory or pass `--data_dir /path/to/data`.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/train_reference_example.sh /path/to/official/data
```

The example command is intended as a readable template, not as the exact final submission recipe. Exact platform scores require the official environment, full dataset, competition evaluation service, and private run records.

## Reading Order

1. [`docs/01_competition_overview.md`](docs/01_competition_overview.md)
2. [`docs/02_solution_report.md`](docs/02_solution_report.md)
3. [`docs/03_temporal_validation.md`](docs/03_temporal_validation.md)
4. [`docs/04_experiment_summary.md`](docs/04_experiment_summary.md)
5. [`docs/05_timeline.md`](docs/05_timeline.md)
6. [`docs/06_technical_report.md`](docs/06_technical_report.md)
7. [`docs/07_chinese_retrospective.md`](docs/07_chinese_retrospective.md)

## Citation

If you reference this repository, please cite it as an unofficial competition solution:

```text
TencentUniRec-TAAC2026: Unofficial TAAC x KDD Cup 2026 Industrial Track solution notes and implementation.
Top 6% public leaderboard finish; exact score and final recipe withheld from the public repository.
```

## Disclaimer

This project is not an official Tencent, TAAC, or KDD Cup repository. All competition names belong to their respective organizers. The implementation is provided for educational and portfolio purposes. See [`NOTICE.md`](NOTICE.md) for source, licensing, and data-handling notes.
