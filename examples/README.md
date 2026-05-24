# Examples

This directory intentionally does not contain official competition data.

To run the training code, download or mount the official dataset according to the competition rules and pass its path explicitly:

```bash
bash scripts/train_m148_example.sh /path/to/official/data
```

Expected data files:

```text
data/
├── schema.json
└── *.parquet
```

The public repository keeps only code and documentation. Do not commit official train/test data, generated predictions, platform logs, or checkpoints.
