## Experiments

This project uses **MLflow** to track:

- **Parameters** (flattened YAML config)
- **Metrics** (epoch-level validation, final test metrics)
- **Artifacts** (trained model checkpoint, per-user reports, comparison tables)

### Where artifacts go

- Local MLflow store (default): `mlruns/`
- Project artifacts (checked by the pipeline): `artifacts/`

### Typical workflow

From the repo root (`metabolic-intelligence/`):

```bash
make simulate
make preprocess
make train        # logs a "global_*" run
make finetune     # logs a "personalization_finetune" run
make mlflow       # open the UI
```

### What to compare in the UI

- **Global model**: `val_mse`, `val_mae`, `test_mse`, `test_mae`
- **Personalization**: `avg_global_mse` vs `avg_finetuned_mse` and the CSV artifact:
  `artifacts/reports/personalization_comparison.csv`

