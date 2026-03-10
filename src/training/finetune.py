from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import WearableSequenceDataset
from src.models.factory import build_model
from src.training.mlflow_utils import flatten_dict
from src.utils.artifacts import load_feature_spec
from src.utils.config import get_mlflow_tracking_uri, load_yaml
from src.utils.logging import setup_logging
from src.utils.metrics import mae, mse
from src.utils.seed import set_seed


def _device(name: str) -> torch.device:
    if name.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _eval(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    preds = []
    tgts = []
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        preds.append(model(x).detach().cpu())
        tgts.append(y.detach().cpu())
    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(tgts, dim=0)
    return float(mse(pred, tgt).item()), float(mae(pred, tgt).item())


def _freeze_backbone(model: nn.Module) -> None:
    # Heuristic: freeze obvious backbone modules and keep prediction head trainable.
    for name, p in model.named_parameters():
        p.requires_grad = True
        if name.startswith("lstm.") or name.startswith("encoder.") or name.startswith("input_proj.") or name.startswith("pos."):
            p.requires_grad = False
        # Always keep head trainable if present
        if ".head." in name or name.startswith("head."):
            p.requires_grad = True


def _train_user(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    patience: int = 2,
) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.MSELoss()
    best = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad = 0

    for _epoch in range(1, int(epochs) + 1):
        model.train()
        for x, y, _ in tqdm(train_loader, desc="finetune", leave=False):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            opt.step()

        val_mse, _ = _eval(model, val_loader, device)
        if val_mse < best_val:
            best_val = val_mse
            best = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-user personalization via fine-tuning (MLflow tracked).")
    parser.add_argument("--config", type=str, required=True, help="Path to finetune.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logging("finetune")
    set_seed(int(cfg.get("seed", 42)))
    device = _device(str(cfg.get("device", "cpu")))
    logger.info("Using device=%s", device.type)

    tracking_uri = get_mlflow_tracking_uri(cfg["mlflow"].get("tracking_uri"))
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(cfg["mlflow"]["experiment_name"]))

    processed_dir = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path(cfg["artifacts"]["output_dir"])
    global_model_path = Path(cfg["artifacts"]["global_model_path"])
    feature_spec_path = artifacts_dir / "features" / "feature_spec.json"

    spec = load_feature_spec(feature_spec_path)
    ckpt = torch.load(global_model_path, map_location=device)
    base_model = build_model(ckpt["model_cfg"], input_dim=len(spec.feature_cols)).to(device)
    base_model.load_state_dict(ckpt["state_dict"])

    # Determine held-out users from the global user split file.
    split_file = processed_dir / "splits_global_users.json"
    split_data = json.loads(split_file.read_text(encoding="utf-8"))
    heldout_users: List[str] = list(split_data["test_users"])
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    rng.shuffle(heldout_users)
    heldout_users = heldout_users[: int(cfg["personalization"]["n_users_to_finetune"])]
    logger.info("Fine-tuning on %d held-out users.", len(heldout_users))

    rows: List[Dict[str, Any]] = []
    with mlflow.start_run(run_name="personalization_finetune"):
        mlflow.log_params(flatten_dict(cfg))
        mlflow.log_param("n_features", len(spec.feature_cols))

        for user_id in heldout_users:
            user_train = WearableSequenceDataset(
                processed_dir=processed_dir,
                feature_spec_path=feature_spec_path,
                split="train",
                split_mode="within_user",
                users_subset=[user_id],
            )
            user_val = WearableSequenceDataset(
                processed_dir=processed_dir,
                feature_spec_path=feature_spec_path,
                split="val",
                split_mode="within_user",
                users_subset=[user_id],
            )
            user_test = WearableSequenceDataset(
                processed_dir=processed_dir,
                feature_spec_path=feature_spec_path,
                split="test",
                split_mode="within_user",
                users_subset=[user_id],
            )

            bs = int(cfg["data"]["batch_size"])
            nw = int(cfg["data"]["num_workers"])
            train_loader = DataLoader(user_train, batch_size=bs, shuffle=True, num_workers=nw)
            val_loader = DataLoader(user_val, batch_size=bs, shuffle=False, num_workers=nw)
            test_loader = DataLoader(user_test, batch_size=bs, shuffle=False, num_workers=nw)

            # Global-only evaluation on this user's test split
            global_mse, global_mae = _eval(base_model, test_loader, device)

            # Fine-tune copy
            ft_model = copy.deepcopy(base_model)
            if bool(cfg["personalization"]["freeze_backbone"]):
                _freeze_backbone(ft_model)

            ft_model = _train_user(
                ft_model,
                train_loader,
                val_loader,
                device,
                epochs=int(cfg["personalization"]["finetune_epochs"]),
                lr=float(cfg["personalization"]["finetune_lr"]),
                weight_decay=float(cfg["personalization"]["weight_decay"]),
                grad_clip_norm=float(cfg["personalization"]["grad_clip_norm"]),
            )
            ft_mse, ft_mae = _eval(ft_model, test_loader, device)

            rows.append(
                {
                    "user_id": user_id,
                    "global_mse": global_mse,
                    "global_mae": global_mae,
                    "finetuned_mse": ft_mse,
                    "finetuned_mae": ft_mae,
                    "mse_improvement": global_mse - ft_mse,
                    "mae_improvement": global_mae - ft_mae,
                    "n_test_samples": len(user_test),
                }
            )
            logger.info(
                "user=%s | global(mse=%.3f mae=%.3f) -> finetuned(mse=%.3f mae=%.3f)",
                user_id,
                global_mse,
                global_mae,
                ft_mse,
                ft_mae,
            )

        report = pd.DataFrame(rows).sort_values("mse_improvement", ascending=False).reset_index(drop=True)
        reports_dir = artifacts_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_csv = reports_dir / "personalization_comparison.csv"
        report.to_csv(out_csv, index=False)
        mlflow.log_artifact(out_csv.as_posix(), artifact_path="reports")

        mlflow.log_metric("avg_global_mse", float(report["global_mse"].mean()))
        mlflow.log_metric("avg_finetuned_mse", float(report["finetuned_mse"].mean()))
        mlflow.log_metric("avg_global_mae", float(report["global_mae"].mean()))
        mlflow.log_metric("avg_finetuned_mae", float(report["finetuned_mae"].mean()))

        summary = {
            "n_users": int(len(report)),
            "avg_mse_improvement": float(report["mse_improvement"].mean()),
            "avg_mae_improvement": float(report["mae_improvement"].mean()),
        }
        (reports_dir / "personalization_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        mlflow.log_artifact((reports_dir / "personalization_summary.json").as_posix(), artifact_path="reports")
        logger.info("Wrote comparison table to %s", out_csv.as_posix())


if __name__ == "__main__":
    main()

