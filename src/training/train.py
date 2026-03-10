from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import numpy as np
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    model.eval()
    preds = []
    targets = []
    users = []
    for x, y, user_id in loader:
        x = x.to(device)
        y = y.to(device)
        p = model(x)
        preds.append(p.detach().cpu())
        targets.append(y.detach().cpu())
        users.extend(list(user_id))

    pred = torch.cat(preds, dim=0)
    tgt = torch.cat(targets, dim=0)
    global_mse = float(mse(pred, tgt).item())
    global_mae = float(mae(pred, tgt).item())

    per_user: Dict[str, Dict[str, float]] = {}
    users_arr = np.array(users)
    for u in np.unique(users_arr):
        m = users_arr == u
        pu_pred = pred[m]
        pu_tgt = tgt[m]
        per_user[str(u)] = {
            "mse": float(mse(pu_pred, pu_tgt).item()),
            "mae": float(mae(pu_pred, pu_tgt).item()),
            "n": int(m.sum()),
        }
    return global_mse, global_mae, per_user


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    losses = []
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train global glycemic forecasting model (MLflow tracked).")
    parser.add_argument("--config", type=str, required=True, help="Path to train.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logging("train")
    set_seed(int(cfg.get("seed", 42)))

    device = _device(str(cfg.get("device", "cpu")))
    logger.info("Using device=%s", device.type)

    tracking_uri = get_mlflow_tracking_uri(cfg["mlflow"].get("tracking_uri"))
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(str(cfg["mlflow"]["experiment_name"]))

    processed_dir = Path(cfg["data"]["processed_dir"])
    artifacts_dir = Path(cfg["artifacts"]["output_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "models").mkdir(parents=True, exist_ok=True)

    feature_spec_path = artifacts_dir / "features" / "feature_spec.json"
    spec = load_feature_spec(feature_spec_path)

    train_ds = WearableSequenceDataset(
        processed_dir=processed_dir,
        feature_spec_path=feature_spec_path,
        split="train",
        split_mode="global_users",
        seed=int(cfg.get("seed", 42)),
    )
    val_ds = WearableSequenceDataset(
        processed_dir=processed_dir,
        feature_spec_path=feature_spec_path,
        split="val",
        split_mode="global_users",
        seed=int(cfg.get("seed", 42)),
    )
    test_ds = WearableSequenceDataset(
        processed_dir=processed_dir,
        feature_spec_path=feature_spec_path,
        split="test",
        split_mode="global_users",
        seed=int(cfg.get("seed", 42)),
    )

    loader_cfg = cfg["data"]
    train_loader = DataLoader(train_ds, batch_size=int(loader_cfg["batch_size"]), shuffle=True, num_workers=int(loader_cfg["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=int(loader_cfg["batch_size"]), shuffle=False, num_workers=int(loader_cfg["num_workers"]))
    test_loader = DataLoader(test_ds, batch_size=int(loader_cfg["batch_size"]), shuffle=False, num_workers=int(loader_cfg["num_workers"]))

    model_cfg = cfg["model"]
    model = build_model(model_cfg, input_dim=len(spec.feature_cols)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    best_val_mse = float("inf")
    best_path = artifacts_dir / "models" / "global_model.pt"
    patience = int(cfg["training"]["early_stopping_patience"])
    bad_epochs = 0

    with mlflow.start_run(run_name=f"global_{model_cfg['name']}"):
        mlflow.log_params(flatten_dict(cfg))
        mlflow.log_param("n_features", len(spec.feature_cols))
        mlflow.log_param("window_steps", int(spec.window_steps))
        mlflow.log_param("horizon_steps", int(spec.horizon_steps))

        for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
            )
            val_mse, val_mae, _ = evaluate(model, val_loader, device)

            mlflow.log_metric("train_mse_loss", train_loss, step=epoch)
            mlflow.log_metric("val_mse", val_mse, step=epoch)
            mlflow.log_metric("val_mae", val_mae, step=epoch)
            logger.info("epoch=%d train_mse=%.4f val_mse=%.4f val_mae=%.4f", epoch, train_loss, val_mse, val_mae)

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                bad_epochs = 0
                torch.save(
                    {
                        "model_name": str(model_cfg["name"]),
                        "model_cfg": model_cfg,
                        "state_dict": model.state_dict(),
                        "feature_spec_path": str(feature_spec_path),
                    },
                    best_path,
                )
                mlflow.log_artifact(best_path.as_posix(), artifact_path="models")
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info("Early stopping after %d bad epochs.", bad_epochs)
                    break

        # Final evaluation using best checkpoint
        ckpt = torch.load(best_path, map_location=device)
        model = build_model(ckpt["model_cfg"], input_dim=len(spec.feature_cols)).to(device)
        model.load_state_dict(ckpt["state_dict"])

        test_mse, test_mae, per_user = evaluate(model, test_loader, device)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_mae", test_mae)

        per_user_path = artifacts_dir / "reports" / "global_per_user_metrics.json"
        per_user_path.parent.mkdir(parents=True, exist_ok=True)
        per_user_path.write_text(json.dumps(per_user, indent=2, sort_keys=True), encoding="utf-8")
        mlflow.log_artifact(per_user_path.as_posix(), artifact_path="reports")

        logger.info("test_mse=%.4f test_mae=%.4f | model=%s", test_mse, test_mae, best_path.as_posix())


if __name__ == "__main__":
    main()

