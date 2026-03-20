#!/usr/bin/env python3
"""
多模型 Ensemble 訓練腳本
======================
一次訓練多個分類器（不同 loss、model、dropout），儲存至 ensemble/ 供 predict_ensemble 使用。

使用方式：
    python train_ensemble.py --config train_ensemble.args.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from train_from_features import (
    USED_LABELS,
    LinearClassifier,
    MLPClassifier,
    FocalLoss,
    AsymmetricLoss,
    FeatureDataset,
    compute_pos_weight,
    compute_sample_weights,
    set_seed,
)

# 預設的 ensemble 配置：不同 loss、model、dropout 組合
DEFAULT_ENSEMBLE_CONFIGS = [
    {"model": "linear", "loss": "bce", "dropout": 0.3},
    {"model": "linear", "loss": "focal", "dropout": 0.3},
    {"model": "linear", "loss": "asymmetric", "dropout": 0.3},
    {"model": "mlp", "loss": "bce", "dropout": 0.3},
    {"model": "mlp", "loss": "focal", "dropout": 0.3},
]


def train_single_model(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: np.ndarray,
    embed_dim: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    focal_alpha: float,
    focal_gamma: float,
    asl_gamma_neg: float,
    asl_gamma_pos: float,
) -> tuple:
    """訓練單一模型，回傳 (model, best_val_loss, threshold)"""
    model_type = config["model"]
    loss_type = config["loss"]
    dropout = config["dropout"]
    hidden_dim = config.get("hidden_dim", 256)

    if model_type == "linear":
        model = LinearClassifier(embed_dim, len(USED_LABELS), dropout=dropout)
    else:
        model = MLPClassifier(embed_dim, len(USED_LABELS), hidden_dim=hidden_dim, dropout=dropout)
    model = model.to(device)

    pos_weight = compute_pos_weight(train_labels).to(device)
    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
    else:
        criterion = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for features, labels, _ in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, _ in val_loader:
                features, labels = features.to(device), labels.to(device)
                val_loss += criterion(model(features), labels).item() * features.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # 簡單用 0.5 當閾值（ensemble 時可再搜尋）
    return model, best_val_loss, 0.5


def main():
    parser = argparse.ArgumentParser(description="Train ensemble of classifiers")
    parser.add_argument("--config", default="")
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--ensemble-dir", default="ensemble", help="儲存各模型的目錄")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=1.0)
    parser.add_argument("--weighted-sampling", action="store_true")
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help='JSON 路徑或 inline JSON，例如 [{"model":"linear","loss":"bce","dropout":0.3}]',
    )

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    config_args, _ = config_parser.parse_known_args()

    default_path = Path(__file__).with_suffix(".args.json")
    config_path = config_args.config or (str(default_path) if default_path.exists() else "")
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            defaults = json.load(f)
        if isinstance(defaults, dict):
            parser.set_defaults(**defaults)

    args = parser.parse_args()

    set_seed(args.seed)

    features_dir = Path(args.features_dir)
    ensemble_dir = Path(args.ensemble_dir)
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading features...")
    print("=" * 60)
    train_features = np.load(features_dir / "train_features.npy")
    val_features = np.load(features_dir / "val_features.npy")
    train_labels = np.load(features_dir / "train_labels.npy")
    val_labels = np.load(features_dir / "val_labels.npy")
    with open(features_dir / "config.json", "r") as f:
        feat_cfg = json.load(f)
    embed_dim = feat_cfg.get("embed_dim", train_features.shape[1])
    print(f"Train: {train_features.shape}, Val: {val_features.shape}, embed_dim: {embed_dim}")

    train_ds = FeatureDataset(train_features, train_labels)
    val_ds = FeatureDataset(val_features, val_labels)

    if args.weighted_sampling:
        sample_weights = compute_sample_weights(train_labels)
        sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 解析 ensemble configs
    if args.configs:
        cfg_path = Path(args.configs)
        if cfg_path.exists():
            with open(cfg_path, "r") as f:
                configs = json.load(f)
        else:
            configs = json.loads(args.configs)
    else:
        configs = DEFAULT_ENSEMBLE_CONFIGS

    print(f"\nEnsemble configs: {len(configs)} models")
    for i, c in enumerate(configs):
        print(f"  [{i}] model={c.get('model','linear')}, loss={c.get('loss','bce')}, dropout={c.get('dropout',0.3)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = []

    for i, cfg in enumerate(configs):
        print("\n" + "=" * 60)
        print(f"Training model {i + 1}/{len(configs)}: {cfg}")
        print("=" * 60)
        model, best_val_loss, thresh = train_single_model(
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            train_labels=train_labels,
            embed_dim=embed_dim,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            asl_gamma_neg=args.asl_gamma_neg,
            asl_gamma_pos=args.asl_gamma_pos,
        )
        save_path = ensemble_dir / f"model_{i}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": cfg.get("model", "linear"),
            "embed_dim": embed_dim,
            "num_labels": len(USED_LABELS),
            "hidden_dim": cfg.get("hidden_dim", 256),
            "dropout": cfg.get("dropout", 0.3),
            "threshold": thresh,
            "config": cfg,
        }, save_path)
        manifest.append({
            "path": str(save_path),
            "config": cfg,
            "best_val_loss": float(best_val_loss),
        })
        print(f"Saved {save_path} (val_loss={best_val_loss:.4f})")

    with open(ensemble_dir / "manifest.json", "w") as f:
        json.dump({"models": manifest, "embed_dim": embed_dim}, f, indent=2)
    print(f"\nSaved {ensemble_dir / 'manifest.json'}")
    print("Done. Run: python predict_ensemble.py --config predict_ensemble.args.json")


if __name__ == "__main__":
    main()
