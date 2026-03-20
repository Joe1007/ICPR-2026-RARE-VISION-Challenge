#!/usr/bin/env python3
"""
從預提取特徵訓練分類器
======================
此腳本載入預提取的特徵，快速訓練分類器並評估。
支援多種 loss function 和時序建模選項。

使用方式：
    python train_from_features.py --features-dir features --epochs 50

特點：
    - 訓練速度極快（幾分鐘到幾小時）
    - 支援多種 loss function：BCE, Focal Loss, Asymmetric Loss
    - 支援時序建模：可選 LSTM 層
    - 支援 temporal smoothing 後處理
    - 完整的 Temporal mAP 評估
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


USED_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]


def set_seed(seed: int) -> None:
    """設定所有隨機種子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class FeatureDataset(Dataset):
    """從預提取特徵載入的 Dataset"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], idx


# ============================================================
# Loss Functions
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    對難分類的樣本給予更高權重，有助於處理類別不平衡。
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # 計算 focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # 計算 BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        # 應用 focal weight 和 alpha
        loss = self.alpha * focal_weight * bce
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification
    
    對正負樣本使用不同的 gamma，特別適合極端類別不平衡。
    
    Reference: https://arxiv.org/abs/2009.14119
    """
    
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Asymmetric Clipping
        probs_neg = (probs + self.clip).clamp(max=1)
        
        # 計算 loss
        loss_pos = targets * torch.log(probs.clamp(min=1e-8)) * ((1 - probs) ** self.gamma_pos)
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8)) * (probs_neg ** self.gamma_neg)
        
        loss = -loss_pos - loss_neg
        return loss.mean()


# ============================================================
# Models
# ============================================================

class LinearClassifier(nn.Module):
    """簡單的線性分類器"""
    
    def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(input_dim, num_labels)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)


class MLPClassifier(nn.Module):
    """多層感知機分類器"""
    
    def __init__(self, input_dim: int, num_labels: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# Temporal mAP Evaluation (from official scoring.py)
# ============================================================

def tiou(a: dict, b: dict) -> float:
    """計算 Temporal IoU"""
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]) + 1)
    union = (a["end"] - a["start"] + 1) + (b["end"] - b["start"] + 1) - inter
    return inter / union if union > 0 else 0.0


def df_to_events(df: pd.DataFrame, video_id: str, label_columns: list, index_col: str = "frame"):
    """將 frame-level 預測轉換為事件格式"""
    df = df.sort_values(index_col).reset_index(drop=True)
    df[index_col] = df[index_col].astype(int)
    
    def active_labels(row):
        return tuple(sorted([lbl for lbl in label_columns if lbl in row and row[lbl] == 1]))
    
    df["active"] = df.apply(active_labels, axis=1)
    
    events = []
    if df.empty:
        return {"video_id": video_id, "events": []}
    
    current_labels = df.loc[0, "active"]
    start_idx = int(df.loc[0, index_col])
    
    for i in range(1, len(df)):
        idx = int(df.loc[i, index_col])
        labels = df.loc[i, "active"]
        if labels != current_labels:
            events.append({"start": start_idx, "end": idx - 1, "label": list(current_labels)})
            start_idx = idx
            current_labels = labels
    
    last_idx = int(df.loc[len(df) - 1, index_col])
    events.append({"start": start_idx, "end": last_idx, "label": list(current_labels)})
    
    return {"video_id": video_id, "events": events}


def build_events_json(df: pd.DataFrame, video_col: str = "recording", frame_col: str = "frame"):
    """建立事件 JSON"""
    videos = []
    for vid, group in df.groupby(video_col):
        videos.append(df_to_events(group, video_id=str(vid), label_columns=USED_LABELS, index_col=frame_col))
    return {"videos": videos}


def extract_by_video_label(data: dict) -> dict:
    """從事件 JSON 提取按影片和標籤組織的事件"""
    out = {}
    for v in data["videos"]:
        vid = v["video_id"]
        out.setdefault(vid, {})
        for e in v["events"]:
            for lbl in e["label"]:
                out[vid].setdefault(lbl, []).append({"start": e["start"], "end": e["end"]})
    return out


def average_precision(gt_segs: list, pr_segs: list, thr: float) -> float:
    """計算單一類別的 Average Precision"""
    matched = set()
    tp = []
    
    for p in pr_segs:
        hit = False
        for i, g in enumerate(gt_segs):
            if i in matched:
                continue
            if tiou(p, g) >= thr:
                matched.add(i)
                hit = True
                break
        tp.append(1 if hit else 0)
    
    if not gt_segs:
        return 0.0 if pr_segs else 1.0
    
    cum_tp = 0
    precisions = []
    recalls = []
    for i, v in enumerate(tp):
        cum_tp += v
        precisions.append(cum_tp / (i + 1))
        recalls.append(cum_tp / len(gt_segs))
    
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_r)
        prev_r = r
    
    return ap


def compute_map(gt: dict, pr: dict, thr: float) -> float:
    """計算 Temporal mAP"""
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pr)
    
    video_maps = []
    for vid in gt_ev:
        aps = []
        for lbl in USED_LABELS:
            gt_segs = gt_ev[vid].get(lbl, [])
            pr_segs = pr_ev.get(vid, {}).get(lbl, [])
            aps.append(average_precision(gt_segs, pr_segs, thr))
        video_maps.append(sum(aps) / len(aps))
    
    return sum(video_maps) / len(video_maps) if video_maps else 0.0


# ============================================================
# Post-processing
# ============================================================

def temporal_smooth(predictions: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    對預測結果進行時序平滑
    
    使用滑動窗口平均，減少事件邊界的抖動。
    需要按影片分組處理。
    """
    from scipy.ndimage import uniform_filter1d
    
    smoothed = np.zeros_like(predictions)
    for i in range(predictions.shape[1]):
        smoothed[:, i] = uniform_filter1d(predictions[:, i], size=window_size, mode='nearest')
    
    return smoothed


def temporal_smooth_by_video(
    predictions: np.ndarray, 
    meta_df: pd.DataFrame, 
    window_size: int = 5
) -> np.ndarray:
    """
    按影片分組進行時序平滑
    
    確保平滑不會跨越不同影片的邊界。
    """
    from scipy.ndimage import uniform_filter1d
    
    smoothed = np.zeros_like(predictions)
    
    for recording, group in meta_df.groupby("recording"):
        indices = group.index.tolist()
        for i in range(predictions.shape[1]):
            video_preds = predictions[indices, i]
            video_smoothed = uniform_filter1d(video_preds, size=window_size, mode='nearest')
            smoothed[indices, i] = video_smoothed
    
    return smoothed


# ============================================================
# Training Utilities
# ============================================================

def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """計算正樣本權重"""
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    pos_weight = neg / np.maximum(pos, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """計算類別權重（用於 sampling）"""
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    return neg / np.maximum(pos, 1.0)


def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    """計算樣本權重（用於 WeightedRandomSampler）"""
    class_weights = compute_class_weights(labels)
    
    # 每個樣本的權重 = 其正標籤權重的總和（或平均）
    sample_weights = np.zeros(len(labels))
    for i in range(len(labels)):
        positive_labels = np.where(labels[i] == 1)[0]
        if len(positive_labels) > 0:
            sample_weights[i] = class_weights[positive_labels].mean()
        else:
            sample_weights[i] = 1.0
    
    return sample_weights


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    """計算分類指標"""
    metrics = {}
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["micro_precision"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["micro_recall"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    per_class_ap = {}
    for i, lbl in enumerate(USED_LABELS):
        try:
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            ap = 0.0
        per_class_ap[lbl] = float(ap)
    
    metrics["per_class_ap"] = per_class_ap
    metrics["mAP_frame"] = float(np.mean(list(per_class_ap.values())))
    
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="", help="Path to args.json")
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train classifier from pre-extracted features"
    )
    parser.add_argument("--config", default="", help="Path to args.json")
    
    # 資料參數
    parser.add_argument("--features-dir", default="features", help="特徵目錄")
    
    # 模型參數
    parser.add_argument("--model", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP 隱藏層維度")
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # 訓練參數
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Loss function
    parser.add_argument("--loss", choices=["bce", "focal", "asymmetric"], default="bce")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-gamma-pos", type=float, default=1.0)
    
    # Sampling
    parser.add_argument("--weighted-sampling", action="store_true", help="使用加權採樣")
    
    # Post-processing
    parser.add_argument("--smooth-window", type=int, default=0, help="時序平滑窗口大小（0=不平滑）")
    
    # 閾值
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--find-best-threshold", action="store_true", help="搜尋最佳閾值")
    
    # 輸出
    parser.add_argument("--out-pred-json", default="val_pred.json")
    parser.add_argument("--out-gt-json", default="val_gt.json")
    parser.add_argument("--metrics-json", default="val_metrics.json")
    parser.add_argument("--save-model", default="", help="儲存模型路徑")
    
    default_args_path = Path(__file__).with_suffix(".args.json")
    config_path = config_args.config or (str(default_args_path) if default_args_path.exists() else "")
    if config_path:
        cfg_path = Path(config_path)
        if cfg_path.exists():
            with cfg_path.open("r") as f:
                defaults = json.load(f)
            if isinstance(defaults, dict):
                parser.set_defaults(**defaults)

    args = parser.parse_args()
    
    # 設定隨機種子
    set_seed(args.seed)
    
    # ===== 載入資料 =====
    print("=" * 60)
    print("Loading features...")
    print("=" * 60)
    
    features_dir = Path(args.features_dir)
    
    train_features = np.load(features_dir / "train_features.npy")
    val_features = np.load(features_dir / "val_features.npy")
    train_labels = np.load(features_dir / "train_labels.npy")
    val_labels = np.load(features_dir / "val_labels.npy")
    
    train_meta = pd.read_csv(features_dir / "train_meta.csv")
    val_meta = pd.read_csv(features_dir / "val_meta.csv")
    
    with open(features_dir / "config.json", "r") as f:
        config = json.load(f)
    
    frame_col = config.get("frame_col", "frame")
    embed_dim = config.get("embed_dim", train_features.shape[1])
    
    print(f"Train: {train_features.shape}, Val: {val_features.shape}")
    print(f"Embed dim: {embed_dim}")
    
    # ===== 建立 Dataset =====
    train_ds = FeatureDataset(train_features, train_labels)
    val_ds = FeatureDataset(val_features, val_labels)
    
    # 建立 sampler
    if args.weighted_sampling:
        print("Using weighted random sampling...")
        sample_weights = compute_sample_weights(train_labels)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    # ===== 建立模型 =====
    print("\n" + "=" * 60)
    print("Building model...")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model == "linear":
        model = LinearClassifier(embed_dim, len(USED_LABELS), dropout=args.dropout)
    else:
        model = MLPClassifier(embed_dim, len(USED_LABELS), hidden_dim=args.hidden_dim, dropout=args.dropout)
    
    model = model.to(device)
    print(f"Model: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== 建立 Loss Function =====
    pos_weight = compute_pos_weight(train_labels).to(device)
    
    if args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Using BCEWithLogitsLoss with pos_weight")
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, pos_weight=pos_weight)
        print(f"Using FocalLoss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        criterion = AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos)
        print(f"Using AsymmetricLoss (gamma_neg={args.asl_gamma_neg}, gamma_pos={args.asl_gamma_pos})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ===== 訓練 =====
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for features, labels, _ in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * features.size(0)
        
        train_loss /= len(train_ds)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, _ in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * features.size(0)
        
        val_loss /= len(val_ds)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
    
    # 載入最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model (val_loss={best_val_loss:.4f})")
    
    # ===== 預測 =====
    print("\n" + "=" * 60)
    print("Predicting...")
    print("=" * 60)
    
    model.eval()
    pred_scores = np.zeros((len(val_ds), len(USED_LABELS)), dtype=np.float32)
    
    with torch.no_grad():
        for features, _, idxs in val_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            pred_scores[idxs.numpy(), :] = probs
    
    # 時序平滑
    if args.smooth_window > 0:
        print(f"Applying temporal smoothing (window={args.smooth_window})...")
        pred_scores = temporal_smooth_by_video(pred_scores, val_meta, args.smooth_window)
    
    # 搜尋最佳閾值
    threshold = args.threshold
    if args.find_best_threshold:
        print("Searching for best threshold...")
        best_f1 = 0.0
        best_thresh = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            y_pred = (pred_scores >= t).astype(np.int32)
            f1 = f1_score(val_labels, y_pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        threshold = best_thresh
        print(f"Best threshold: {threshold:.2f} (macro F1={best_f1:.4f})")
    
    # 閾值化
    pred_binary = (pred_scores >= threshold).astype(np.int32)
    
    # ===== 建立事件 JSON =====
    print("\n" + "=" * 60)
    print("Building event JSONs...")
    print("=" * 60)
    
    # 建立預測 DataFrame
    pred_df = val_meta.copy()
    for i, lbl in enumerate(USED_LABELS):
        pred_df[lbl] = pred_binary[:, i]
    
    # 建立 GT DataFrame
    gt_df = val_meta.copy()
    for i, lbl in enumerate(USED_LABELS):
        gt_df[lbl] = val_labels[:, i]
    
    # 轉換成事件 JSON
    gt_json = build_events_json(gt_df, video_col="recording", frame_col=frame_col)
    pred_json = build_events_json(pred_df, video_col="recording", frame_col=frame_col)
    
    # 儲存 JSON
    with open(args.out_gt_json, "w") as f:
        json.dump(gt_json, f, indent=2)
    with open(args.out_pred_json, "w") as f:
        json.dump(pred_json, f, indent=2)
    
    print(f"Saved {args.out_gt_json}")
    print(f"Saved {args.out_pred_json}")
    
    # ===== 評估 =====
    print("\n" + "=" * 60)
    print("Evaluating...")
    print("=" * 60)
    
    # Temporal mAP
    map_05 = compute_map(gt_json, pred_json, 0.5)
    map_095 = compute_map(gt_json, pred_json, 0.95)
    
    print(f"\n{'='*40}")
    print(f"Temporal mAP@0.5:  {map_05:.4f}")
    print(f"Temporal mAP@0.95: {map_095:.4f}")
    print(f"{'='*40}")
    
    # Frame-level metrics
    metrics = compute_classification_metrics(val_labels, pred_scores, pred_binary)
    metrics["temporal_mAP_0.5"] = map_05
    metrics["temporal_mAP_0.95"] = map_095
    metrics["threshold"] = threshold
    
    print(f"\nFrame-level metrics:")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Micro F1:        {metrics['micro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"  Frame mAP:       {metrics['mAP_frame']:.4f}")
    
    print(f"\nPer-class AP:")
    for lbl, ap in metrics["per_class_ap"].items():
        pos_count = val_labels[:, USED_LABELS.index(lbl)].sum()
        print(f"  {lbl:20s}: {ap:.4f} (n={int(pos_count):6d})")
    
    # 儲存 metrics
    with open(args.metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved {args.metrics_json}")
    
    # 儲存模型
    if args.save_model:
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_type": args.model,
            "embed_dim": embed_dim,
            "num_labels": len(USED_LABELS),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "threshold": threshold,
        }, args.save_model)
        print(f"Saved model to {args.save_model}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()