#!/usr/bin/env python3
"""
EndoFM-LV 特徵預提取腳本
========================
此腳本使用預訓練的 EndoFM-LV backbone 提取所有幀的特徵，
儲存為 .npy 檔案供後續快速訓練使用。

使用方式：
    python extract_features.py --config extract_features.args.json

輸出檔案：
    features/
    ├── train_features.npy   (N_train, 768)
    ├── val_features.npy     (N_val, 768)
    ├── train_labels.npy     (N_train, 17)
    ├── val_labels.npy       (N_val, 17)
    ├── train_meta.csv       (recording, frame, image_path)
    ├── val_meta.csv         (recording, frame, image_path)
    └── config.json          (提取時的設定，確保一致性)
"""

import argparse
import bisect
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


USED_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]


def set_seed(seed: int) -> None:
    """設定所有隨機種子以確保可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GalarClipDataset(Dataset):
    """
    Galar 資料集的 Clip 載入器
    
    對每一幀建立一個樣本，並根據 num_frames 和 frame_stride 
    載入周圍的幀組成一個 clip。
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: Optional[str],
        frame_template: Optional[str],
        transform,
        num_frames: int,
        frame_stride: int,
        clip_mode: str,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root) if image_root else None
        self.frame_template = frame_template
        self.transform = transform
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.clip_mode = clip_mode

        # 確保所有標籤欄位存在
        for lbl in USED_LABELS:
            if lbl not in self.df.columns:
                self.df[lbl] = 0

        if "recording" not in self.df.columns:
            raise ValueError("Missing 'recording' column in dataframe.")
        
        self.frame_col = "frame" if "frame" in self.df.columns else "index"
        if self.frame_col not in self.df.columns:
            raise ValueError("Missing frame index column ('frame' or 'index').")

        # 建立圖片路徑映射
        self.image_paths = None
        if "image_path" in self.df.columns:
            self.image_paths = self.df["image_path"].astype(str).tolist()

        self._path_map = {}
        self._indices_by_recording = {}
        for i, row in self.df.iterrows():
            recording = str(row["recording"])
            frame_idx = int(row[self.frame_col])
            if self.image_paths is not None:
                self._path_map[(recording, frame_idx)] = self._resolve_path(row, i)
            self._indices_by_recording.setdefault(recording, []).append(frame_idx)
        
        for recording in self._indices_by_recording:
            self._indices_by_recording[recording].sort()

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row, idx):
        """解析圖片路徑"""
        if self.image_paths is not None:
            rel = self.image_paths[idx]
            path = Path(rel)
            if path.is_absolute():
                return path
            if self.image_root:
                root = self.image_root
                rel_parts = list(path.parts)
                root_parts = list(root.parts)
                if root_parts and rel_parts[: len(root_parts)] == root_parts:
                    return path
                return root / path
            return path

        if self.frame_template:
            data = {
                "root": str(self.image_root) if self.image_root else "",
                "recording": row.get("recording", ""),
                "index": int(row[self.frame_col]),
                "index_zfill": f"{int(row[self.frame_col]):06d}",
            }
            return Path(self.frame_template.format(**data))

        raise ValueError("No image path available. Provide image_path column or frame_template.")

    def _path_for_index(self, recording: str, frame_idx: int) -> Path:
        """根據 recording 和 frame_idx 取得圖片路徑，若不存在則找最近的"""
        key = (recording, frame_idx)
        if key in self._path_map:
            return self._path_map[key]
        
        indices = self._indices_by_recording.get(recording, [])
        if not indices:
            raise FileNotFoundError(f"No frames found for recording {recording}")
        
        pos = bisect.bisect_left(indices, frame_idx)
        if pos == 0:
            nearest = indices[0]
        elif pos >= len(indices):
            nearest = indices[-1]
        else:
            before = indices[pos - 1]
            after = indices[pos]
            nearest = before if abs(frame_idx - before) <= abs(after - frame_idx) else after
        
        if (recording, nearest) in self._path_map:
            return self._path_map[(recording, nearest)]
        
        if self.frame_template:
            data = {
                "root": str(self.image_root) if self.image_root else "",
                "recording": recording,
                "index": int(nearest),
                "index_zfill": f"{int(nearest):06d}",
            }
            return Path(self.frame_template.format(**data))
        
        if self.image_root:
            return self.image_root / recording / f"frame_{int(nearest):06d}.PNG"
        
        raise FileNotFoundError(f"Cannot resolve image path for {recording}:{nearest}")

    def _build_indices(self, center_idx: int) -> list[int]:
        """根據中心幀建立 clip 的幀索引列表"""
        if self.num_frames <= 1:
            return [center_idx]
        
        stride = max(1, self.frame_stride)
        if self.clip_mode == "forward":
            return [center_idx + i * stride for i in range(self.num_frames)]
        
        # center mode
        start = -self.num_frames // 2
        offsets = [i * stride for i in range(start, start + self.num_frames)]
        return [center_idx + off for off in offsets]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        recording = str(row["recording"])
        center_idx = int(row[self.frame_col])
        frame_indices = self._build_indices(center_idx)
        
        frames = []
        for frame_idx in frame_indices:
            img_path = self._path_for_index(recording, frame_idx)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)
        
        clip = torch.stack(frames, dim=1)  # (C, T, H, W)
        labels = row[USED_LABELS].to_numpy(dtype=np.float32)
        return clip, torch.from_numpy(labels), idx


def add_image_paths(df: pd.DataFrame, image_root: str) -> pd.DataFrame:
    """為 DataFrame 添加 image_path 欄位"""
    df = df.copy()
    df["recording"] = df["recording"].astype(str)
    frame_col = "frame" if "frame" in df.columns else "index"
    frame_ids = df[frame_col].astype(int).map(lambda x: f"{x:06d}")
    root = Path(image_root).resolve().as_posix()
    df["image_path"] = (
        root + "/" + df["recording"] + "/frame_" + frame_ids + ".PNG"
    )
    return df


def load_labels_from_dir(labels_dir: str) -> pd.DataFrame:
    """從標籤目錄載入所有 CSV 並合併"""
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")
    
    dfs = []
    for csv_path in sorted(labels_path.glob("*.csv")):
        df = pd.read_csv(csv_path)
        df["recording"] = csv_path.stem
        dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No CSV files found in {labels_dir}")
    
    return pd.concat(dfs, ignore_index=True)


def split_by_recording(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int):
    """按影片（recording）分割資料集，避免 data leakage"""
    recordings = sorted(df["recording"].astype(str).unique())
    rng = random.Random(seed)
    rng.shuffle(recordings)
    
    n_total = len(recordings)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    
    train_ids = set(recordings[:n_train])
    val_ids = set(recordings[n_train:n_train + n_val])
    
    train_df = df[df["recording"].astype(str).isin(train_ids)].reset_index(drop=True)
    val_df = df[df["recording"].astype(str).isin(val_ids)].reset_index(drop=True)
    
    print(f"Split: {len(train_ids)} train videos, {len(val_ids)} val videos")
    print(f"       {len(train_df)} train frames, {len(val_df)} val frames")
    
    return train_df, val_df


def build_endofm_backbone(num_frames: int, image_size: int):
    """建立 EndoFM-LV backbone 模型"""
    endofm_root = Path(__file__).resolve().parent / "EndoFM-LV"
    sys.path.insert(0, str(endofm_root))
    
    from models import get_vit_base_patch16_224
    from utils.parser import load_config

    cfg_args = SimpleNamespace(
        cfg_file=str(endofm_root / "models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"),
        opts=None,
        num_shards=1,
        shard_id=0,
    )
    cfg = load_config(cfg_args)
    cfg.DATA.NUM_FRAMES = num_frames
    cfg.DATA.TRAIN_CROP_SIZE = image_size
    cfg.MODEL.NUM_CLASSES = 0
    cfg.TIMESFORMER.PRETRAINED_MODEL = ""
    
    return get_vit_base_patch16_224(cfg=cfg, no_head=True)


def load_endofm_weights(model: nn.Module, ckpt_path: str, checkpoint_key: Optional[str] = "teacher") -> None:
    """載入 EndoFM-LV 預訓練權重"""
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    if checkpoint_key and isinstance(ckpt, dict) and checkpoint_key in ckpt:
        ckpt = ckpt[checkpoint_key]
    
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    
    if isinstance(ckpt, dict) and any(k.startswith("backbone.") for k in ckpt.keys()):
        ckpt = {k[len("backbone."):]: v for k, v in ckpt.items() if k.startswith("backbone.")}
    
    # 處理 time_embed 尺寸不匹配
    if isinstance(ckpt, dict) and "time_embed" in ckpt:
        target = model.time_embed
        source = ckpt["time_embed"]
        if source.shape != target.shape:
            print(f"Resizing time_embed from {source.shape} to {target.shape}")
            resized = F.interpolate(
                source.transpose(1, 2),
                size=target.shape[1],
                mode="nearest",
            ).transpose(1, 2)
            ckpt["time_embed"] = resized
    
    # 處理 pos_embed 尺寸不匹配
    if isinstance(ckpt, dict) and "pos_embed" in ckpt:
        target = model.pos_embed
        source = ckpt["pos_embed"]
        if source.shape != target.shape:
            print(f"Resizing pos_embed from {source.shape} to {target.shape}")
            cls_pos = source[:, :1, :]
            patch_pos = source[:, 1:, :]
            num_patches = patch_pos.shape[1]
            grid_size = int(num_patches ** 0.5)
            target_patches = target.shape[1] - 1
            target_grid = int(target_patches ** 0.5)
            if grid_size * grid_size == num_patches and target_grid * target_grid == target_patches:
                patch_pos = patch_pos.transpose(1, 2).reshape(1, patch_pos.shape[2], grid_size, grid_size)
                patch_pos = F.interpolate(patch_pos, size=(target_grid, target_grid), mode="nearest")
                patch_pos = patch_pos.flatten(2).transpose(1, 2)
                ckpt["pos_embed"] = torch.cat((cls_pos, patch_pos), dim=1)
    
    msg = model.load_state_dict(ckpt, strict=False)
    print(f"Loaded EndoFM-LV weights: {msg}")


@torch.no_grad()
def extract_backbone_features(
    backbone: nn.Module, 
    loader: DataLoader, 
    device: torch.device,
    desc: str = "Extracting features"
) -> np.ndarray:
    """
    使用 backbone 提取所有樣本的特徵
    
    Args:
        backbone: EndoFM-LV backbone 模型
        loader: DataLoader（必須 shuffle=False）
        device: 計算設備
        desc: 進度條描述
    
    Returns:
        features: (N, embed_dim) 的特徵矩陣
    """
    backbone.eval()
    embed_dim = backbone.embed_dim
    n_samples = len(loader.dataset)
    
    features = np.zeros((n_samples, embed_dim), dtype=np.float32)
    
    for clips, _, idxs in tqdm(loader, desc=desc):
        clips = clips.to(device)
        cls_token, _ = backbone(clips)
        features[idxs.numpy(), :] = cls_token.cpu().numpy()
    
    return features


def save_metadata(df: pd.DataFrame, save_path: Path, frame_col: str):
    """
    儲存 metadata，用於後續事件轉換
    
    包含：recording, frame, image_path（如果有的話）
    """
    cols_to_save = ["recording", frame_col]
    if "image_path" in df.columns:
        cols_to_save.append("image_path")
    
    meta_df = df[cols_to_save].copy()
    meta_df.to_csv(save_path, index=False)
    print(f"Saved metadata to {save_path}")


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="", help="Path to args.json")
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="EndoFM-LV Feature Extraction for RARE-VISION-2026"
    )
    parser.add_argument("--config", default="", help="Path to args.json")
    
    # 資料相關參數
    parser.add_argument("--train-csv", default="", help="預先分割的訓練集 CSV")
    parser.add_argument("--val-csv", default="", help="預先分割的驗證集 CSV")
    parser.add_argument("--labels-dir", default="./galar_dataset/downloads/Labels")
    parser.add_argument("--image-root", default="./galar_dataset/downloads")
    parser.add_argument("--frame-template", default=None)
    
    # 分割參數
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    
    # 模型參數
    parser.add_argument("--num-frames", type=int, default=4, help="每個 clip 的幀數")
    parser.add_argument("--frame-stride", type=int, default=2, help="幀之間的間隔")
    parser.add_argument("--clip-mode", choices=["center", "forward"], default="center")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pretrained-weights", default="", help="EndoFM-LV checkpoint 路徑")
    parser.add_argument("--checkpoint-key", default="teacher")
    
    # 執行參數
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-skip-missing-images", action="store_true", default=False)
    
    # 輸出參數
    parser.add_argument("--features-dir", default="features", help="特徵儲存目錄")
    
    # 從 JSON 檔案載入預設值
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
    
    if not args.pretrained_weights:
        parser.error("--pretrained-weights is required (set in args.json or CLI)")

    # 設定隨機種子
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    
    # 建立輸出目錄
    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 載入資料 =====
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    train_csv = Path(args.train_csv) if args.train_csv else None
    val_csv = Path(args.val_csv) if args.val_csv else None
    
    if train_csv and val_csv and train_csv.exists() and val_csv.exists():
        print(f"Loading from existing CSVs: {train_csv}, {val_csv}")
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    else:
        print(f"Loading labels from {args.labels_dir}")
        full_df = load_labels_from_dir(args.labels_dir)
        train_df, val_df = split_by_recording(
            full_df, 
            train_ratio=args.train_ratio, 
            val_ratio=args.val_ratio, 
            seed=args.seed
        )
    
    # 確認必要欄位
    if "recording" not in train_df.columns or "recording" not in val_df.columns:
        raise ValueError("Missing 'recording' column.")
    
    # 添加圖片路徑
    if "image_path" not in train_df.columns:
        train_df = add_image_paths(train_df, args.image_root)
    if "image_path" not in val_df.columns:
        val_df = add_image_paths(val_df, args.image_root)
    
    # 過濾不存在的圖片
    if not args.no_skip_missing_images:
        print("Filtering missing images...")
        train_before = len(train_df)
        val_before = len(val_df)
        train_df = train_df[train_df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        val_df = val_df[val_df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        print(f"Train: {train_before} -> {len(train_df)} (removed {train_before - len(train_df)})")
        print(f"Val: {val_before} -> {len(val_df)} (removed {val_before - len(val_df)})")
    
    frame_col = "frame" if "frame" in train_df.columns else "index"
    print(f"Using frame column: {frame_col}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # ===== 建立 Dataset 和 DataLoader =====
    print("\n" + "=" * 60)
    print("Creating datasets...")
    print("=" * 60)
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_ds = GalarClipDataset(
        train_df, args.image_root, args.frame_template, transform,
        num_frames=args.num_frames, frame_stride=args.frame_stride, clip_mode=args.clip_mode,
    )
    val_ds = GalarClipDataset(
        val_df, args.image_root, args.frame_template, transform,
        num_frames=args.num_frames, frame_stride=args.frame_stride, clip_mode=args.clip_mode,
    )
    
    # 特徵提取必須 shuffle=False 以保持順序
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    # ===== 載入模型 =====
    print("\n" + "=" * 60)
    print("Loading EndoFM-LV backbone...")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    backbone = build_endofm_backbone(num_frames=args.num_frames, image_size=args.image_size)
    load_endofm_weights(backbone, args.pretrained_weights, args.checkpoint_key)
    backbone = backbone.to(device)
    backbone.eval()
    
    print(f"Backbone embed_dim: {backbone.embed_dim}")
    
    # ===== 提取特徵 =====
    print("\n" + "=" * 60)
    print("Extracting features...")
    print("=" * 60)
    
    train_features = extract_backbone_features(
        backbone, train_loader, device, desc="Extracting train features"
    )
    val_features = extract_backbone_features(
        backbone, val_loader, device, desc="Extracting val features"
    )
    
    print(f"Train features shape: {train_features.shape}")
    print(f"Val features shape: {val_features.shape}")
    
    # ===== 儲存結果 =====
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)
    
    # 儲存特徵
    np.save(features_dir / "train_features.npy", train_features)
    np.save(features_dir / "val_features.npy", val_features)
    print(f"Saved train_features.npy: {train_features.shape}")
    print(f"Saved val_features.npy: {val_features.shape}")
    
    # 儲存標籤
    train_labels = train_df[USED_LABELS].to_numpy(dtype=np.int32)
    val_labels = val_df[USED_LABELS].to_numpy(dtype=np.int32)
    np.save(features_dir / "train_labels.npy", train_labels)
    np.save(features_dir / "val_labels.npy", val_labels)
    print(f"Saved train_labels.npy: {train_labels.shape}")
    print(f"Saved val_labels.npy: {val_labels.shape}")
    
    # 儲存 metadata（用於事件轉換）
    save_metadata(train_df, features_dir / "train_meta.csv", frame_col)
    save_metadata(val_df, features_dir / "val_meta.csv", frame_col)
    
    # 儲存設定（確保一致性）
    config = {
        "num_frames": args.num_frames,
        "frame_stride": args.frame_stride,
        "clip_mode": args.clip_mode,
        "image_size": args.image_size,
        "pretrained_weights": args.pretrained_weights,
        "checkpoint_key": args.checkpoint_key,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "embed_dim": backbone.embed_dim,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "used_labels": USED_LABELS,
        "frame_col": frame_col,
    }
    with open(features_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json")
    
    # 儲存類別統計（用於計算 pos_weight）
    label_stats = {
        "train": {
            lbl: int(train_labels[:, i].sum()) 
            for i, lbl in enumerate(USED_LABELS)
        },
        "val": {
            lbl: int(val_labels[:, i].sum()) 
            for i, lbl in enumerate(USED_LABELS)
        },
        "n_train": len(train_df),
        "n_val": len(val_df),
    }
    with open(features_dir / "label_stats.json", "w") as f:
        json.dump(label_stats, f, indent=2)
    print(f"Saved label_stats.json")
    
    # 計算儲存空間
    total_size = (
        train_features.nbytes + val_features.nbytes + 
        train_labels.nbytes + val_labels.nbytes
    )
    print(f"\nTotal feature storage: {total_size / 1e9:.2f} GB")
    
    print("\n" + "=" * 60)
    print("Feature extraction complete!")
    print(f"Results saved to: {features_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()