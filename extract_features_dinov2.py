#!/usr/bin/env python3
"""
多 Backbone 特徵提取腳本
========================
從額外的預訓練 backbone 提取 per-frame 特徵。
使用現有的 train_meta.csv / val_meta.csv 確保與 EndoFM-LV 相同的 train/val split。

支援的 backbone：
  - dinov3_vitl16：DINOv3 ViT-L/16（1024 維，最新最強，推薦首選）
  - dinov3_vitb16：DINOv3 ViT-B/16（768 維）
  - dinov2_vitl14：DINOv2 ViT-L/14（1024 維）
  - dinov2_vitb14：DINOv2 ViT-B/14（768 維，速度較快）
  - clip_vitl14：CLIP ViT-L/14（768 維）
  - biomedclip：BiomedCLIP（512 維，醫學影像專用）
  - resnet50：ImageNet-pretrained ResNet-50（2048 維，GAP）

使用方式：
    # DINOv3 ViT-L（推薦，最強）
    python extract_features_dinov2.py --backbone dinov3_vitl16 --output-dir features_dinov3

    # DINOv2 ViT-L（備選）
    python extract_features_dinov2.py --backbone dinov2_vitl14 --output-dir features_dinov2

使用既有 split：
    腳本會讀取 --endofm-features-dir (預設 features/) 中的 train_meta.csv 和 val_meta.csv，
    確保 train/val split 與 EndoFM-LV 完全一致。

輸出結構（與 EndoFM-LV features/ 相同）：
    features_dinov2/
    ├── train_features.npy
    ├── val_features.npy
    ├── train_labels.npy
    ├── val_labels.npy
    ├── train_meta.csv
    ├── val_meta.csv
    └── config.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


USED_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]


class SingleFrameDataset(Dataset):
    """
    單幀 Dataset，載入 meta CSV 裡指定的圖片。
    不做 clip（DINOv2/CLIP 都是單幀模型）。
    """

    def __init__(self, meta_df: pd.DataFrame, image_root: str, transform, labels_df=None):
        self.meta = meta_df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.labels_df = labels_df

        # 建立圖片路徑
        if "image_path" in self.meta.columns:
            self.paths = self.meta["image_path"].tolist()
        else:
            frame_col = "frame" if "frame" in self.meta.columns else "index"
            self.paths = [
                str(self.image_root / str(row["recording"]) / f"frame_{int(row[frame_col]):06d}.PNG")
                for _, row in self.meta.iterrows()
            ]

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx


class TestFrameDataset(Dataset):
    """測試集單幀 Dataset"""

    def __init__(self, meta_df: pd.DataFrame, transform):
        self.meta = meta_df.reset_index(drop=True)
        self.transform = transform
        self.paths = self.meta["image_path"].tolist()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx


def build_backbone(backbone_name: str, device: torch.device):
    """
    建立 backbone 模型。
    回傳 (model, transform, embed_dim)
    """

    if backbone_name == "dinov2_vitl14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        model = model.to(device).eval()
        transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        embed_dim = 1024
        return model, transform, embed_dim

    elif backbone_name == "dinov2_vitb14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model = model.to(device).eval()
        transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        embed_dim = 768
        return model, transform, embed_dim

    elif backbone_name == "dinov2_vitg14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14")
        model = model.to(device).eval()
        transform = transforms.Compose([
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(518),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        embed_dim = 1536
        return model, transform, embed_dim

    elif backbone_name == "dinov3_vitl16":
        # DINOv3 ViT-L/16 via timm (v1.0.20+)
        # 1024 維，比 DINOv2 強 ~6 mIoU on ADE20K
        timm_names = [
            "vit_large_patch16_dinov3.lvd1689m",  # 較新 timm
            "vit_large_patch16_dinov3.sat493m",
            "vit_large_patch16_224.dinov3",  # 舊版 timm
        ]
        for timm_name in timm_names:
            try:
                import timm
                model = timm.create_model(timm_name, pretrained=True, num_classes=0)
                model = model.to(device).eval()
                data_cfg = timm.data.resolve_model_data_config(model)
                transform = timm.data.create_transform(**data_cfg, is_training=False)
                embed_dim = model.num_features  # 1024
                print(f"  Loaded DINOv3 ViT-L/16 via timm ({timm_name}), embed_dim={embed_dim}")
                return model, transform, embed_dim
            except Exception:
                continue
        # timm 全部失敗，嘗試 HuggingFace（需 transformers 較新 + PyTorch 2.1+）
        try:
            from transformers import AutoModel
            try:
                from transformers import AutoImageProcessor
                processor = AutoImageProcessor.from_pretrained("facebook/dinov3-large")
            except ImportError:
                from transformers import AutoFeatureExtractor
                processor = AutoFeatureExtractor.from_pretrained("facebook/dinov3-large")
            hf_model = AutoModel.from_pretrained("facebook/dinov3-large")
            hf_model = hf_model.to(device).eval()

            class HFDINOv3Wrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    outputs = self.model(x)
                    return outputs.last_hidden_state[:, 0, :]  # CLS token

            transform = transforms.Compose([
                transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            wrapper = HFDINOv3Wrapper(hf_model).to(device).eval()
            embed_dim = hf_model.config.hidden_size
            print(f"  Loaded DINOv3 ViT-L via HuggingFace, embed_dim={embed_dim}")
            return wrapper, transform, embed_dim
        except Exception as e:
            raise RuntimeError(
                f"DINOv3 載入失敗。請嘗試：\n"
                f"  1. 升級 timm: pip install -U timm  # 需 v1.0.20+ 支援 DINOv3\n"
                f"  2. 或改用 DINOv2: --backbone dinov2_vitl14  # 使用 torch.hub，無需 timm/transformers\n"
                f"  HuggingFace 錯誤: {e}"
            )

    elif backbone_name == "dinov3_vitb16":
        try:
            import timm
            model = timm.create_model("vit_base_patch16_224.dinov3", pretrained=True, num_classes=0)
            model = model.to(device).eval()
            data_cfg = timm.data.resolve_model_data_config(model)
            transform = timm.data.create_transform(**data_cfg, is_training=False)
            embed_dim = model.num_features  # 768
            print(f"  Loaded DINOv3 ViT-B/16 via timm, embed_dim={embed_dim}")
            return model, transform, embed_dim
        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv3 ViT-B: {e}. Install: pip install timm>=1.0.20")

    elif backbone_name == "clip_vitl14":
        try:
            import clip
            model, preprocess = clip.load("ViT-L/14", device=device)
            model = model.eval()

            # CLIP 的 visual encoder
            class CLIPVisualWrapper(torch.nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.visual = clip_model.visual

                def forward(self, x):
                    return self.visual(x.half()).float()

            wrapper = CLIPVisualWrapper(model).to(device).eval()
            embed_dim = 768
            return wrapper, preprocess, embed_dim
        except ImportError:
            raise ImportError("Install CLIP: pip install git+https://github.com/openai/CLIP.git")

    elif backbone_name == "biomedclip":
        try:
            from open_clip import create_model_from_pretrained, get_tokenizer

            model, preprocess = create_model_from_pretrained(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            )

            class BiomedCLIPVisualWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.visual = model.visual

                def forward(self, x):
                    return self.visual(x)

            wrapper = BiomedCLIPVisualWrapper(model).to(device).eval()
            embed_dim = 512
            return wrapper, preprocess, embed_dim
        except ImportError:
            raise ImportError("Install open_clip: pip install open_clip_torch")

    elif backbone_name == "resnet50":
        from torchvision.models import resnet50
        try:
            backbone = resnet50(weights="IMAGENET1K_V1")
        except TypeError:
            backbone = resnet50(pretrained=True)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

        class ResNet50Wrapper(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            def forward(self, x):
                out = self.backbone(x)  # (B, 2048, 1, 1)
                return out.flatten(1)   # (B, 2048)

        model = ResNet50Wrapper(backbone).to(device).eval()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        embed_dim = 2048
        return model, transform, embed_dim

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}. "
                         f"Supported: dinov3_vitl16, dinov3_vitb16, dinov2_vitl14, dinov2_vitb14, dinov2_vitg14, clip_vitl14, biomedclip, resnet50")


@torch.no_grad()
def extract_features(model, loader, device, embed_dim, desc="Extracting"):
    """提取所有樣本的 CLS token 特徵"""
    n = len(loader.dataset)
    features = np.zeros((n, embed_dim), dtype=np.float32)

    for images, idxs in tqdm(loader, desc=desc):
        images = images.to(device)
        out = model(images)

        # 處理不同模型的輸出格式
        if isinstance(out, dict):
            # Some models return dict
            if "x_norm_clstoken" in out:
                feats = out["x_norm_clstoken"]
            elif "cls_token" in out:
                feats = out["cls_token"]
            else:
                feats = list(out.values())[0]
        elif isinstance(out, tuple):
            feats = out[0]
        else:
            feats = out

        # 確保是 2D
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        if feats.dim() == 3:
            feats = feats[:, 0, :]  # CLS token

        features[idxs.numpy()] = feats.cpu().numpy()

    return features


def main():
    parser = argparse.ArgumentParser(description="Extract features from additional backbones")
    parser.add_argument("--backbone", default="dinov3_vitl16",
                        choices=["dinov2_vitl14", "dinov2_vitb14", "dinov2_vitg14",
                                 "dinov3_vitl16", "dinov3_vitb16",
                                 "clip_vitl14", "biomedclip", "resnet50"],
                        help="Backbone model to use (recommended: dinov3_vitl16)")
    parser.add_argument("--endofm-features-dir", default="features",
                        help="Existing features dir (for train_meta.csv, val_meta.csv)")
    parser.add_argument("--labels-dir", default="./galar_dataset/downloads/Labels",
                        help="Labels directory (for loading label columns)")
    parser.add_argument("--image-root", default="./galar_dataset/downloads",
                        help="Image root directory")
    parser.add_argument("--output-dir", default="features_dinov2",
                        help="Output directory for new features")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (single frame, can be larger than clip)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--test-root", default="",
                        help="If set, also extract test features")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    endofm_dir = Path(args.endofm_features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===== Load existing meta (same train/val split) =====
    print("=" * 60)
    print(f"Loading meta from {endofm_dir}...")
    print("=" * 60)

    train_meta = pd.read_csv(endofm_dir / "train_meta.csv")
    val_meta = pd.read_csv(endofm_dir / "val_meta.csv")
    train_labels = np.load(endofm_dir / "train_labels.npy")
    val_labels = np.load(endofm_dir / "val_labels.npy")

    with open(endofm_dir / "config.json") as f:
        endofm_cfg = json.load(f)
    frame_col = endofm_cfg.get("frame_col", "frame")

    print(f"Train: {len(train_meta)} frames, Val: {len(val_meta)} frames")

    # 確保 image_path 欄位存在
    if "image_path" not in train_meta.columns:
        image_root = Path(args.image_root)
        train_meta["image_path"] = train_meta.apply(
            lambda r: str(image_root / str(r["recording"]) / f"frame_{int(r[frame_col]):06d}.PNG"), axis=1
        )
        val_meta["image_path"] = val_meta.apply(
            lambda r: str(image_root / str(r["recording"]) / f"frame_{int(r[frame_col]):06d}.PNG"), axis=1
        )

    # 過濾不存在的圖片
    train_exists = train_meta["image_path"].map(lambda p: Path(p).exists())
    val_exists = val_meta["image_path"].map(lambda p: Path(p).exists())
    if not train_exists.all():
        print(f"Warning: {(~train_exists).sum()} train images not found, filtering...")
        mask = train_exists.values
        train_meta = train_meta[mask].reset_index(drop=True)
        train_labels = train_labels[mask]
    if not val_exists.all():
        print(f"Warning: {(~val_exists).sum()} val images not found, filtering...")
        mask = val_exists.values
        val_meta = val_meta[mask].reset_index(drop=True)
        val_labels = val_labels[mask]

    # ===== Build backbone =====
    print(f"\n{'=' * 60}")
    print(f"Building backbone: {args.backbone}")
    print("=" * 60)

    model, transform, embed_dim = build_backbone(args.backbone, device)
    print(f"Embed dim: {embed_dim}")

    # ===== Extract train features =====
    print(f"\n{'=' * 60}")
    print("Extracting train features...")
    print("=" * 60)

    train_ds = SingleFrameDataset(train_meta, args.image_root, transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    train_features = extract_features(model, train_loader, device, embed_dim, "Train")
    print(f"Train features: {train_features.shape}")

    # ===== Extract val features =====
    print(f"\n{'=' * 60}")
    print("Extracting val features...")
    print("=" * 60)

    val_ds = SingleFrameDataset(val_meta, args.image_root, transform)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    val_features = extract_features(model, val_loader, device, embed_dim, "Val")
    print(f"Val features: {val_features.shape}")

    # ===== Save =====
    print(f"\n{'=' * 60}")
    print(f"Saving to {output_dir}...")
    print("=" * 60)

    np.save(output_dir / "train_features.npy", train_features)
    np.save(output_dir / "val_features.npy", val_features)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "val_labels.npy", val_labels)
    train_meta.to_csv(output_dir / "train_meta.csv", index=False)
    val_meta.to_csv(output_dir / "val_meta.csv", index=False)

    config = {
        "backbone": args.backbone,
        "embed_dim": embed_dim,
        "frame_col": frame_col,
        "n_train": len(train_meta),
        "n_val": len(val_meta),
        "used_labels": USED_LABELS,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Label stats
    label_stats = {
        "train": {lbl: int(train_labels[:, i].sum()) for i, lbl in enumerate(USED_LABELS)},
        "val": {lbl: int(val_labels[:, i].sum()) for i, lbl in enumerate(USED_LABELS)},
        "n_train": len(train_meta), "n_val": len(val_meta),
    }
    with open(output_dir / "label_stats.json", "w") as f:
        json.dump(label_stats, f, indent=2)

    # ===== Extract test features (optional) =====
    if args.test_root:
        import re
        print(f"\n{'=' * 60}")
        print(f"Extracting test features from {args.test_root}...")
        print("=" * 60)

        test_root = Path(args.test_root)
        dfs = []
        for csv_path in sorted(test_root.glob("*.csv")):
            recording = csv_path.stem
            df = pd.read_csv(csv_path, sep=";")
            df["frame"] = df["frame_file"].apply(
                lambda x: int(re.search(r"frame_(\d+)\.png", str(x), re.I).group(1)))
            df["recording"] = recording
            df["image_path"] = (test_root / recording / df["frame_file"]).astype(str)
            dfs.append(df)
        test_meta = pd.concat(dfs, ignore_index=True)
        test_meta = test_meta[test_meta["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)

        test_ds = TestFrameDataset(test_meta, transform)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        test_features = extract_features(model, test_loader, device, embed_dim, "Test")

        np.save(output_dir / "test_features.npy", test_features)
        test_meta.to_csv(output_dir / "test_meta.csv", index=False)
        print(f"Test features: {test_features.shape}")

    total_size = (train_features.nbytes + val_features.nbytes) / 1e9
    print(f"\nTotal storage: {total_size:.2f} GB")
    print(f"Done! Features saved to {output_dir}")


if __name__ == "__main__":
    main()