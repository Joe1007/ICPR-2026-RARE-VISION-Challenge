#!/usr/bin/env python3
"""
Multi-Backbone Ensemble 預測腳本 v5
====================================
融合多個 backbone 的 ensemble 預測，加上 V4 的全部後處理技巧。

支援的 backbone 來源：
  - EndoFM-LV：推論時即時提取特徵（需要 backbone weights）
  - DINOv2 / CLIP / BiomedCLIP：使用 extract_features_dinov2.py 預提取的特徵

工作流程：
    # 1. 提取 DINOv2 特徵（含 train/val/test）
    python extract_features_dinov2.py --backbone dinov2_vitl14 \
        --output-dir features_dinov2 --test-root ./Testingset

    # 2. 在 DINOv2 特徵上訓練 ensemble（重用現有腳本）
    python train_ensemble.py --features-dir features_dinov2 \
        --ensemble-dir ensemble_dinov2

    # 3. 多 backbone 融合預測
    python predict_ensemble_v5.py --config predict_ensemble_v5.args.json

架構：
    EndoFM-LV ensemble → probs_endofm (N, 17)  ─┐
    DINOv2 ensemble    → probs_dinov2 (N, 17)   ─┼─ weighted avg → V4 post-processing → JSON
    (更多 backbone)                               ─┘
"""

import argparse
import json
import re
import sys
import bisect
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import precision_recall_curve, average_precision_score


# ============================================================
# Constants
# ============================================================

USED_LABELS = [
    "mouth", "esophagus", "stomach", "small intestine", "colon",
    "z-line", "pylorus", "ileocecal valve",
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]

VIDEO_ID_MAP = {
    "ukdd_navi_00051": "vid_001",
    "ukdd_navi_00068": "vid_002",
    "ukdd_navi_00076": "vid_003",
}

ANATOMICAL_REGIONS = ["mouth", "esophagus", "stomach", "small intestine", "colon"]
ANATOMICAL_LANDMARKS = ["z-line", "pylorus", "ileocecal valve"]
PATHOLOGICAL_LABELS = [
    "active bleeding", "angiectasia", "blood", "erosion", "erythema",
    "hematin", "lymphangioectasis", "polyp", "ulcer",
]

REGION_INDICES = [USED_LABELS.index(l) for l in ANATOMICAL_REGIONS]
LANDMARK_INDICES = [USED_LABELS.index(l) for l in ANATOMICAL_LANDMARKS]
ANATOMICAL_INDICES = REGION_INDICES + LANDMARK_INDICES
PATHOLOGICAL_INDICES = [USED_LABELS.index(l) for l in PATHOLOGICAL_LABELS]

REGION_ORDER = {"mouth": 0, "esophagus": 1, "stomach": 2, "small intestine": 3, "colon": 4}
LANDMARK_REGION_MAP = {
    "z-line": ["esophagus", "stomach"],
    "pylorus": ["stomach", "small intestine"],
    "ileocecal valve": ["small intestine", "colon"],
}


# ============================================================
# Models (V1 compatible)
# ============================================================

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(input_dim, num_labels)
    def forward(self, x): return self.classifier(self.dropout(x))


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels))
    def forward(self, x): return self.net(x)


def load_classifier(ckpt_path, embed_dim_default, device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    mt = ckpt.get("model_type", "linear")
    ed = ckpt.get("embed_dim", embed_dim_default)
    nl = ckpt.get("num_labels", len(USED_LABELS))
    hd = ckpt.get("hidden_dim", 256)
    do = ckpt.get("dropout", 0.3)
    model = LinearClassifier(ed, nl, do) if mt == "linear" else MLPClassifier(ed, nl, hd, do)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


# ============================================================
# EndoFM-LV Backbone + Test Data
# ============================================================

def build_endofm_backbone(num_frames, image_size):
    endofm_root = Path(__file__).resolve().parent / "EndoFM-LV"
    sys.path.insert(0, str(endofm_root))
    from models import get_vit_base_patch16_224
    from utils.parser import load_config
    cfg_args = SimpleNamespace(
        cfg_file=str(endofm_root / "models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"),
        opts=None, num_shards=1, shard_id=0)
    cfg = load_config(cfg_args)
    cfg.DATA.NUM_FRAMES = num_frames
    cfg.DATA.TRAIN_CROP_SIZE = image_size
    cfg.MODEL.NUM_CLASSES = 0
    cfg.TIMESFORMER.PRETRAINED_MODEL = ""
    return get_vit_base_patch16_224(cfg=cfg, no_head=True)


def load_endofm_weights(model, ckpt_path, ck="teacher"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if ck and isinstance(ckpt, dict) and ck in ckpt: ckpt = ckpt[ck]
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state"):
            if k in ckpt and isinstance(ckpt[k], dict): ckpt = ckpt[k]; break
    if isinstance(ckpt, dict) and any(k.startswith("backbone.") for k in ckpt):
        ckpt = {k[9:]: v for k, v in ckpt.items() if k.startswith("backbone.")}
    if isinstance(ckpt, dict) and "time_embed" in ckpt:
        t, s = model.time_embed, ckpt["time_embed"]
        if s.shape != t.shape:
            ckpt["time_embed"] = F.interpolate(s.transpose(1,2), size=t.shape[1], mode="nearest").transpose(1,2)
    if isinstance(ckpt, dict) and "pos_embed" in ckpt:
        t, s = model.pos_embed, ckpt["pos_embed"]
        if s.shape != t.shape:
            cp = s[:,:1,:]; pp = s[:,1:,:]
            n = pp.shape[1]; g = int(n**0.5); tp = t.shape[1]-1; tg = int(tp**0.5)
            if g*g==n and tg*tg==tp:
                pp = F.interpolate(pp.transpose(1,2).reshape(1,pp.shape[2],g,g), size=(tg,tg), mode="nearest").flatten(2).transpose(1,2)
                ckpt["pos_embed"] = torch.cat((cp, pp), dim=1)
    model.load_state_dict(ckpt, strict=False)


def load_test_csvs(test_root):
    dfs = []
    for csv_path in sorted(Path(test_root).glob("*.csv")):
        df = pd.read_csv(csv_path, sep=";")
        df["frame"] = df["frame_file"].apply(lambda x: int(re.search(r"frame_(\d+)\.png", str(x), re.I).group(1)))
        df["recording"] = csv_path.stem
        df["image_path"] = (Path(test_root) / csv_path.stem / df["frame_file"]).astype(str)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


class TestClipDataset(Dataset):
    def __init__(self, df, transform, num_frames, frame_stride, clip_mode):
        self.df = df.reset_index(drop=True); self.transform = transform
        self.num_frames = num_frames; self.frame_stride = frame_stride; self.clip_mode = clip_mode
        self._pm, self._ir = {}, {}
        for i, r in self.df.iterrows():
            rc = str(r["recording"]); fi = int(r["frame"])
            self._pm[(rc, fi)] = Path(r["image_path"])
            self._ir.setdefault(rc, []).append(fi)
        for r in self._ir: self._ir[r].sort()
    def __len__(self): return len(self.df)
    def _pfi(self, rc, fi):
        if (rc, fi) in self._pm: return self._pm[(rc, fi)]
        idx = self._ir.get(rc, [])
        p = bisect.bisect_left(idx, fi)
        if p == 0: n = idx[0]
        elif p >= len(idx): n = idx[-1]
        else:
            b, a = idx[p-1], idx[p]; n = b if abs(fi-b) <= abs(a-fi) else a
        return self._pm[(rc, n)]
    def _bi(self, c):
        if self.num_frames <= 1: return [c]
        s = max(1, self.frame_stride)
        if self.clip_mode == "forward": return [c+i*s for i in range(self.num_frames)]
        st = -self.num_frames//2; return [c+i*s for i in range(st, st+self.num_frames)]
    def __getitem__(self, idx):
        r = self.df.iloc[idx]; rc = str(r["recording"]); c = int(r["frame"])
        fs = [self.transform(Image.open(self._pfi(rc, fi)).convert("RGB")) for fi in self._bi(c)]
        return torch.stack(fs, dim=1), idx


class TestClipDatasetTTA(TestClipDataset):
    def __init__(self, df, transform_n, transform_f, num_frames, frame_stride, clip_mode):
        super().__init__(df, transform_n, num_frames, frame_stride, clip_mode)
        self.tf = transform_f
    def __getitem__(self, idx):
        r = self.df.iloc[idx]; rc = str(r["recording"]); c = int(r["frame"])
        fn, ff = [], []
        for fi in self._bi(c):
            img = Image.open(self._pfi(rc, fi)).convert("RGB")
            fn.append(self.transform(img)); ff.append(self.tf(img))
        return torch.stack(fn, dim=1), torch.stack(ff, dim=1), idx


# ============================================================
# V4 Post-processing (complete pipeline)
# ============================================================

def tiou(a, b):
    i = max(0, min(a["end"],b["end"])-max(a["start"],b["start"])+1)
    u = (a["end"]-a["start"]+1)+(b["end"]-b["start"]+1)-i
    return i/u if u > 0 else 0.0

def ap_single(gs, ps, t):
    if not gs: return 1.0 if not ps else 0.0
    m = set(); tp = []
    for p in ps:
        h = False
        for i, g in enumerate(gs):
            if i not in m and tiou(p, g) >= t: m.add(i); h = True; break
        tp.append(1 if h else 0)
    if not tp: return 0.0
    c, ap, pr = 0, 0.0, 0.0
    for i, v in enumerate(tp):
        c += v; p = c/(i+1); r = c/len(gs); ap += p*(r-pr); pr = r
    return ap

def events_to_ld(events):
    """Convert events to per-video per-label segment lists. Handles both per-label and tuple-based events."""
    o = {}
    for e in events:
        lbl = e["label"]
        if isinstance(lbl, str):
            labels = [lbl]
        elif isinstance(lbl, list):
            labels = lbl if lbl else []
        else:
            labels = []
        for l in labels:
            if l in USED_LABELS:
                o.setdefault(l, []).append({"start": e["start"], "end": e["end"]})
    return o

def compute_tmap(gt_ev, pr_ev, t=0.5):
    vm = []
    for v in gt_ev:
        aps = [ap_single(gt_ev[v].get(l,[]), pr_ev.get(v,{}).get(l,[]), t) for l in USED_LABELS]
        vm.append(np.mean(aps))
    return float(np.mean(vm)) if vm else 0.0


def compute_tmap_anatomical_pathological(gt_ev, pr_ev, t=0.5):
    """Return (anatomical_mAP, pathological_mAP). gt_ev/pr_ev: {video: {label: [segments]}}."""
    aps = {l: [] for l in USED_LABELS}
    for v in gt_ev:
        for l in USED_LABELS:
            ap = ap_single(gt_ev[v].get(l, []), pr_ev.get(v, {}).get(l, []), t)
            aps[l].append(ap)
    anat_vals = [np.mean(aps[l]) for l in ANATOMICAL_REGIONS + ANATOMICAL_LANDMARKS if aps[l]]
    path_vals = [np.mean(aps[l]) for l in PATHOLOGICAL_LABELS if aps[l]]
    vanat = float(np.mean(anat_vals)) if anat_vals else None
    vpath = float(np.mean(path_vals)) if path_vals else None
    return vanat, vpath

def binary_to_events(binary, frames):
    """Per-label event generation: each label independently finds contiguous segments."""
    events = []
    T = len(frames)
    for c, lbl in enumerate(USED_LABELS):
        v = binary[:, c]; ie = False; st = None
        for i in range(T):
            if v[i] == 1:
                if not ie: st = int(frames[i]); ie = True
            else:
                if ie: events.append({"start": st, "end": int(frames[i-1]), "label": [lbl]}); ie = False
        if ie: events.append({"start": st, "end": int(frames[T-1]), "label": [lbl]})
    return events


def binary_to_events_tuple(binary, frames):
    """Tuple-based event generation: segment by per-frame 17-label state changes."""
    events = []
    T = len(frames)
    if T == 0:
        return events
    current = tuple(sorted(USED_LABELS[c] for c in range(len(USED_LABELS)) if binary[0, c] == 1))
    start = int(frames[0])
    for i in range(1, T):
        state = tuple(sorted(USED_LABELS[c] for c in range(len(USED_LABELS)) if binary[i, c] == 1))
        if state != current:
            events.append({"start": start, "end": int(frames[i - 1]), "label": list(current)})
            start = int(frames[i])
            current = state
    events.append({"start": start, "end": int(frames[T - 1]), "label": list(current)})
    return events

def smooth_by_video(pv, wa=15, wp=7, m="median"):
    from scipy.ndimage import median_filter, uniform_filter1d, gaussian_filter1d
    def _s(a, w, m):
        if w <= 1: return a
        if m == "median": return median_filter(a, size=w, mode="nearest")
        elif m == "gaussian": return gaussian_filter1d(a, sigma=w/4.0, mode="nearest")
        return uniform_filter1d(a, size=w, mode="nearest")
    r = {}
    for v, p in pv.items():
        s = np.zeros_like(p)
        for i in ANATOMICAL_INDICES: s[:,i] = _s(p[:,i], wa, m)
        for i in PATHOLOGICAL_INDICES: s[:,i] = _s(p[:,i], wp, m)
        r[v] = s
    return r

def enforce_excl(pv):
    r = {}
    for v, p in pv.items():
        o = p.copy(); mx = np.argmax(p[:, REGION_INDICES], axis=1)
        for ri in REGION_INDICES: o[:, ri] = 0.0
        for i in range(len(p)): o[i, REGION_INDICES[mx[i]]] = p[i, REGION_INDICES[mx[i]]]
        r[v] = o
    return r

def enforce_order(bv, pv):
    r = {}
    for v in bv:
        o = bv[v].copy(); p = pv[v]; T = len(o); mro = -1
        for i in range(T):
            rp = p[i, REGION_INDICES]; bi = np.argmax(rp)
            bo = REGION_ORDER[ANATOMICAL_REGIONS[bi]]
            if bo >= mro: mro = bo
            else:
                for ri in REGION_INDICES: o[i, ri] = 0
                fi = list(REGION_ORDER.values()).index(mro)
                o[i, REGION_INDICES[fi]] = 1
        r[v] = o
    return r

def enforce_landmark(bv):
    r = {}
    for v, b in bv.items():
        o = b.copy(); T = len(o)
        for lm, regs in LANDMARK_REGION_MAP.items():
            li = USED_LABELS.index(lm); ris = [USED_LABELS.index(rr) for rr in regs]
            for i in range(T):
                if o[i, li] == 1:
                    w = 50; s = max(0, i-w); e = min(T, i+w+1)
                    if o[s:e, ris].sum() == 0: o[i, li] = 0
        r[v] = o
    return r

def morph_pp(bv, ma=10, mp_=3, ga=5, gp=2):
    from scipy.ndimage import binary_opening, binary_closing
    r = {}
    for v, b in bv.items():
        o = b.copy()
        for i in ANATOMICAL_INDICES:
            c = o[:,i].astype(bool)
            if ma > 1: c = binary_opening(c, structure=np.ones(ma))
            if ga > 0: c = binary_closing(c, structure=np.ones(ga+1))
            o[:,i] = c.astype(np.int32)
        for i in PATHOLOGICAL_INDICES:
            c = o[:,i].astype(bool)
            if mp_ > 1: c = binary_opening(c, structure=np.ones(mp_))
            if gp > 0: c = binary_closing(c, structure=np.ones(gp+1))
            o[:,i] = c.astype(np.int32)
        r[v] = o
    return r

def ensure_region(bv, pv):
    r = {}
    for v in bv:
        o = bv[v].copy(); p = pv[v]
        for i in range(len(o)):
            if o[i, REGION_INDICES].sum() == 0:
                o[i, REGION_INDICES[np.argmax(p[i, REGION_INDICES])]] = 1
        r[v] = o
    return r

def find_thresh_f1(gt, pr):
    th = {}
    for i, l in enumerate(USED_LABELS):
        if gt[:, i].sum() == 0: th[l] = 0.5; continue
        ps, rs, ts = precision_recall_curve(gt[:, i], pr[:, i])
        bf, bt = 0.0, 0.5
        for j, t in enumerate(ts):
            if ps[j]+rs[j] == 0: continue
            f = 2*ps[j]*rs[j]/(ps[j]+rs[j])
            if f > bf: bf = f; bt = float(t)
        th[l] = bt
    return th

def find_thresh_tmap(sv, gv, fv, base_th, pp):
    th = base_th.copy()
    for c, lbl in enumerate(USED_LABELS):
        bt = base_th[lbl]; bm = 0.0
        for t in np.linspace(max(0.05, bt-0.15), min(0.95, bt+0.15), 9):
            tt = th.copy(); tt[lbl] = float(t)
            binary = {}
            for v, p in sv.items():
                b = np.zeros_like(p, dtype=np.int32)
                for i2, l2 in enumerate(USED_LABELS): b[:,i2] = (p[:,i2] >= tt[l2]).astype(np.int32)
                binary[v] = b
            binary = morph_pp(binary, pp.get("min_a",10), pp.get("min_p",3), pp.get("gap_a",5), pp.get("gap_p",2))
            binary = ensure_region(binary, sv)
            ge, pe = {}, {}
            for v in gv:
                ge[v] = events_to_ld(binary_to_events(gv[v], fv[v]))
                pe[v] = events_to_ld(binary_to_events(binary[v], fv[v]))
            m = compute_tmap(ge, pe, 0.5)
            if m > bm: bm = m; th[lbl] = float(t)
    return th

def search_pp(sv, gv, fv, th):
    print("  Searching PP params...")
    cur = {"wa":15,"wp":7,"min_a":10,"min_p":3,"gap_a":5,"gap_p":2}
    search = {"wa":[9,15,21,31],"wp":[3,5,7,11],"min_a":[5,10,15,20],"min_p":[2,3,5],"gap_a":[3,5,8,12],"gap_p":[1,2,3,5]}
    for pn, vals in search.items():
        bv_, bm_ = cur[pn], 0.0
        for v in vals:
            p = cur.copy(); p[pn] = v
            binary = {}
            for vid, pr in sv.items():
                b = np.zeros_like(pr, dtype=np.int32)
                for i, l in enumerate(USED_LABELS): b[:,i] = (pr[:,i] >= th[l]).astype(np.int32)
                binary[vid] = b
            binary = morph_pp(binary, p["min_a"], p["min_p"], p["gap_a"], p["gap_p"])
            binary = ensure_region(binary, sv)
            ge, pe = {}, {}
            for vid in gv:
                ge[vid] = events_to_ld(binary_to_events(gv[vid], fv[vid]))
                pe[vid] = events_to_ld(binary_to_events(binary[vid], fv[vid]))
            m = compute_tmap(ge, pe, 0.5)
            if m > bm_: bm_ = m; bv_ = v
        cur[pn] = bv_
        print(f"    {pn}: {bv_} (mAP@0.5={bm_:.4f})")
    return cur

def apply_temperature(p, t):
    if t == 1.0: return p
    l = np.log(p.clip(1e-7,1-1e-7)/(1-p.clip(1e-7,1-1e-7)))
    return 1.0/(1.0+np.exp(-l/t))

def find_temperature(p, gt):
    bm, bt = 0.0, 1.0
    for t in [0.5,0.7,0.8,0.9,1.0,1.1,1.2,1.5,2.0]:
        sp = apply_temperature(p, t)
        aps = [average_precision_score(gt[:,c], sp[:,c]) for c in range(len(USED_LABELS)) if gt[:,c].sum()>0]
        m = np.mean(aps) if aps else 0
        if m > bm: bm = m; bt = t
    return bt

def full_pipeline(pv, th, wa=15, wp=7, sm="median", min_a=10, min_p=3, gap_a=5, gap_p=2, abb=None):
    """abb: ablation dict with no_smooth, no_enforce_excl, no_enforce_order, no_enforce_landmark, no_morph_pp, no_ensure_region"""
    if abb is None: abb = {}
    s = pv if abb.get("no_smooth") else smooth_by_video(pv, wa, wp, sm)
    s = s if abb.get("no_enforce_excl") else enforce_excl(s)
    b = {}
    for v, p in s.items():
        bb = np.zeros_like(p, dtype=np.int32)
        for i, l in enumerate(USED_LABELS): bb[:,i] = (p[:,i] >= th.get(l,0.5)).astype(np.int32)
        b[v] = bb
    b = b if abb.get("no_morph_pp") else morph_pp(b, min_a, min_p, gap_a, gap_p)
    b = b if abb.get("no_enforce_order") else enforce_order(b, s)
    b = b if abb.get("no_enforce_landmark") else enforce_landmark(b)
    b = b if abb.get("no_ensure_region") else ensure_region(b, s)
    return b, s

def weighted_ens(all_p, w):
    ns, nc = all_p[0].shape
    r = np.zeros((ns, nc), dtype=np.float32)
    for m in range(len(all_p)):
        for c in range(nc): r[:,c] += all_p[m][:,c]*w[m,c]
    return r

def compute_weights(all_p, gt):
    nm = len(all_p); nc = len(USED_LABELS)
    w = np.ones((nm, nc), dtype=np.float32)
    for m in range(nm):
        for c in range(nc):
            if gt[:,c].sum() == 0: continue
            try: w[m,c] = max(average_precision_score(gt[:,c], all_p[m][:,c]), 0.01)
            except: pass
    for c in range(nc):
        t = w[:,c].sum(); w[:,c] = w[:,c]/t if t > 0 else 1.0/nm
    return w


# ============================================================
# Per-backbone ensemble prediction
# ============================================================

def predict_backbone_ensemble(features, model_paths, embed_dim, device):
    """Run all models in an ensemble on given features, return list of per-model probs"""
    all_probs = []
    for mp in model_paths:
        model = load_classifier(mp, embed_dim, device)
        p = np.zeros((len(features), len(USED_LABELS)), dtype=np.float32)
        ft = torch.from_numpy(features).float()
        with torch.no_grad():
            for j in range(0, len(ft), 2048):
                b = ft[j:j+2048].to(device)
                p[j:j+len(b)] = torch.sigmoid(model(b)).cpu().numpy()
        all_probs.append(p)
        del model
    return all_probs


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-backbone ensemble prediction v5")
    parser.add_argument("--config", default="")
    parser.add_argument("--test-root", default="./Testingset")

    # EndoFM backbone
    parser.add_argument("--endofm-features-dir", default="features")
    parser.add_argument("--endofm-ensemble-dir", default="ensemble")
    parser.add_argument("--pretrained-weights", default="")
    parser.add_argument("--checkpoint-key", default="teacher")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--clip-mode", default="center")
    parser.add_argument("--image-size", type=int, default=224)

    # Additional backbones: list of (features_dir, ensemble_dir) pairs
    parser.add_argument("--extra-backbones", type=str, default="",
                        help='JSON list: [{"features_dir":"features_dinov2","ensemble_dir":"ensemble_dinov2"}, ...]')

    # Inference
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--out-json", default="test_v5_pred.json")

    # Backbone weighting
    parser.add_argument("--backbone-weights", type=str, default="",
                        help='JSON: {"endofm": 1.0, "dinov2": 1.0}. Empty = auto from val AP.')

    # Ablation flags (for run_ablation_v5.py)
    parser.add_argument("--ablation-uniform-model-weights", action="store_true", default=False)
    parser.add_argument("--ablation-uniform-backbone-weights", action="store_true", default=False)
    parser.add_argument("--ablation-fixed-temperature", type=float, default=None)
    parser.add_argument("--ablation-no-pp-search", action="store_true", default=False)
    parser.add_argument("--ablation-no-smooth", action="store_true", default=False)
    parser.add_argument("--ablation-no-enforce-excl", action="store_true", default=False)
    parser.add_argument("--ablation-no-enforce-order", action="store_true", default=False)
    parser.add_argument("--ablation-no-enforce-landmark", action="store_true", default=False)
    parser.add_argument("--ablation-no-morph-pp", action="store_true", default=False)
    parser.add_argument("--ablation-no-ensure-region", action="store_true", default=False)
    parser.add_argument("--ablation-backbone-only", type=str, default="",
                        help='Only use one backbone: "endofm" or "dinov3_vitl16" (name from extra_backbones)')
    parser.add_argument("--ablation-tuple-event", action="store_true", default=False,
                        help="Use tuple-based event generation instead of per-label")

    # Load config
    args, _ = parser.parse_known_args()
    cp = args.config
    if not cp:
        dp = Path(__file__).with_suffix(".args.json")
        if dp.exists(): cp = str(dp)
    if cp and Path(cp).exists():
        with open(cp) as f: parser.set_defaults(**json.load(f))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_root = Path(args.test_root)
    endofm_fdir = Path(args.endofm_features_dir)
    endofm_edir = Path(args.endofm_ensemble_dir)

    # Load EndoFM config
    with open(endofm_fdir / "config.json") as f: efcfg = json.load(f)
    args.num_frames = efcfg.get("num_frames", args.num_frames)
    args.frame_stride = efcfg.get("frame_stride", args.frame_stride)
    args.clip_mode = efcfg.get("clip_mode", args.clip_mode)
    args.image_size = efcfg.get("image_size", args.image_size)
    if not args.pretrained_weights: args.pretrained_weights = efcfg.get("pretrained_weights", "")
    frame_col = efcfg.get("frame_col", "frame")

    # Parse extra backbones (可為 list、路徑字串、或 JSON 字串)
    extra_backbones = []
    if args.extra_backbones:
        if isinstance(args.extra_backbones, list):
            extra_backbones = args.extra_backbones
        else:
            p = Path(args.extra_backbones)
            if p.exists():
                with open(p) as f: extra_backbones = json.load(f)
            else:
                extra_backbones = json.loads(args.extra_backbones)

    # Ablation: 只用單一 backbone
    ablation_backbone_only = getattr(args, "ablation_backbone_only", "") or ""
    if ablation_backbone_only == "endofm":
        extra_backbones = []
        print("Ablation: endofm_only")
    elif ablation_backbone_only:
        # 只保留指定名稱的 backbone，並跳過 EndoFM
        matched = [eb for eb in extra_backbones if eb.get("name", "") == ablation_backbone_only]
        if not matched:
            matched = [eb for eb in extra_backbones if ablation_backbone_only in str(eb.get("name", ""))]
        extra_backbones = matched if matched else extra_backbones
        args._ablation_skip_endofm = True  # 標記跳過 EndoFM
        print(f"Ablation: {ablation_backbone_only}_only (skip EndoFM)")
    else:
        args._ablation_skip_endofm = False

    print(f"Device: {device} | TTA: {args.tta}")
    print(f"Backbones: EndoFM-LV + {len(extra_backbones)} extra")

    # ================================================================
    # Step 1: Validation — all backbones
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 1: Validation analysis (all backbones)...")
    print("=" * 60)

    val_labels = np.load(endofm_fdir / "val_labels.npy")
    val_meta = pd.read_csv(endofm_fdir / "val_meta.csv")
    skip_endofm = getattr(args, "_ablation_skip_endofm", False)

    # --- EndoFM validation ---
    endofm_val_probs = None
    endofm_models = []
    endofm_edim = 768
    if not skip_endofm:
        val_features_endofm = np.load(endofm_fdir / "val_features.npy")
        with open(endofm_edir / "manifest.json") as f: em = json.load(f)
        endofm_models = [Path(m["path"]) for m in em["models"]]
        endofm_edim = em.get("embed_dim", 768)
        print(f"  EndoFM: {len(endofm_models)} models, dim={endofm_edim}")
        endofm_val_probs = predict_backbone_ensemble(val_features_endofm, endofm_models, endofm_edim, device)

    # --- Extra backbones validation ---
    extra_val_probs_list = []  # list of lists
    extra_configs = []
    for eb in extra_backbones:
        fdir = Path(eb["features_dir"])
        edir = Path(eb["ensemble_dir"])
        vf = np.load(fdir / "val_features.npy")
        with open(edir / "manifest.json") as f: mf = json.load(f)
        models = [Path(m["path"]) for m in mf["models"]]
        edim = mf.get("embed_dim", vf.shape[1])
        name = eb.get("name", fdir.stem)
        print(f"  {name}: {len(models)} models, dim={edim}")
        vp = predict_backbone_ensemble(vf, models, edim, device)
        extra_val_probs_list.append(vp)
        extra_configs.append({"name": name, "fdir": fdir, "edir": edir, "models": models, "edim": edim})

    # --- Compute per-backbone weighted ensemble ---
    # First: within each backbone, compute weighted average
    endofm_w = None
    endofm_val_ens = None
    if not skip_endofm and endofm_val_probs is not None:
        if getattr(args, "ablation_uniform_model_weights", False):
            nm = len(endofm_val_probs); nc = len(USED_LABELS)
            endofm_w = np.ones((nm, nc), dtype=np.float32) / nm
        else:
            endofm_w = compute_weights(endofm_val_probs, val_labels)
        endofm_val_ens = weighted_ens(endofm_val_probs, endofm_w)

    extra_val_ens_list = []
    extra_weights_list = []
    for vp in extra_val_probs_list:
        if getattr(args, "ablation_uniform_model_weights", False):
            nm, nc = len(vp), len(USED_LABELS)
            w = np.ones((nm, nc), dtype=np.float32) / nm
        else:
            w = compute_weights(vp, val_labels)
        extra_val_ens_list.append(weighted_ens(vp, w))
        extra_weights_list.append(w)

    # --- Cross-backbone weighting ---
    if endofm_val_ens is not None:
        all_backbone_ens = [endofm_val_ens] + extra_val_ens_list
        backbone_names = ["endofm"] + [ec["name"] for ec in extra_configs]
    else:
        all_backbone_ens = extra_val_ens_list
        backbone_names = [ec["name"] for ec in extra_configs]
    if not all_backbone_ens:
        raise RuntimeError("No backbones available. Check ablation_backbone_only and extra_backbones.")

    if getattr(args, "ablation_uniform_backbone_weights", False):
        backbone_w = np.ones(len(all_backbone_ens)) / len(all_backbone_ens)
    elif args.backbone_weights:
        bw = json.loads(args.backbone_weights) if isinstance(args.backbone_weights, str) else args.backbone_weights
        backbone_w = np.array([bw.get(n, 1.0) for n in backbone_names])
    else:
        # Auto: weight by frame mAP
        backbone_w = np.zeros(len(all_backbone_ens))
        for bi, bp in enumerate(all_backbone_ens):
            aps = [average_precision_score(val_labels[:,c], bp[:,c]) for c in range(len(USED_LABELS)) if val_labels[:,c].sum()>0]
            backbone_w[bi] = np.mean(aps)
            print(f"  {backbone_names[bi]} frame mAP: {backbone_w[bi]:.4f}")

    backbone_w = backbone_w / backbone_w.sum()
    print(f"\n  Backbone weights: {dict(zip(backbone_names, [f'{w:.3f}' for w in backbone_w]))}")

    # Combine
    val_combined = sum(bp * w for bp, w in zip(all_backbone_ens, backbone_w))

    # Temperature
    fixed_temp = getattr(args, "ablation_fixed_temperature", None)
    if fixed_temp is not None:
        temp = float(fixed_temp)
        print(f"  Temperature: {temp} (fixed, ablation)")
    else:
        temp = find_temperature(val_combined, val_labels)
        print(f"  Temperature: {temp}")
    val_combined = apply_temperature(val_combined, temp)

    # Organize by video
    vpv, vgv, vfv = {}, {}, {}
    for rec, grp in val_meta.groupby("recording"):
        grp = grp.sort_values(frame_col); idx = grp.index.tolist(); v = str(rec)
        vpv[v] = val_combined[idx]; vgv[v] = val_labels[idx]; vfv[v] = grp[frame_col].values.astype(int)

    # Ablation dict for post-processing
    abb = {
        "no_smooth": getattr(args, "ablation_no_smooth", False),
        "no_enforce_excl": getattr(args, "ablation_no_enforce_excl", False),
        "no_enforce_order": getattr(args, "ablation_no_enforce_order", False),
        "no_enforce_landmark": getattr(args, "ablation_no_enforce_landmark", False),
        "no_morph_pp": getattr(args, "ablation_no_morph_pp", False),
        "no_ensure_region": getattr(args, "ablation_no_ensure_region", False),
    }

    # Smooth + threshold search
    if abb["no_smooth"]:
        sv = vpv
    else:
        sv = smooth_by_video(vpv, 15, 7, "median")
    sv = sv if abb["no_enforce_excl"] else enforce_excl(sv)
    gf = np.concatenate([vgv[v] for v in sorted(vgv)])
    pf = np.concatenate([sv[v] for v in sorted(sv)])
    f1_th = find_thresh_f1(gf, pf)

    if getattr(args, "ablation_no_pp_search", False):
        pp = {"wa": 15, "wp": 7, "min_a": 10, "min_p": 3, "gap_a": 5, "gap_p": 2}
        print("  Using default PP params (ablation)")
    else:
        pp = search_pp(sv, vgv, vfv, f1_th)

    if abb["no_smooth"]:
        sv2 = vpv
    else:
        sv2 = smooth_by_video(vpv, pp["wa"], pp["wp"], "median")
    sv2 = sv2 if abb["no_enforce_excl"] else enforce_excl(sv2)
    pf2 = np.concatenate([sv2[v] for v in sorted(sv2)])
    f1_th = find_thresh_f1(gf, pf2)

    print("\n  Optimizing thresholds for temporal mAP...")
    map_th = find_thresh_tmap(sv2, vgv, vfv, f1_th, pp)

    # Val eval
    use_tuple_event = getattr(args, "ablation_tuple_event", False)
    event_fn = binary_to_events_tuple if use_tuple_event else binary_to_events
    bval, _ = full_pipeline(vpv, map_th, pp["wa"], pp["wp"], "median", pp["min_a"], pp["min_p"], pp["gap_a"], pp["gap_p"], abb)
    ge, pe = {}, {}
    for v in vgv:
        ge[v] = events_to_ld(binary_to_events(vgv[v], vfv[v]))  # gt always per-label for fair comparison
        pe[v] = events_to_ld(event_fn(bval[v], vfv[v]))
    vm05, vm95 = compute_tmap(ge, pe, 0.5), compute_tmap(ge, pe, 0.95)
    vanat, vpath = compute_tmap_anatomical_pathological(ge, pe, 0.5)
    print(f"\n  Val mAP@0.5={vm05:.4f}, mAP@0.95={vm95:.4f}", flush=True)
    if vanat is not None: print(f"  Anatomical mAP@0.5={vanat:.4f}, Pathological mAP@0.5={vpath:.4f}", flush=True)
    # 寫入 val_metrics 供 ablation runner 讀取（避免 stdout 解析失敗）
    val_metrics_path = Path(args.out_json).with_suffix(".val_metrics.json")
    with open(val_metrics_path, "w") as f:
        json.dump({"mAP_05": vm05, "mAP_95": vm95, "anatomical_mAP_05": vanat, "pathological_mAP_05": vpath}, f)

    # ================================================================
    # Step 2: Test — extract EndoFM features (skip if ablation_backbone_only)
    # ================================================================
    test_df = load_test_csvs(test_root)
    test_df = test_df[test_df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    print(f"\n  Test: {len(test_df)} frames")

    endofm_test_ens = None
    if not skip_endofm:
        print("\n" + "=" * 60)
        print("Step 2: Test feature extraction (EndoFM)...")
        print("=" * 60)
        backbone = build_endofm_backbone(args.num_frames, args.image_size)
        load_endofm_weights(backbone, args.pretrained_weights, args.checkpoint_key)
        backbone = backbone.to(device).eval()
        tn = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        if args.tta:
            tf = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            ds = TestClipDatasetTTA(test_df, tn, tf, args.num_frames, args.frame_stride, args.clip_mode)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            fn = np.zeros((len(ds), backbone.embed_dim), dtype=np.float32)
            ff = np.zeros((len(ds), backbone.embed_dim), dtype=np.float32)
            with torch.no_grad():
                for cn, cf, idx in tqdm(dl, desc="EndoFM TTA"):
                    cn, cf = cn.to(device), cf.to(device)
                    c1, _ = backbone(cn); c2, _ = backbone(cf)
                    fn[idx.numpy()] = c1.cpu().numpy(); ff[idx.numpy()] = c2.cpu().numpy()
            endofm_test_feats = [fn, ff]
        else:
            ds = TestClipDataset(test_df, tn, args.num_frames, args.frame_stride, args.clip_mode)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            fn = np.zeros((len(ds), backbone.embed_dim), dtype=np.float32)
            with torch.no_grad():
                for cl, idx in tqdm(dl, desc="EndoFM"):
                    c, _ = backbone(cl.to(device)); fn[idx.numpy()] = c.cpu().numpy()
            endofm_test_feats = [fn]
        del backbone; torch.cuda.empty_cache()
        endofm_aug_probs = []
        for fi, feats in enumerate(endofm_test_feats):
            mp = predict_backbone_ensemble(feats, endofm_models, endofm_edim, device)
            endofm_aug_probs.append(weighted_ens(mp, endofm_w))
        endofm_test_ens = np.mean(endofm_aug_probs, axis=0)
        print(f"  EndoFM: done ({len(endofm_test_feats)} augs)")

    # ================================================================
    # Step 3: Test — predict all backbones
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 3: Multi-backbone prediction...")
    print("=" * 60)

    # Extra backbone predictions
    extra_test_ens_list = []
    for bi, ec in enumerate(extra_configs):
        test_feat_path = ec["fdir"] / "test_features.npy"
        if not test_feat_path.exists():
            print(f"  WARNING: {test_feat_path} not found! Skipping {ec['name']}.")
            print(f"  Run: python extract_features_dinov2.py --backbone <name> --output-dir {ec['fdir']} --test-root {test_root}")
            continue
        tf = np.load(test_feat_path)
        mp = predict_backbone_ensemble(tf, ec["models"], ec["edim"], device)
        extra_test_ens_list.append(weighted_ens(mp, extra_weights_list[bi]))
        print(f"  {ec['name']}: done")

    # Cross-backbone combine
    if endofm_test_ens is not None:
        all_test_ens = [endofm_test_ens] + extra_test_ens_list
    else:
        all_test_ens = extra_test_ens_list
    actual_weights = backbone_w[:len(all_test_ens)]
    actual_weights = actual_weights / actual_weights.sum()
    test_combined = sum(p * w for p, w in zip(all_test_ens, actual_weights))
    test_combined = apply_temperature(test_combined, temp)

    # ================================================================
    # Step 4: Post-processing + output
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 4: Post-processing...")
    print("=" * 60)

    tpv, tfv = {}, {}
    for rec, grp in test_df.groupby("recording"):
        grp = grp.sort_values("frame"); idx = grp.index.tolist(); v = str(rec)
        tpv[v] = test_combined[idx]; tfv[v] = grp["frame"].values.astype(int)

    bt, st = full_pipeline(tpv, map_th, pp["wa"], pp["wp"], "median", pp["min_a"], pp["min_p"], pp["gap_a"], pp["gap_p"], abb)

    vout = []
    for v in sorted(bt):
        # 評分系統需要原始 ID (ukdd_navi_00051 等)，不使用 vid_001 映射
        did = v
        ev = event_fn(bt[v], tfv[v])
        vout.append({"video_id": did, "events": ev})
        lc = {}
        for e in ev:
            for lbl in (e["label"] if isinstance(e["label"], list) else [e["label"]]):
                lc[lbl] = lc.get(lbl, 0) + 1
        print(f"\n  {did}: {len(ev)} events")
        for l in USED_LABELS:
            if l in lc: print(f"    {l:20s}: {lc[l]} events")

    with open(args.out_json, "w") as f:
        json.dump({"videos": vout}, f, indent=2)
    print(f"\nSaved {args.out_json}")

    with open(Path(args.out_json).with_suffix(".config.json"), "w") as f:
        json.dump({"thresholds": map_th, "pp": pp, "temperature": temp,
                    "backbone_weights": dict(zip(backbone_names[:len(actual_weights)],
                                                  actual_weights.tolist()))}, f, indent=2, default=str)
    print("Done!")


if __name__ == "__main__":
    main()