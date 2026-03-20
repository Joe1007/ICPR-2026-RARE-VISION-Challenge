#!/usr/bin/env python3
"""
Ablation Study (Guideline-compliant)
====================================
以 Full pipeline 為中心的 ablation study，驗證：
1. backbone choice 是否重要
2. hierarchical fusion / calibration 是否重要
3. temporal decoding 與 per-label event generation 是否重要

實驗規格：
- 相同 train/val split、17 labels
- temporal mAP@0.5、mAP@0.95
- 可選 anatomical / pathological average

使用方式：
    # 執行全部實驗（需先準備 ResNet-50 features + ensemble）
    python run_ablation_guideline.py --base-config predict_ensembleV5.args.json --output ablation_guideline_results.json

    # 只執行不需 ResNet-50 的實驗
    python run_ablation_guideline.py --ablations A0,A2,A3,A4,B1,B2,B3,C1,C2,C3,C4,C5,C6,C7,C8 --output ablation_guideline_results.json

    # 準備 ResNet-50（Exp-A1 需要，需先執行）：
    python extract_features_dinov2.py --backbone resnet50 --output-dir features_resnet50 --test-root ./Testingset
    python train_ensemble.py --features-dir features_resnet50 --ensemble-dir ensemble_resnet50

    # 優先 8 組（算力有限時）：
    python run_ablation_guideline.py --ablations priority8 --output ablation_guideline_results.json
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

DEFAULT_BASE_CONFIG = "predict_ensembleV5.args.json"
PROJECT_ROOT = Path(__file__).resolve().parent

# Base config 需包含 EndoFM + DINOv3；ResNet-50 為可選
BASE_EXTRA_BACKBONES = [
    {"name": "dinov3_vitl16", "features_dir": "features_dinov3", "ensemble_dir": "ensemble_dinov3"},
]
RESNET50_BACKBONE = {"name": "resnet50", "features_dir": "features_resnet50", "ensemble_dir": "ensemble_resnet50"}


# 實驗定義：Exp-ID -> (desc, overrides, needs_resnet50)
EXPERIMENTS = {
    # Group A: Backbone ablation
    "A0": {
        "desc": "Full method (reference)",
        "variant": "Full",
        "overrides": {},
        "needs_resnet50": False,
        "backbones": "EndoFM+DINOv3",
        "fusion": "weighted",
        "calibration": "T+TT",
        "temporal_decoding": "full",
        "event_gen": "per-label",
    },
    "A1": {
        "desc": "ResNet-50 only",
        "variant": "ResNet-50 only",
        "overrides": {"extra_backbones": [RESNET50_BACKBONE], "ablation_backbone_only": "resnet50"},
        "needs_resnet50": True,
        "backbones": "ResNet",
        "fusion": "single-backbone",
        "calibration": "T+TT",
        "temporal_decoding": "full",
        "event_gen": "per-label",
    },
    "A2": {
        "desc": "DINOv3 only",
        "variant": "DINOv3 only",
        "overrides": {"ablation_backbone_only": "dinov3_vitl16"},
        "needs_resnet50": False,
        "backbones": "DINOv3",
        "fusion": "single-backbone",
        "calibration": "T+TT",
        "temporal_decoding": "full",
        "event_gen": "per-label",
    },
    "A3": {
        "desc": "EndoFM-LV only",
        "variant": "EndoFM only",
        "overrides": {"ablation_backbone_only": "endofm"},
        "needs_resnet50": False,
        "backbones": "EndoFM",
        "fusion": "single-backbone",
        "calibration": "T+TT",
        "temporal_decoding": "full",
        "event_gen": "per-label",
    },
    "A4": {
        "desc": "EndoFM+DINOv3, uniform backbone weights",
        "variant": "Uniform backbone weights",
        "overrides": {"ablation_uniform_backbone_weights": True},
        "needs_resnet50": False,
        "backbones": "EndoFM+DINOv3",
        "fusion": "uniform",
        "calibration": "T+TT",
        "temporal_decoding": "full",
        "event_gen": "per-label",
    },
    # Group B: Fusion / calibration
    "B1": {
        "desc": "Uniform model weights",
        "variant": "Uniform model weights",
        "overrides": {"ablation_uniform_model_weights": True},
        "needs_resnet50": False,
        "backbones": "EndoFM+DINOv3",
        "fusion": "weighted",
        "calibration": "T+TT",
        "temporal_decoding": "full",
        "event_gen": "per-label",
    },
    "B2": {
        "desc": "No temperature scaling",
        "variant": "No T-scaling",
        "overrides": {"ablation_fixed_temperature": 1.0},
        "needs_resnet50": False,
        "backbones": "EndoFM+DINOv3",
        "fusion": "weighted",
        "calibration": "TT",
        "temporal_decoding": "full",
        "event_gen": "per-label",
    },
    "B3": {
        "desc": "No TTA",
        "variant": "No TTA",
        "overrides": {"tta": False},
        "needs_resnet50": False,
        "backbones": "EndoFM+DINOv3",
        "fusion": "weighted",
        "calibration": "T",
        "temporal_decoding": "full",
        "event_gen": "per-label",
    },
    # Group C: Temporal decoding
    "C1": {
        "desc": "No structured temporal decoding",
        "variant": "No struct. decoding",
        "overrides": {
            "ablation_no_pp_search": True,
            "ablation_no_smooth": True,
            "ablation_no_enforce_excl": True,
            "ablation_no_enforce_order": True,
            "ablation_no_enforce_landmark": True,
            "ablation_no_morph_pp": True,
            "ablation_no_ensure_region": True,
        },
        "needs_resnet50": False,
        "backbones": "EndoFM+DINOv3",
        "fusion": "weighted",
        "calibration": "T+TT",
        "temporal_decoding": "none",
        "event_gen": "per-label",
    },
    "C2": {"desc": "No smoothing", "variant": "No smooth", "overrides": {"ablation_no_smooth": True},
           "needs_resnet50": False, "backbones": "EndoFM+DINOv3", "fusion": "weighted", "calibration": "T+TT",
           "temporal_decoding": "partial", "event_gen": "per-label"},
    "C3": {"desc": "No anatomical exclusivity", "variant": "No excl", "overrides": {"ablation_no_enforce_excl": True},
           "needs_resnet50": False, "backbones": "EndoFM+DINOv3", "fusion": "weighted", "calibration": "T+TT",
           "temporal_decoding": "partial", "event_gen": "per-label"},
    "C4": {"desc": "No monotonic order", "variant": "No order", "overrides": {"ablation_no_enforce_order": True},
           "needs_resnet50": False, "backbones": "EndoFM+DINOv3", "fusion": "weighted", "calibration": "T+TT",
           "temporal_decoding": "partial", "event_gen": "per-label"},
    "C5": {"desc": "No landmark-region consistency", "variant": "No landmark", "overrides": {"ablation_no_enforce_landmark": True},
           "needs_resnet50": False, "backbones": "EndoFM+DINOv3", "fusion": "weighted", "calibration": "T+TT",
           "temporal_decoding": "partial", "event_gen": "per-label"},
    "C6": {"desc": "No morphology", "variant": "No morph", "overrides": {"ablation_no_morph_pp": True},
           "needs_resnet50": False, "backbones": "EndoFM+DINOv3", "fusion": "weighted", "calibration": "T+TT",
           "temporal_decoding": "partial", "event_gen": "per-label"},
    "C7": {"desc": "No ensure-region", "variant": "No ensure-region", "overrides": {"ablation_no_ensure_region": True},
           "needs_resnet50": False, "backbones": "EndoFM+DINOv3", "fusion": "weighted", "calibration": "T+TT",
           "temporal_decoding": "partial", "event_gen": "per-label"},
    "C8": {
        "desc": "Tuple-based event generation",
        "variant": "Tuple event gen",
        "overrides": {"ablation_tuple_event": True},
        "needs_resnet50": False,
        "backbones": "EndoFM+DINOv3",
        "fusion": "weighted",
        "calibration": "T+TT",
        "temporal_decoding": "full",
        "event_gen": "tuple-based",
    },
}


def merge_config(base: dict, overrides: dict) -> dict:
    out = base.copy()
    for k, v in overrides.items():
        out[k] = v
    return out


def run_single_exp(exp_id: str, config_path: Path, out_dir: Path, timeout: int = 7200) -> dict:
    out_json = out_dir / f"ablation_{exp_id}_pred.json"
    val_metrics_path = out_dir / f"ablation_{exp_id}_pred.val_metrics.json"
    cmd = [sys.executable, str(PROJECT_ROOT / "predict_ensembleV5.py"), "--config", str(config_path), "--out-json", str(out_json)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT_ROOT))
        stdout, stderr = result.stdout, result.stderr
        m05, m95, vanat, vpath = None, None, None, None

        # 1. 優先從 val_metrics.json 讀取（最穩妥）
        if val_metrics_path.exists():
            try:
                with open(val_metrics_path) as f:
                    vm = json.load(f)
                m05 = vm.get("mAP_05")
                m95 = vm.get("mAP_95")
                vanat = vm.get("anatomical_mAP_05")
                vpath = vm.get("pathological_mAP_05")
            except Exception:
                pass

        # 2. 若無檔案，從 stdout 解析
        if m05 is None:
            pat = re.compile(r"Val mAP@0\.5=([\d.]+),\s*mAP@0\.95=([\d.]+)")
            for line in stdout.split("\n"):
                m = pat.search(line)
                if m:
                    m05, m95 = float(m.group(1)), float(m.group(2))
                    break
        if m05 is None:
            for line in stdout.split("\n"):
                if "mAP@0.5=" in line and "Val" in line:
                    try: m05 = float(re.search(r"mAP@0\.5=([\d.]+)", line).group(1))
                    except (AttributeError, ValueError): pass
                if "mAP@0.95=" in line:
                    try: m95 = float(re.search(r"mAP@0\.95=([\d.]+)", line).group(1))
                    except (AttributeError, ValueError): pass
        if vanat is None:
            for line in stdout.split("\n"):
                if m := re.search(r"Anatomical mAP@0\.5=([\d.]+)", line): vanat = float(m.group(1))
                if m := re.search(r"Pathological mAP@0\.5=([\d.]+)", line): vpath = float(m.group(1))

        return {"mAP_05": m05, "mAP_95": m95, "anatomical_mAP_05": vanat, "pathological_mAP_05": vpath,
                "success": result.returncode == 0, "stderr": stderr[:500] if stderr else ""}
    except subprocess.TimeoutExpired:
        # Timeout 時仍嘗試讀取 val_metrics（validation 可能已完成，僅 test 階段超時）
        m05, m95, vanat, vpath = None, None, None, None
        if val_metrics_path.exists():
            try:
                with open(val_metrics_path) as f:
                    vm = json.load(f)
                m05 = vm.get("mAP_05")
                m95 = vm.get("mAP_95")
                vanat = vm.get("anatomical_mAP_05")
                vpath = vm.get("pathological_mAP_05")
            except Exception:
                pass
        return {"mAP_05": m05, "mAP_95": m95, "anatomical_mAP_05": vanat, "pathological_mAP_05": vpath,
                "success": False, "stderr": "Timeout (val metrics may still be valid)"}
    except Exception as e:
        return {"mAP_05": None, "mAP_95": None, "anatomical_mAP_05": None, "pathological_mAP_05": None,
                "success": False, "stderr": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Guideline-compliant ablation study")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--ablations", default="all", help="Comma-separated: A0,A1,A2,... or 'all' or 'priority8'")
    parser.add_argument("--output", default="ablation_guideline_results.json")
    parser.add_argument("--out-dir", default="ablation_guideline_runs")
    parser.add_argument("--skip-resnet50", action="store_true", help="Skip experiments that need ResNet-50")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-experiment timeout in seconds (default 7200=2h, EndoFM test extraction is slow)")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    if not base_path.exists():
        print(f"Base config not found: {base_path}")
        sys.exit(1)

    with open(base_path) as f:
        base_config = json.load(f)

    # 確保 extra_backbones 有 DINOv3
    if "extra_backbones" not in base_config or not base_config["extra_backbones"]:
        base_config["extra_backbones"] = BASE_EXTRA_BACKBONES

    if args.ablations.lower() == "all":
        to_run = list(EXPERIMENTS.keys())
    elif args.ablations.lower() == "priority8":
        to_run = ["A0", "A1", "A2", "A3", "A4", "B1", "C1", "C8"]
    else:
        to_run = [x.strip().upper() for x in args.ablations.split(",") if x.strip()]
        for e in to_run:
            if e not in EXPERIMENTS:
                print(f"Unknown experiment: {e}. Available: {list(EXPERIMENTS.keys())}")
                sys.exit(1)

    if args.skip_resnet50:
        to_run = [e for e in to_run if not EXPERIMENTS[e]["needs_resnet50"]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, exp_id in enumerate(to_run):
        cfg = EXPERIMENTS[exp_id]
        if cfg["needs_resnet50"]:
            r50_feat = PROJECT_ROOT / "features_resnet50" / "val_features.npy"
            r50_ens = PROJECT_ROOT / "ensemble_resnet50" / "manifest.json"
            if not r50_feat.exists() or not r50_ens.exists():
                print(f"\n[{i+1}/{len(to_run)}] {exp_id}: SKIP (ResNet-50 not prepared. Run extract_features_dinov2.py --backbone resnet50 and train_ensemble)")
                results.append({"exp_id": exp_id, **cfg, "mAP_05": None, "mAP_95": None, "anatomical_mAP_05": None, "pathological_mAP_05": None, "success": False, "skipped": True})
                continue

        merged = merge_config(base_config, cfg["overrides"])
        if cfg["needs_resnet50"]:
            merged["extra_backbones"] = [RESNET50_BACKBONE]
        config_path = out_dir / f"ablation_{exp_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(merged, f, indent=2)

        print(f"\n[{i+1}/{len(to_run)}] {exp_id}: {cfg['desc']}")
        r = run_single_exp(exp_id, config_path, out_dir, timeout=args.timeout)
        results.append({
            "exp_id": exp_id,
            "variant": cfg["variant"],
            "desc": cfg["desc"],
            "backbones": cfg["backbones"],
            "fusion": cfg["fusion"],
            "calibration": cfg["calibration"],
            "temporal_decoding": cfg["temporal_decoding"],
            "event_gen": cfg["event_gen"],
            "mAP_05": r["mAP_05"],
            "mAP_95": r["mAP_95"],
            "anatomical_mAP_05": r.get("anatomical_mAP_05"),
            "pathological_mAP_05": r.get("pathological_mAP_05"),
            "success": r["success"],
            "stderr": r.get("stderr", ""),
        })
        status = f"mAP@0.5={r['mAP_05']}, mAP@0.95={r['mAP_95']}" if r["mAP_05"] is not None else "FAILED"
        print(f"  {status}")
        if r["mAP_05"] is None and r.get("stderr"):
            err_file = out_dir / f"ablation_{exp_id}_stderr.txt"
            with open(err_file, "w") as f:
                f.write(r["stderr"])
            print(f"  [stderr 已存至 {err_file}]")

    # Save JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "base_config": str(base_path),
        "metrics_note": "mAP_05/95 = Temporal mAP @ IoU 0.5/0.95. T=temp scaling, TT=TTA.",
        "experiments_run": to_run,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved {args.output}")

    # Ablation table (markdown)
    table_path = Path(args.output).with_suffix(".md")
    with open(table_path, "w") as f:
        f.write("| Exp ID | Variant | Backbone(s) | Fusion | Calibration | Temporal Decoding | Event Gen | mAP@0.5 | mAP@0.95 |\n")
        f.write("|-------|---------|-------------|--------|-------------|-------------------|-----------|---------|----------|\n")
        for r in results:
            m05 = f"{r['mAP_05']:.4f}" if r["mAP_05"] is not None else "N/A"
            m95 = f"{r['mAP_95']:.4f}" if r["mAP_95"] is not None else "N/A"
            f.write(f"| {r['exp_id']} | {r['variant']} | {r['backbones']} | {r['fusion']} | {r['calibration']} | {r['temporal_decoding']} | {r['event_gen']} | {m05} | {m95} |\n")
    print(f"Saved {table_path}")

    # CSV
    csv_path = Path(args.output).with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("exp_id,variant,backbones,fusion,calibration,temporal_decoding,event_gen,mAP_05,mAP_95,anatomical_mAP_05,pathological_mAP_05\n")
        for r in results:
            m05 = f"{r['mAP_05']:.4f}" if r["mAP_05"] is not None else "N/A"
            m95 = f"{r['mAP_95']:.4f}" if r["mAP_95"] is not None else "N/A"
            vanat = f"{r['anatomical_mAP_05']:.4f}" if r.get("anatomical_mAP_05") is not None else "N/A"
            vpath = f"{r['pathological_mAP_05']:.4f}" if r.get("pathological_mAP_05") is not None else "N/A"
            f.write(f"{r['exp_id']},{r['variant']},{r['backbones']},{r['fusion']},{r['calibration']},{r['temporal_decoding']},{r['event_gen']},{m05},{m95},{vanat},{vpath}\n")
    print(f"Saved {csv_path}")

    # Summary
    print("\n" + "=" * 90)
    print("Ablation Table (Guideline-compliant)")
    print("=" * 90)
    for r in results:
        m05 = f"{r['mAP_05']:.4f}" if r["mAP_05"] is not None else "N/A"
        m95 = f"{r['mAP_95']:.4f}" if r["mAP_95"] is not None else "N/A"
        print(f"  {r['exp_id']:4s}  {r['variant']:25s}  mAP@0.5={m05:>8s}  mAP@0.95={m95:>8s}  {r['desc']}")
    print("=" * 90)
    print("Done!")


if __name__ == "__main__":
    main()
