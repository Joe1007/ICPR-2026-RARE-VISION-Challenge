# ICPR-2026-RARE-VISION-Challenge

This repository contains the solution of **ACVLab** for the **ICPR 2026 RARE VISION Challenge**.  
Our approach is implemented in a set of scripts that build features, train lightweight multi-label classifiers, and perform **multi-backbone ensemble inference** to decode **temporal events**.

---

## V5 Output

The main output of the V5 pipeline is:

- `test_v5_pred.json` (a events JSON file in the format expected by the challenge evaluator)

---

## Repository Overview (Where to Put Files)

After cloning, run everything from the repo root. The expected directory layout is:

```text
ICPR-2026-RARE-VISION-Challenge/
├─ predict_ensembleV5.py
├─ predict_ensembleV5.args.json
├─ extract_features.py
├─ extract_features_dinov2.py
├─ train_ensemble.py
├─ train_from_features.py
├─ convert_to_submission.py
├─ rarevision_v5_pipeline.txt
├─ EndoFM-LV/                      (code only; weights are downloaded separately)
│  ├─ models/
│  ├─ utils/
│  └─ ...
├─ Testingset/                    (dataset frames + CSV metadata; not included here)
├─ features/                     (generated: pre-extracted features for EndoFM split)
│  ├─ config.json
│  ├─ train_features.npy
│  ├─ val_features.npy
│  ├─ train_labels.npy
│  ├─ val_labels.npy
│  ├─ train_meta.csv
│  └─ val_meta.csv
├─ ensemble/                     (generated: EndoFM classifier ensemble weights + manifest)
│  ├─ manifest.json
│  └─ model_*.pth
├─ features_dinov3/             (generated: DINOv3 test-time features for the same split)
│  ├─ config.json
│  ├─ val_features.npy
│  ├─ val_labels.npy
│  ├─ val_meta.csv
│  ├─ train_features.npy
│  ├─ train_labels.npy
│  └─ test_features.npy
├─ ensemble_dinov3/            (generated: DINOv3 classifier ensemble weights + manifest)
│  ├─ manifest.json
│  └─ model_*.pth
└─ (other small scripts/docs)

> **Note on `EndoFM-LV/`**
> We do not commit the EndoFM-LV checkpoint weights inside `EndoFM-LV/checkpoints/`. Download the pretrained checkpoint separately and place it as instructed below.

> **Note on Dataset**
> The dataset is large. We do not include it in this repository.
> Please obtain it separately and place it under `Testingset/`.

---

## (A) From Scratch: Generate `test_v5_pred.json`

### Step A0: Download Required Assets (Weights / Optional Precomputed Assets)
You will need the EndoFM-LV checkpoint and, depending on your workflow, may also provide precomputed features/ensembles.

**Download link placeholders (to be filled by you):**
* **EndoFM-LV checkpoint** (place under `EndoFM-LV/checkpoints/`): 
  `[DOWNLOAD LINK PLACEHOLDER]`
* **Optional:** precomputed `features/`, `features_dinov3/`, `ensemble/`, `ensemble_dinov3/` bundles: 
  `[DOWNLOAD LINK PLACEHOLDER]`

**Expected EndoFM checkpoint location:**
Place the checkpoint here:
`EndoFM-LV/checkpoints/endofm_lv.pth`
*(If your file name differs, update `predict_ensembleV5.args.json` accordingly.)*

### Step A1: Extract EndoFM-LV Features (Train/Val Split)
Run:
```bash
python extract_features.py --config extract_features.args.json
```
This generates the `features/` directory (train/val `.npy`, labels, and meta CSV).

### Step A2: Train EndoFM Classifier Ensemble
Run:
```bash
python train_ensemble.py --config train_ensemble.args.json
```
This generates:
* `ensemble/manifest.json`
* `ensemble/model_*.pth`

### Step A3: Extract DINOv3 Features (Val + Test)
Run:
```bash
python extract_features_dinov2.py \
  --backbone dinov3_vitl16 \
  --endofm-features-dir features \
  --output-dir features_dinov3 \
  --test-root ./Testingset
```
This generates:
* `features_dinov3/val_features.npy` (and matching val labels/meta)
* `features_dinov3/test_features.npy`

### Step A4: Train DINOv3 Classifier Ensemble
Run:
```bash
python train_ensemble.py \
  --features-dir features_dinov3 \
  --ensemble-dir ensemble_dinov3
```
*(Alternatively, provide a config JSON to control the ensemble training settings.)*

### Step A5: Run V5 Inference (Produces `test_v5_pred.json`)
Run:
```bash
python predict_ensembleV5.py --config predict_ensembleV5.args.json
```
Outputs:
* `test_v5_pred.json`
* `test_v5_pred.config.json`
* `test_v5_pred.val_metrics.json` (generated during validation-guided parameter selection)

### Step A6 (Optional): Convert to Submission Format
If needed by the evaluator:
```bash
python convert_to_submission.py --input test_v5_pred.json --output test_trans_pred.json
```

---

## (B) Quick Run: Generate `test_v5_pred.json` Using Precomputed Assets

If you already have precomputed `features/`, `ensemble/`, `features_dinov3/`, and `ensemble_dinov3/`, you can run inference directly.

### Quick steps
Ensure you have:
* `Testingset/` with `*.csv` and `frame_*.png` folders
* `EndoFM-LV/checkpoints/endofm_lv.pth`
* `features/`
* `ensemble/`
* `features_dinov3/`
* `ensemble_dinov3/`

Run:
```bash
python predict_ensembleV5.py --config predict_ensembleV5.args.json
```
This will generate `test_v5_pred.json`.

### Quick Directory Placement Checklist
Precomputed artifacts should be placed into:
* **EndoFM checkpoint:** `EndoFM-LV/checkpoints/endofm_lv.pth`
* **EndoFM features/ensemble:** `features/`, `ensemble/`
* **DINOv3 features/ensemble:** `features_dinov3/`, `ensemble_dinov3/`
* **Dataset:** `Testingset/`

We recommend downloading missing large assets from:
`[DOWNLOAD LINK PLACEHOLDER]` (weights and/or precomputed numpy/ensemble bundles)

---

## Notes
* **The dataset is large:** we do not commit `Testingset/` or `galar_dataset/` in this repository.
* We expect large weights/features to be downloaded separately.
* All scripts are intended to be run from the repository root.

