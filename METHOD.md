# RARE-VISION 2026 Challenge 方法說明

本文件說明本專案針對 RARE-VISION 2026 競賽所採用的資料集結構、方法流程與程式碼架構。

---

## 一、競賽與資料集結構

### 1.1 競賽任務

- **任務類型**：內視鏡影片的 frame-level 多標籤分類
- **輸出格式**：將 frame-level 預測轉換為 **事件（events）** JSON，每個事件包含 `start`、`end`、`label`
- **評估指標**：Temporal mAP @ 0.5 與 @ 0.95

### 1.2 訓練資料集（galar_dataset）

路徑：`galar_dataset/downloads/`

```
galar_dataset/downloads/
├── 1/                    # 影片 1 的幀圖檔
│   ├── frame_000105.PNG
│   ├── frame_000110.PNG
│   └── ...
├── 2/
├── ...
├── 80/
├── Labels/
│   ├── 1.csv             # 影片 1 的 frame-level 標註
│   ├── 2.csv
│   └── ...
└── metadata.csv          # 影片層級 metadata
```

**影像幀**：
- 資料夾名稱 = 影片編號（recording ID）
- 檔名格式：`frame_XXXXXX.PNG`，`XXXXXX` 為 6 位數幀號

**標註 CSV**：
- 每支影片一個 CSV，檔名為影片編號
- 重要欄位：`index`、`frame`（對應影像幀號）、以及多個標籤欄位（0/1）

**影像與標註對應**：
```
影像路徑 = galar_dataset/downloads/<recording>/frame_<frame:06d>.PNG
```

### 1.3 測試資料集（Testingset）

路徑：`Testingset/`

```
Testingset/
├── ukdd_navi_00051/           # 影片 1 的幀圖檔
│   ├── frame_0000049.png
│   ├── frame_0000050.png
│   └── ...
├── ukdd_navi_00051.csv       # 影片 1 的 frame 列表（sep=";"）
├── ukdd_navi_00068/
├── ukdd_navi_00068.csv
├── ukdd_navi_00076/
└── ukdd_navi_00076.csv
```

**與訓練集差異**：
- 檔名為小寫 `.png`（訓練集為 `.PNG`）
- CSV 使用 `;` 分隔，欄位 `frame_file` 格式為 `frame_0000049.png`
- 無標註（測試用）
- 影片 ID 為 `ukdd_navi_00051` 等，比賽評分器要求轉為 `vid_001`、`vid_002`、`vid_003`

### 1.4 標籤定義（USED_LABELS）

共 17 個標籤，分為解剖結構與病變：

| 類別 | 標籤 |
|------|------|
| 解剖結構 | mouth, esophagus, stomach, small intestine, colon |
| 地標 | z-line, pylorus, ileocecal valve |
| 病變 | active bleeding, angiectasia, blood, erosion, erythema, hematin, lymphangioectasis, polyp, ulcer |

### 1.5 預測 JSON 格式

比賽要求輸出格式（V5 採用 per-label 獨立事件）：

```json
{
  "videos": [
    {
      "video_id": "vid_001",
      "events": [
        {"start": 100, "end": 500, "label": "active bleeding"},
        {"start": 501, "end": 1000, "label": "stomach"}
      ]
    }
  ]
}
```

- `video_id`：必須與評分器預期的 ID 一致（`vid_001`、`vid_002`、`vid_003`）
- `events`：每個事件只有一個標籤字串，與官方範例格式一致

### 1.6 評估指標

- **Temporal mAP @ 0.5**：IoU 閾值 0.5 下的 mean Average Precision
- **Temporal mAP @ 0.95**：IoU 閾值 0.95 下的 mean Average Precision
- 計算方式：將 frame-level 預測轉為 events，與 ground truth events 依 Temporal IoU 配對，計算各類別 AP 後取平均

---

## 二、V5 方法流程總覽

V5 是一個**多 backbone 融合**的推論管線，不修改任何已訓練的模型，只在推論階段整合多個來源的預測並做完整的後處理。整個管線分成四大步驟：

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 驗證集分析                                        │
│   載入 val_features + val_labels                          │
│   → 每個 backbone 的每個模型對驗證集做預測                    │
│   → 計算 per-class weighted ensemble                      │
│   → cross-backbone weighting                             │
│   → temperature scaling                                  │
│   → 搜尋 post-processing 最佳參數                          │
│   → F1-based threshold → temporal mAP-optimized threshold │
└─────────────────────────┬───────────────────────────────┘
                          ↓ 產出：weights, temperature,
                            thresholds, pp_params
┌─────────────────────────────────────────────────────────┐
│ Step 2: 測試集特徵提取                                     │
│   EndoFM backbone 載入                                    │
│   → 原圖 clip 提取特徵 features_normal                     │
│   → 水平翻轉 clip 提取特徵 features_flip (TTA)              │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 3: 多 backbone 融合預測                               │
│   EndoFM: normal + flip 各過 5 模型 → weighted avg → 再平均  │
│   DINOv3: 載入 test_features.npy → 過 5 模型 → weighted avg │
│   → cross-backbone weighted average                      │
│   → temperature scaling                                  │
└─────────────────────────┬───────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Step 4: 後處理 + 事件輸出                                   │
│   分類別 temporal smoothing                                │
│   → 解剖區域互斥                                           │
│   → per-class threshold                                   │
│   → morphological opening/closing                         │
│   → 解剖順序約束                                           │
│   → 地標-區域關聯約束                                       │
│   → 確保每幀有解剖區域                                      │
│   → per-label 獨立事件轉換                                  │
│   → video ID 映射 → JSON                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 三、多 Backbone 特徵

### 3.1 EndoFM-LV（768 維）

EndoFM-LV 是一個內視鏡影片專用的 Video Transformer（TimeSformer 架構，ViT-B/16）。它用 teacher-student 自蒸餾方式在大量內視鏡影片上預訓練，訓練目標包含 masked token matching、cross-view matching 和 dynamic motion matching。

**特徵提取方式**：clip-based。以每一幀為中心，取前後各 1 幀（共 4 幀，stride=2），組成一個 clip 輸入 TimeSformer，取出 CLS token 作為該幀的 768 維特徵。每幀的特徵已經編碼了短距離的時序資訊（約 8 幀的感受野）。

### 3.2 DINOv3 ViT-L/16（1024 維）

DINOv3 是 Meta 在 2025 年 8 月發佈的最新視覺基礎模型，用純自監督學習在 17 億張圖上訓練，模型最大到 7B 參數。V5 使用的是 ViT-L/16 版本（約 300M 參數），透過 timm 載入。

**特徵提取方式**：single-frame。每幀獨立輸入 DINOv3，取 CLS token 得到 1024 維特徵。沒有時序資訊，但特徵品質極高，尤其在結構理解和語意分割上遠超 DINOv2。DINOv3 的關鍵創新 Gram Anchoring 解決了長時間訓練後 dense feature 退化的問題。

### 3.3 兩個 backbone 的互補性

EndoFM-LV 擅長內視鏡 domain 的特定紋理和短距時序關係（例如蠕動、氣泡變化），DINOv3 擅長通用的結構和語意理解（例如器官形態、病灶邊界）。兩者在特徵空間上幾乎沒有重疊，融合後能覆蓋更完整的視覺資訊。

---

## 四、Ensemble 結構

### 4.1 Per-backbone Ensemble

每個 backbone 上各訓練了 5 個分類頭，使用不同的 loss function 和模型架構組合：

| # | 模型 | Loss | Dropout |
|---|------|------|---------|
| 0 | Linear | BCE + pos_weight | 0.3 |
| 1 | Linear | Focal Loss | 0.3 |
| 2 | Linear | Asymmetric Loss | 0.3 |
| 3 | MLP (2層) | BCE + pos_weight | 0.3 |
| 4 | MLP (2層) | Focal Loss | 0.3 |

每個模型輸出 logits → sigmoid 得到 17 維機率向量。5 個模型的機率不是簡單平均，而是用 **per-class weighted average**。

### 4.2 Per-class Model Weighting

在驗證集上，計算每個模型在每個類別上的 Average Precision（AP）。AP 越高的模型在該類別上權重越大。

具體做法：對第 m 個模型和第 c 個類別，計算 `w[m,c] = AP(val_labels[:,c], model_m_probs[:,c])`。然後對每個類別做 L1 歸一化：`w[:,c] /= sum(w[:,c])`。

不同 loss function 對不同類別的表現差異很大。例如 Asymmetric Loss 對稀有病理類別通常 recall 更好，而 BCE + pos_weight 對常見解剖類別更穩定。Per-class weighting 讓每個類別自動選擇最適合它的模型組合。

### 4.3 Cross-backbone Weighting

兩個 backbone 的 ensemble 預測也不是簡單平均。在驗證集上，計算每個 backbone 的 frame-level mAP（17 個類別 AP 的平均），作為 backbone 層級的權重。

例如若 EndoFM 的 frame mAP = 0.45，DINOv3 的 frame mAP = 0.52，則權重分別是 0.45/0.97 ≈ 0.464 和 0.52/0.97 ≈ 0.536。

最終的 ensemble 機率：`probs_final = w_endofm * probs_endofm + w_dinov3 * probs_dinov3`

---

## 五、Test-Time Augmentation (TTA)

對 EndoFM backbone 的測試集特徵提取，使用兩種 transform：

| 類型 | Transform |
|------|------------|
| **Normal** | Resize(224) → ToTensor → Normalize |
| **Flip** | Resize(224) → HorizontalFlip(p=1.0) → ToTensor → Normalize |

兩組特徵分別通過 5 個 ensemble 模型得到兩組機率預測，最後取平均。

水平翻轉之所以有效，是因為膠囊鏡頭可能在任意方向旋轉，翻轉後的圖片在醫學上同樣合理。兩組預測平均可以減少模型對特定方向的偏向，提升預測穩定性。

DINOv3 沒有做 TTA，因為它的特徵是預提取的 `.npy` 檔案（test_features.npy），重新提取需要額外的 DINOv3 backbone 推論，時間成本較高。

---

## 六、Temperature Scaling

Ensemble 平均後的機率分佈可能不是最佳校準的。Temperature Scaling 用一個標量 T 來調整：

```
logits = log(p / (1-p))          # 從機率反推 logits
scaled_logits = logits / T        # 除以溫度
calibrated_probs = sigmoid(scaled_logits)
```

- **T > 1**：讓機率更接近 0.5（降低過度自信）
- **T < 1**：讓機率更 peaked（增加自信度）

在驗證集上 grid search `T ∈ {0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0}`，選擇使 frame-level mAP 最高的 T。

---

## 七、後處理參數自動搜尋

後處理有 6 個參數，V5 不用手寫固定值，而是在驗證集上逐參數 grid search。

搜尋策略是 **coordinate descent**（逐參數優化）：固定其他參數，對當前參數嘗試所有候選值，選 temporal mAP@0.5 最高的。然後移到下一個參數，依此類推。

| 參數 | 候選值 | 用途 |
|------|--------|------|
| wa (smooth window anatomical) | 9, 15, 21, 31 | 解剖標籤的 smoothing 窗口 |
| wp (smooth window pathological) | 3, 5, 7, 11 | 病理標籤的 smoothing 窗口 |
| min_a (min event anatomical) | 5, 10, 15, 20 | 解剖事件最短長度 |
| min_p (min event pathological) | 2, 3, 5 | 病理事件最短長度 |
| gap_a (fill gap anatomical) | 3, 5, 8, 12 | 解剖事件的最大填補間隙 |
| gap_p (fill gap pathological) | 1, 2, 3, 5 | 病理事件的最大填補間隙 |

---

## 八、閾值搜尋（兩階段）

### 8.1 第一階段：F1-based threshold

對驗證集上 smoothed + exclusivity 處理後的機率，用 sklearn 的 `precision_recall_curve` 為每個類別找最佳 F1 閾值。

`F1 = 2 * precision * recall / (precision + recall)`

每個類別獨立搜尋。如果某個類別在驗證集中沒有正樣本，預設閾值為 0.5。

### 8.2 第二階段：Temporal mAP-optimized threshold

F1 最佳閾值不一定是 temporal mAP 最佳閾值。例如降低某個稀有類別的閾值可能讓 F1 下降（因為 FP 增多），但如果多偵測到一個完整的事件且 IoU ≥ 0.5，temporal mAP 反而提升。

**做法**：以 F1 閾值為基礎，在 ±0.15 範圍內以 9 個等距點逐類別搜尋。每次修改一個類別的閾值，跑完整的後處理管線和事件轉換，計算 temporal mAP@0.5。保留最好的值，然後移到下一個類別。

---

## 九、後處理管線（7 步）

| Step | 名稱 | 說明 |
|------|------|------|
| 1 | 分類別 Temporal Smoothing | 對每支影片獨立做 median filter，解剖用大窗口（15-31 幀），病理用小窗口（3-11 幀）。保留 sharp edge，去除 impulse noise。 |
| 2 | 解剖區域互斥 | 5 個主要區域（mouth, esophagus, stomach, small intestine, colon）取 argmax，每幀只保留機率最高的區域。地標不受影響。 |
| 3 | Per-class Threshold | 套用搜尋到的 per-class 最佳閾值。 |
| 4 | Morphological Opening / Closing | Opening 移除短事件碎片；Closing 填補短間隙，合併被噪音切斷的同一事件。解剖用較大 structuring element，病理用較小。 |
| 5 | 解剖順序約束 | 膠囊只能前進：mouth → esophagus → stomach → small intestine → colon。追蹤 max_reached_order，若預測倒退則強制改為已到達的最遠區域。 |
| 6 | 地標-區域關聯約束 | z-line 只能出現在 esophagus/stomach 附近 50 幀；pylorus 在 stomach/small intestine 附近；ileocecal valve 在 small intestine/colon 附近。不符則移除地標。 |
| 7 | 確保每幀有解剖區域 | 若某幀所有解剖區域被清掉，用 smoothed probabilities 的 argmax 補上最可能的區域。 |

---

## 十、事件轉換（Per-label 獨立）

這是 V2 最關鍵的修正，一直沿用到 V5。

**舊版做法**：把每幀的 17 個標籤組合成一個 tuple，相鄰幀的 tuple 不同就切斷事件。問題是解剖標籤的變化會切斷病理事件。例如 bleeding 持續 frame 100-500，但 frame 300 從 stomach 變成 small intestine，bleeding 就被切成 (100-299) 和 (300-500) 兩個事件。若 ground truth 的 bleeding 是完整的 (100-500)，兩個短事件的 IoU 都可能 < 0.5，導致 0 個 TP。

**V5 做法**：每個標籤獨立遍歷時序，找連續的 1 區段，各自產生事件。Bleeding 的事件不受解剖標籤變化影響，完整保留 (100-500)。

輸出格式：`{"start": 100, "end": 500, "label": "active bleeding"}`，每個事件只有一個標籤字串，與官方範例格式一致。

---

## 十一、Video ID 映射

測試集的影片名稱是 `ukdd_navi_00051` 等，但比賽評分器要求 `vid_001` 等格式。在最後輸出 JSON 時做映射：

| 原始 ID | 比賽格式 |
|---------|----------|
| ukdd_navi_00051 | vid_001 |
| ukdd_navi_00068 | vid_002 |
| ukdd_navi_00076 | vid_003 |

---

## 十二、程式碼說明

### 12.1 predict_ensembleV5.py — V5 主推論腳本

**功能**：多 backbone 融合推論，整合 EndoFM-LV 與 DINOv3（或更多 backbone），執行完整的四步驟管線。

**流程**：見「二、V5 方法流程總覽」。所有後處理參數、閾值、權重均在驗證集上自動搜尋，無需手動設定。

**輸出**：
- `test_v5_pred.json`：比賽格式的預測結果
- `test_v5_pred.config.json`：本次推論使用的 thresholds、pp 參數、temperature、backbone_weights

---

### 12.2 前置腳本（V5 所需）

| 腳本 | 用途 |
|------|------|
| extract_features.py | EndoFM 特徵：galar_dataset → features/（train/val） |
| extract_features_dinov2.py | DINOv2/DINOv3 特徵：含 train/val/test，輸出至 features_dinov3/ 等 |
| train_ensemble.py | 在 EndoFM 特徵上訓練 5 模型 → ensemble/ |
| train_ensembleV2.py 或類似 | 在 DINOv3 特徵上訓練 5 模型 → ensemble_dinov3/ |

---

### 12.3 extract_features.py — EndoFM 特徵預提取

**功能**：使用 EndoFM-LV backbone 對訓練/驗證集提取特徵，供 train_ensemble 使用。

**輸出**：`features/` 含 `train_features.npy`、`val_features.npy`、`val_labels.npy`、`val_meta.csv`、`config.json` 等。

---

### 12.4 extract_features_dinov2.py — DINOv2/DINOv3 特徵預提取

**功能**：使用 timm 載入 DINOv2 或 DINOv3，對 train/val/test 提取特徵。V5 需要 `test_features.npy` 供測試集推論。

**輸出**：`features_dinov3/` 含 `val_features.npy`、`test_features.npy`、`config.json` 等。

---

### 12.5 train_ensemble.py — 多模型 Ensemble 訓練

**功能**：一次訓練 5 個分類器（不同 loss、model），儲存至 `ensemble/` 或 `ensemble_dinov3/`。每個 backbone 各有一組 ensemble。

---

### 12.6 convert_to_submission.py — Video ID 轉換工具

**功能**：將既有 JSON 的 video_id 轉為比賽格式，無需重新推論。V5 已在 predict_ensembleV5.py 內建此映射。

---

## 十三、執行流程（V5）

### 13.1 環境

```bash
conda activate endofm-lv
```

### 13.2 完整流程（V5 多 backbone 融合）

```bash
# 1. 提取 EndoFM 特徵（train/val）
python extract_features.py --config extract_features.args.json

# 2. 提取 DINOv3 特徵（train/val/test）
python extract_features_dinov2.py --backbone dinov3_vitl16 \
    --output-dir features_dinov3 --test-root ./Testingset

# 3. 在 EndoFM 特徵上訓練 ensemble
python train_ensemble.py --features-dir features --ensemble-dir ensemble

# 4. 在 DINOv3 特徵上訓練 ensemble
python train_ensembleV2.py --features-dir features_dinov3 --ensemble-dir ensemble_dinov3

# 5. V5 多 backbone 融合推論
python predict_ensembleV5.py --config predict_ensembleV5.args.json
```

輸出：`test_v5_pred.json`，可直接上傳比賽評分器。

### 13.3 指定 GPU

```bash
CUDA_VISIBLE_DEVICES=0 python predict_ensembleV5.py --config predict_ensembleV5.args.json
```

---

## 十四、參數設定檔（.args.json）

V5 主要使用：

| 檔案 | 用途 |
|------|------|
| predict_ensembleV5.args.json | V5 推論：test_root、endofm/dinov3 路徑、pretrained_weights、extra_backbones 等 |

---

## 十五、predict_ensembleV5.args.json 參數對照

| 參數 | 說明 | 預設 |
|------|------|------|
| test_root | 測試集路徑 | ./Testingset |
| endofm_features_dir | EndoFM 特徵目錄 | features |
| endofm_ensemble_dir | EndoFM ensemble 目錄 | ensemble |
| pretrained_weights | EndoFM-LV checkpoint 路徑 | EndoFM-LV/checkpoints/endofm_lv.pth |
| extra_backbones | 額外 backbone 列表（JSON） | dinov3_vitl16 等 |
| tta | 是否對 EndoFM 做水平翻轉 TTA | true |
| out_json | 輸出 JSON 檔名 | test_v5_pred.json |

---

## 十六、依賴與參考

- **EndoFM-LV**：內視鏡影片預訓練 foundation model（TimeSformer backbone）
- **DINOv3**：透過 timm 載入 ViT-L/16
- **PyTorch**、**torchvision**、**pandas**、**scikit-learn**、**scipy**
- 比賽評分器：`sample_codes/scoring.py`（Temporal mAP 計算邏輯，`tiou` 函數計算 Temporal IoU）
