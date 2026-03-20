# 超參數選擇機制記錄

本文件記錄專案中各腳本使用的超參數選擇（搜尋、優化）機制。

---

## Ablation 主要組件（run_ablation_v5.py）

| 組件名稱 | 內容 | Ablation 名稱 |
|----------|------|---------------|
| **Backbone 拆解** | 各 backbone 單獨效果 | endofm_only、dinov3_only |
| **tta** | 測試時水平翻轉增強 | no_tta：關閉 |
| **weighted_ensemble** | Per-model 與 cross-backbone 皆 AP-weighted | uniform_ensemble：均勻權重 |
| **temperature_calibration** | Temperature scaling 校準機率 | no_calibration：fixed 1.0 |
| **advanced_postprocessing** | smooth + enforce + morph + param search | minimal_pp：全部關閉 |

預設不跑 full（baseline），只跑上述 6 個 ablation。

---

## 一、predict_ensembleV5.py（推論管線）

### 1.1 Per-model 權重（compute_weights）

- **機制**：依 validation 上每個模型、每個類別的 **frame-level Average Precision** 作為權重
- **公式**：`w[m,c] = max(AP(gt[:,c], pred[m][:,c]), 0.01)`，再對每個類別做 L1 正規化
- **用途**：同一 backbone 內多個模型的加權平均
- **程式位置**：`compute_weights(all_p, gt)`

### 1.2 Cross-backbone 權重（backbone_weights）

- **機制**：依 validation 上每個 backbone 的 **frame-level mAP**（跨類別平均）作為權重
- **公式**：`backbone_w[bi] = mean(AP(gt[:,c], bp[:,c]) for c in positive_classes)`
- **用途**：多 backbone 融合時的權重
- **程式位置**：main() 中 `backbone_w` 計算

### 1.3 Temperature Scaling（find_temperature）

- **機制**：Grid search 在 validation 上找最佳 temperature
- **搜尋空間**：`[0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]`
- **目標**：最大化 frame-level mean AP（跨類別）
- **公式**：`p' = sigmoid(logit(p) / T)`
- **程式位置**：`find_temperature(p, gt)`

### 1.4 Per-class 閾值（find_thresh_f1）

- **機制**：依 validation 上每個類別的 precision-recall curve，找 **F1 最大** 的閾值
- **搜尋**：遍歷 PR curve 上所有閾值
- **程式位置**：`find_thresh_f1(gt, pr)`

### 1.5 Per-class 閾值（find_thresh_tmap）

- **機制**：在 F1 閾值附近做 **local search**，以 **Temporal mAP@0.5** 為目標
- **搜尋空間**：`linspace(max(0.05, base-0.15), min(0.95, base+0.15), 9)` 每類別
- **程式位置**：`find_thresh_tmap(sv, gv, fv, base_th, pp)`

### 1.6 後處理參數搜尋（search_pp）

- **機制**：Grid search 每個參數，以 **Temporal mAP@0.5** 為目標
- **搜尋空間**：
  - `wa` (smooth window anatomical): [9, 15, 21, 31]
  - `wp` (smooth window pathological): [3, 5, 7, 11]
  - `min_a` (min event frames anatomical): [5, 10, 15, 20]
  - `min_p` (min event frames pathological): [2, 3, 5]
  - `gap_a` (fill gap anatomical): [3, 5, 8, 12]
  - `gap_p` (fill gap pathological): [1, 2, 3, 5]
- **策略**：逐參數 greedy（每次只改一個參數，取最佳）
- **程式位置**：`search_pp(sv, gv, fv, th)`

---

## 二、train_from_features.py / train_ensemble.py（訓練）

### 2.1 Pos_weight（compute_pos_weight）

- **機制**：依訓練集類別不平衡計算 `pos_weight = (1 - pos_ratio) / pos_ratio`
- **用途**：BCEWithLogitsLoss 的 `pos_weight` 參數
- **程式位置**：`compute_pos_weight(train_labels)`

### 2.2 Weighted Sampling（可選）

- **機制**：`--weighted-sampling` 時，依樣本正類數量做反比加權採樣
- **用途**：平衡稀有類別
- **程式位置**：`WeightedRandomSampler` / `compute_sample_weights`

---

## 三、train_from_featuresV2.py（進階訓練）

### 3.1 Sampling 策略

- **選項**：`random` / `weighted` / `class_balanced`
- **class_balanced**：每個 batch 從每個類別各採若干樣本，確保稀有類出現
- **程式位置**：`ClassBalancedSampler`

### 3.2 Early Stopping

- **機制**：以 validation **Temporal mAP** 為監控指標，patience 內無提升則停止
- **程式位置**：訓練迴圈中的 early stopping 邏輯

---

## 四、train_v3.py（時序模型訓練）

### 4.1 與 train_from_featuresV2 類似

- Per-class threshold、early stopping on mAP、class-balanced sampling 等
- 額外：Temporal 模型（1D TCN）的 kernel_size、num_layers 等為固定或 config 指定

---

## 五、train_ensemble.py（多模型訓練）

### 5.1 預設 Ensemble 配置

- **固定組合**：5 個模型
  - linear + bce + dropout 0.3
  - linear + focal + dropout 0.3
  - linear + asymmetric + dropout 0.3
  - mlp + bce + dropout 0.3
  - mlp + focal + dropout 0.3
- **無自動搜尋**：配置為手動指定
- **程式位置**：`DEFAULT_ENSEMBLE_CONFIGS`

---

## 六、摘要表

| 組件 | 選擇機制 | 目標指標 | 搜尋方式 |
|------|----------|----------|----------|
| Per-model weights | AP-based | frame mAP | 封閉解 |
| Backbone weights | AP-based | frame mAP | 封閉解 |
| Temperature | Grid search | frame mAP | 9 點 |
| Per-class threshold (F1) | PR curve | F1 | 遍歷 |
| Per-class threshold (mAP) | Local search | Temporal mAP@0.5 | 9 點/類 |
| PP params | Greedy grid | Temporal mAP@0.5 | 逐參數 |
| Pos weight | 類別比例 | - | 封閉解 |
| Ensemble config | 手動 | - | 固定 |
