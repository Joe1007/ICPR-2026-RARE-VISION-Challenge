# RARE-VISION 2026 Challenge 方法說明（精簡版）

本文件為 METHOD.md 的精簡版，保留核心架構與重點，供快速查閱。詳細說明請見 METHOD.md。

---

## 一、競賽與資料

- **任務**：內視鏡影片 frame-level 多標籤分類，輸出 events JSON（`start`、`end`、`label`）
- **評估**：Temporal mAP @ 0.5 與 @ 0.95
- **標籤**：17 類（解剖 5、地標 3、病變 9）
- **資料**：訓練集 `galar_dataset/downloads/`，測試集 `Testingset/`（ukdd_navi_* → vid_001/002/003）

---

## 二、V5 管線總覽（四步驟）

```text
Step 1: 驗證集分析
   → 各 backbone 各模型預測 → per-class weighted ensemble → cross-backbone weighting
   → temperature scaling → 搜尋 pp 參數 → F1 閾值 → temporal mAP 閾值
   ↓ 產出：weights, temperature, thresholds, pp_params

Step 2: 測試集特徵提取
   → EndoFM：原圖 clip + 水平翻轉 clip（TTA）

Step 3: 多 backbone 融合預測
   → EndoFM：5 模型 weighted avg → 兩組 TTA 平均
   → DINOv3：test_features.npy → 5 模型 weighted avg
   → cross-backbone weighted average → temperature scaling

Step 4: 後處理 + 事件輸出
   → smoothing → 解剖互斥 → threshold → morphological → 解剖順序 → 地標約束 → 補區域
   → per-label 獨立事件轉換 → video ID 映射 → JSON
```

---

## 三、多 Backbone

| Backbone | 維度 | 特徵方式 | 特點 |
|----------|------|----------|------|
| **EndoFM-LV** | 768 | 4 幀 clip（TimeSformer） | 內視鏡 domain、短距時序 |
| **DINOv3 ViT-L/16** | 1024 | 單幀 CLS | 通用結構、語意理解 |

兩者互補，融合後覆蓋更完整視覺資訊。

---

## 四、Ensemble 與權重

- **Per-backbone**：每 backbone 5 個分類頭（Linear/MLP × BCE/Focal/Asymmetric）
- **Per-class model weighting**：驗證集上依各類別 AP 分配模型權重
- **Cross-backbone weighting**：依各 backbone 的 frame-level mAP 分配權重

---

## 五、關鍵技術

- **TTA**：EndoFM 做水平翻轉，兩組預測平均（DINOv3 用預提取特徵，不做 TTA）
- **Temperature Scaling**：驗證集 grid search T，最佳 frame mAP
- **後處理參數**：6 個參數（wa, wp, min_a, min_p, gap_a, gap_p）coordinate descent 搜尋
- **閾值**：兩階段 — F1-based → temporal mAP-optimized（±0.15 範圍）

---

## 六、後處理管線（7 步）

1. Temporal smoothing（median filter，解剖大窗口、病理小窗口）
2. 解剖區域互斥（5 區域取 argmax）
3. Per-class threshold
4. Morphological opening/closing
5. 解剖順序約束（mouth → colon 不可倒退）
6. 地標-區域關聯約束（50 幀內需有對應區域）
7. 確保每幀有解剖區域

---

## 七、事件轉換

**Per-label 獨立**：每個標籤獨立遍歷時序找連續 1 區段，避免解剖變化切斷病理事件。輸出格式：`{"start": 100, "end": 500, "label": "active bleeding"}`。

---

## 八、執行流程

```bash
# 1. 特徵提取
python extract_features.py --config extract_features.args.json
python extract_features_dinov2.py --backbone dinov3_vitl16 --output-dir features_dinov3 --test-root ./Testingset

# 2. Ensemble 訓練
python train_ensemble.py --features-dir features --ensemble-dir ensemble
python train_ensembleV2.py --features-dir features_dinov3 --ensemble-dir ensemble_dinov3

# 3. V5 推論
python predict_ensembleV5.py --config predict_ensembleV5.args.json
```

輸出：`test_v5_pred.json`，可直接上傳比賽評分器。

---

## 九、主要腳本

| 腳本 | 用途 |
|------|------|
| predict_ensembleV5.py | V5 主推論腳本 |
| extract_features.py | EndoFM 特徵（train/val） |
| extract_features_dinov2.py | DINOv3 特徵（train/val/test） |
| train_ensemble.py | EndoFM ensemble 訓練 |
| train_ensembleV2.py | DINOv3 ensemble 訓練 |

---

## 十、依賴

- EndoFM-LV、DINOv3（timm）、PyTorch、torchvision、pandas、scikit-learn、scipy
