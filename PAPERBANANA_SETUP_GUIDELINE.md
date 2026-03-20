# PaperBanana 安裝指南（Server91）

本指南根據你在 server126 的安裝經驗整理，適用於**從零開始**在 server91 安裝 PaperBanana。

---

## 快速開始（依序執行）

```bash
# 1. 安裝 paperbanana（選 --user 若權限不足）
pip install paperbanana
# pip install --user paperbanana

# 2. 驗證
paperbanana --help

# 3. 設定 API key（會開瀏覽器，或手動建 .env）
paperbanana setup

# 4. 在專案目錄建 .env（若 setup 未完成）
echo "GOOGLE_API_KEY=你的金鑰" > /ssd8/cheng/RARE-VISION-2026-Challenge/.env

# 5. 測試生成
cd /ssd8/cheng/RARE-VISION-2026-Challenge
echo "Pipeline: A → B → C" > test.txt
paperbanana generate --input test.txt --caption "Test" --output test.png --iterations 1
```

**MCP 使用者**：還需安裝 uv、建立 `~/.cursor/mcp.json`，見下方詳細步驟。

---

## 一、目前狀態檢查（Server91）

執行以下指令確認現況：

```bash
# 檢查 uv/uvx（MCP 用）
which uvx uv

# 檢查 paperbanana
pip show paperbanana

# 檢查 mcp.json
ls -la ~/.cursor/mcp.json 2>/dev/null || echo "mcp.json 不存在"

# 檢查 Python
python3 --version
```

**Server91 目前狀態**（截至撰寫時）：
- uvx / uv：未安裝
- paperbanana：未安裝
- mcp.json：不存在
- Python：3.12.9 ✓

---

## 二、安裝流程總覽

```
Step 1: 安裝 uv（可選，用於 MCP）或直接用 pip
   ↓
Step 2: 安裝 paperbanana
   ↓
Step 3: 設定 Google API Key（.env + mcp.json）
   ↓
Step 4: 驗證安裝
```

---

## 三、Step 1：安裝 uv（可選）

**若要用 MCP**（在 Cursor 裡直接呼叫 paperbanana 工具），需要 `uvx`。`uvx` 來自 [uv](https://github.com/astral-sh/uv)：

```bash
# 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重新載入 shell 或
source $HOME/.local/bin/env  # 或 source ~/.bashrc

# 驗證
uvx --version
```

**若不用 MCP**：可跳過此步，直接用 pip 安裝 paperbanana 即可。

---

## 四、Step 2：安裝 PaperBanana

### 方式 A：pip（CLI + 基本使用）

```bash
pip install paperbanana
# 若權限不足，改用：
pip install --user paperbanana
# 或先啟用 conda env：conda activate your_env
```

### 方式 B：pip + MCP 支援（若要用 Cursor MCP）

```bash
pip install "paperbanana[mcp]"
# 若權限不足，改用：
pip install --user "paperbanana[mcp]"
```

### 方式 C：用 uvx 執行 MCP（不需 pip 安裝 paperbanana）

若已安裝 uv，可直接用 `uvx` 跑 MCP，無需事先 `pip install paperbanana`：

```bash
uvx --from paperbanana[mcp] paperbanana-mcp --help
```

若上述指令能執行，代表 MCP 可透過 uvx 啟動。

### 驗證 CLI 安裝

```bash
paperbanana --help
paperbanana setup   # 互動式設定 API key（會開瀏覽器）
```

---

## 五、Step 3：設定 Google API Key

PaperBanana 需要 **Google Gemini API key**。有兩種使用情境，需分別設定：

### 情境 1：CLI（終端機執行 `paperbanana generate`）

CLI **不會**讀 mcp.json，只讀 `.env` 或環境變數。

**在專案目錄建立 `.env`**：

```bash
cd /ssd8/cheng/RARE-VISION-2026-Challenge
echo "GOOGLE_API_KEY=你的_Google_API_金鑰" > .env
```

或手動編輯：

```
GOOGLE_API_KEY=AIzaSy...你的金鑰
```

API key 取得： [Google AI Studio](https://makersuite.google.com/app/apikey)

**注意**：若要用 `gemini-3-pro-image-preview` 等圖像模型，需在 [Google Cloud Console](https://console.cloud.google.com/billing) 啟用 Billing。

---

### 情境 2：MCP（在 Cursor 裡用 paperbanana 工具）

MCP 的 env 來自 `mcp.json`，**不會**讀專案的 `.env`。

**建立 `~/.cursor/mcp.json`**（全域）或專案 `.cursor/mcp.json`：

```json
{
  "mcpServers": {
    "paperbanana": {
      "command": "uvx",
      "args": ["--from", "paperbanana[mcp]", "paperbanana-mcp"],
      "env": {
        "GOOGLE_API_KEY": "你的_Google_API_金鑰"
      }
    }
  }
}
```

**若用 pip 安裝且不用 uvx**，可改為：

```json
{
  "mcpServers": {
    "paperbanana": {
      "command": "paperbanana-mcp",
      "args": [],
      "env": {
        "GOOGLE_API_KEY": "你的_Google_API_金鑰"
      }
    }
  }
}
```

**重要**：修改 mcp.json 後，需**完全關閉並重新啟動 Cursor**，MCP 才會載入。

---

## 六、Step 4：驗證安裝

### CLI 驗證

```bash
cd /ssd8/cheng/RARE-VISION-2026-Challenge

# 建立測試用說明檔
echo "Our pipeline: Input → Encoder → Decoder → Output" > test_method.txt

# 執行（iterations=1 可減少 API 用量）
paperbanana generate \
  --input test_method.txt \
  --caption "Simple pipeline overview" \
  --output test_output.png \
  --iterations 1
```

若成功，會產生 `test_output.png`（或 `outputs/` 下的檔案）。

### MCP 驗證

1. 確認 `~/.cursor/mcp.json` 已建立且格式正確
2. 完全關閉 Cursor 後重新開啟
3. 在 Cursor 對話中請 AI 使用 paperbanana 的 `generate_diagram` 工具

---

## 七、模型與 Billing 注意事項（依你過往經驗）

| 項目 | 說明 |
|------|------|
| **VLM** | `gemini-2.0-flash` 可能 429 rate limit，可改用 `gemini-2.5-flash` 或 `gemini-3.1-pro-preview` |
| **圖像** | `gemini-3-pro-image-preview` 需啟用 Google Cloud Billing，否則會 `limit: 0` |
| **Billing** | 啟用 Billing 後，`gemini-3-pro-image-preview` 可正常使用 |

### 建議的完整指令（有 Billing 時）

```bash
paperbanana generate \
  --input your_method.txt \
  --caption "Your figure caption" \
  --output output.png \
  --vlm-model gemini-3.1-pro-preview \
  --image-model gemini-3-pro-image-preview \
  --iterations 3
```

### 減少 API 用量

- `--iterations 1`：最少 refinement，約 3 次 VLM + 1 次圖像
- 縮短輸入文字可降低 token 成本

---

## 八、快速檢查清單

完成安裝後，確認：

- [ ] `paperbanana --help` 可執行
- [ ] 專案目錄有 `.env` 且含 `GOOGLE_API_KEY`（CLI 用）
- [ ] `~/.cursor/mcp.json` 存在且含 paperbanana 與 `GOOGLE_API_KEY`（MCP 用）
- [ ] 若用 uvx：`uvx --version` 可執行
- [ ] 已重啟 Cursor（若使用 MCP）

---

## 九、常見問題

### Q: CLI 說缺少 API key，但 mcp.json 有設？

**A**：CLI 不讀 mcp.json。在專案目錄建立 `.env`，加入 `GOOGLE_API_KEY=...`。

### Q: MCP 工具呼叫失敗？

**A**：檢查 `~/.cursor/mcp.json` 格式、`GOOGLE_API_KEY` 是否正確，並完全重啟 Cursor。

### Q: gemini-3-pro-image 回傳 429 或 limit: 0？

**A**：需啟用 Google Cloud Billing。或改用 `gemini-2.5-flash-image-preview`（可能有免費額度）或 OpenRouter。

---

## 十、參考

- [PaperBanana GitHub](https://github.com/llmsresearch/paperbanana)
- [PaperBanana PyPI](https://pypi.org/project/paperbanana/)
- [Google AI Studio](https://makersuite.google.com/app/apikey)
- [Google Cloud Billing](https://console.cloud.google.com/billing)
