# 自製 VLM API 規格（與現有 Ollama 呼叫相容）

若需協作方提供「自製視覺語言模型」API，並希望本專案日後僅替換 API 端點即可使用，請要求對方依下列規格實作。

---

## 1. 請求 (Request)

- **方法**: `POST`
- **Content-Type**: `application/json`
- **Body** 需支援以下欄位（可與 Ollama 格式一致，便於直接替換）：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `model` | string | 模型名稱（本端會傳入 `model_name`） |
| `messages` | array | 對話訊息陣列，每筆為 `{ "role": "system" \| "user", "content": "..." }`；最後一則 `role: "user"` 可能帶 `images` |
| `images`（可選） | array of string | Base64 編碼圖片列表；本端會把 `images_b64` 放在最後一則 user message 的 `images` 欄位 |
| `stream` | boolean | 是否串流；本端目前使用 `false`（一次回傳完整內容） |
| `options`（可選） | object | 如 `temperature`, `top_p`, `num_predict` 等，可忽略或自訂 |

**本端實際傳入的對應關係**：
- `model_name` → 請求 body 的 `model`
- `combined_msgs`（或 `event_msgs`）→ 請求 body 的 `messages`
- `images_b64` → 會併入 `messages` 中最後一則 `role: "user"` 的 `images` 欄位

---

## 2. 回應 (Response)

- **HTTP**: 成功時為 `2xx`（例如 200）。
- **Body**: JSON。**必須**能從中取得「模型輸出的純文字」供本端解析事件 JSON 與摘要。

### 建議格式（與 Ollama 相容，可直接沿用現有程式）

```json
{
  "message": {
    "content": "模型輸出的完整文字內容（例如一個 JSON 物件 + 摘要）"
  }
}
```

本端目前取用方式為：
```text
response.json().get("message", {}).get("content", "")
```
回傳值為 **字串**；若缺少 `message.content` 則為空字串。

### 替代格式

若無法提供 `message.content`，可改為頂層 `content`，例如：
```json
{
  "content": "模型輸出的完整文字內容"
}
```
此時本端需改為使用 `response.json().get("content", "")`（或由我們在介面層做一層對應）。

---

## 3. 本端呼叫端語意

- **輸入**：`model_name`, `combined_msgs`（或 `event_msgs`）, `images_b64`（可選）。
- **預期輸出**：一個 **字串**，即模型回傳的整段文字（例如一個 JSON 物件或 JSON + 摘要）。本端會再對此字串做 `_safe_parse_json` / `_extract_first_json` 與 `_clean_summary_text` 等後處理。

只要協作方 API 的「請求參數」與「回應中可取出之字串」符合上述，日後僅需替換請求 URL（及若使用替代回應格式時，改一行取 `content` 的邏輯）即可接上自製模型。
