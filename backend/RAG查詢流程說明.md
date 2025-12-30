# RAG 查詢流程完整說明

## 目錄
1. [整體架構](#整體架構)
2. [查詢流程](#查詢流程)
3. [時間和關鍵字提取](#時間和關鍵字提取)
4. [分數計算機制](#分數計算機制)
5. [資料庫混合查詢](#資料庫混合查詢)
6. [白名單格式說明](#白名單格式說明)
7. [函式流程圖](#函式流程圖)

---

## 整體架構

本系統採用 **PostgreSQL + pgvector** 實現混合搜尋（Hybrid Search），結合：
- **硬篩選（Hard Filter）**：時間範圍、事件類型、關鍵字過濾
- **向量搜尋（Vector Search）**：使用 embedding 的 cosine distance 進行語義相似度計算

### 技術棧
- **資料庫**：PostgreSQL 15
- **向量擴展**：pgvector（提供 `vector` 資料類型和 `cosine_distance` 函數）
- **ORM**：SQLAlchemy 2.0
- **Embedding 模型**：`paraphrase-multilingual-MiniLM-L12-v2`（384 維度）
- **Python 套件**：
  - `pgvector==0.3.0`：PostgreSQL 向量擴展的 Python 綁定
  - `sentence-transformers>=2.3.0`：生成 embedding
  - `sqlalchemy==2.0.23`：ORM 框架

---

## 查詢流程

### 端點：`POST /rag/search`

#### 步驟 1：接收查詢請求
```python
payload = await request.json()
query = payload.get("query")  # 用戶查詢文字
top_k = int(payload.get("top_k") or 5)  # 返回結果數量
score_threshold = float(payload.get("score_threshold") or 0.0)  # 分數門檻
```

#### 步驟 2：解析查詢條件
調用 `_parse_query_filters(query)` 函數，從用戶查詢文字中提取所有過濾條件：

**此函數的處理流程**：
1. **日期解析**：通過 MCP（Model Context Protocol）客戶端調用日期解析工具，按優先順序嘗試：
   - 相對日期關鍵字（今天、昨天、本週等）→ 返回對應的時間範圍
   - YYYYMMDD 格式（2025-12-20、20251220）→ 解析為完整日期
   - MMDD 格式（1220、12/20）→ 使用當前年份補全
   - 自然語言日期（需 date-extractor 庫支援）
   - 最終返回 `time_start`、`time_end`、`date_mode` 等資訊
   
   **MCP 流程說明**：
   - **MCP（Model Context Protocol）**：一個標準化的協議，用於在應用程式和工具之間進行通信
   - **調用位置**：在 `_parse_query_filters()` 函數中，通過 `parse_date_via_mcp(question)` 調用
   - **流程**：
     1. **客戶端**（`mcp_client.py`）發送 JSON-RPC 2.0 請求到 MCP 服務器
     2. **服務器**（`mcp/server.py`）接收請求，調用 `parse_date` 工具
     3. **工具**（`mcp/tools/parse_time.py`）執行日期解析邏輯
     4. **返回結果**：服務器將結果封裝為 JSON-RPC 響應返回給客戶端
     5. **回退機制**：如果 MCP 調用失敗，會直接調用 `parse_query_time_window()` 函數（不使用 MCP 協議）
   - **優勢**：MCP 提供標準化的工具調用接口，可以獨立運行和測試，也方便未來擴展其他工具

2. **事件類型映射**：遍歷 `event_mapping` 字典，檢查查詢中是否包含事件關鍵字（如「火災」、「水災」、「闖入」），將中文關鍵字映射到資料庫欄位名稱（如 `fire`、`water_flood`、`security_door_tamper`），添加到 `event_types` 列表

3. **關鍵字提取**：
   - **事件關鍵字**：檢查查詢中是否包含 `event_keywords_in_message` 列表中的關鍵字（如「火災」、「倒地」、「群聚」），用於後續在 `message` 欄位中進行模糊匹配
   - **描述性關鍵字**：檢查顏色、衣服、車輛相關的關鍵字（如「黃色衣服」、「藍色貨車」），同樣用於 `message` 欄位過濾
   - **地點關鍵字**：檢查地點相關關鍵字（如「路口」、「入口」、「停車場」）

**返回的過濾條件字典**：
```python
{
    "date_filter": date(2025, 12, 20),  # 向後兼容的日期對象
    "time_start": "2025-12-20T00:00:00+08:00",  # ISO 格式開始時間
    "time_end": "2025-12-21T00:00:00+08:00",  # ISO 格式結束時間
    "date_mode": "MMDD_RULE",  # 日期解析模式
    "event_types": ["fire", "water_flood"],  # 事件類型列表（資料庫欄位名稱）
    "message_keywords": ["火災", "黃色衣服"],  # 關鍵字列表（用於 message 欄位過濾）
    "location_keywords": ["路口"]  # 地點關鍵字列表
}
```

這些過濾條件將在後續的資料庫查詢中使用，實現精確的時間範圍篩選、事件類型篩選和關鍵字匹配。

**精準定位方法說明**：

1. **日期數字提取（使用正則表達式）**：
   - **方法**：使用 Python 的 `re.search()` 函數配合正則表達式模式匹配
   - **關鍵技術**：使用**負向先行斷言** `(?<!\d)` 和**負向後行斷言** `(?!\d)` 來確保匹配的是獨立的數字序列，而不是其他數字的一部分
   
   **範例**：
   ```python
   # MMDD 格式：匹配 12/20 或 12-20
   re.search(r"(?<!\d)(\d{1,2})[/-](\d{1,2})(?!\d)", "給我 12/20 的影片")
   # (?<!\d) 確保前面不是數字
   # (\d{1,2}) 匹配 1-2 位數字（月份）
   # [/-] 匹配斜線或連字號
   # (\d{1,2}) 匹配 1-2 位數字（日期）
   # (?!\d) 確保後面不是數字
   # 結果：匹配到 "12/20"，提取出 (12, 20)
   
   # MMDD 格式：匹配獨立的 4 位數字 1220（排除 8 位數的 20251220）
   re.search(r"(?<!\d)(\d{4})(?!\d)", "給我 1220 的影片")
   # (?<!\d) 確保前面不是數字（避免匹配到 20251220 的後 4 位）
   # (\d{4}) 匹配恰好 4 位數字
   # (?!\d) 確保後面不是數字（避免匹配到 20251220 的前 4 位）
   # 結果：匹配到 "1220"，提取出 (12, 20)
   
   # YYYYMMDD 格式：匹配 8 位數字 20251220
   re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", "給我 20251220 的影片")
   # 結果：匹配到 "20251220"，提取出 (2025, 12, 20)
   ```
   
   **為什麼需要負向斷言**：
   - 如果查詢是 "給我 20251220 的影片"，沒有負向斷言的正則可能會錯誤匹配到 "1220"（從 20251220 中提取）
   - 使用 `(?<!\d)` 和 `(?!\d)` 確保匹配的是**獨立的數字序列**，不會被其他數字干擾

2. **事件和關鍵字提取（使用字符串包含檢查）**：
   - **方法**：使用 Python 的 `in` 運算符進行簡單的字符串包含檢查
   - **特點**：不返回精確位置，只檢查關鍵字是否存在於查詢文字中
   
   **範例**：
   ```python
   # 事件類型映射
   question = "給我 1220 的火災影片"
   question_lower = question.lower()  # "給我 1220 的火災影片"
   
   for keyword, db_field in event_mapping.items():
       if keyword in question_lower:  # "火災" in "給我 1220 的火災影片" → True
           filters["event_types"].append(db_field)  # 添加 "fire"
   
   # 關鍵字提取
   question = "給我黃色衣服的影片"
   for keyword in descriptive_keywords:
       if keyword in question:  # "黃色衣服" in "給我黃色衣服的影片" → True
           message_keywords_found.append(keyword)  # 添加 "黃色衣服"
   ```
   
   **匹配優先順序**：
   - 先檢查**完整的多字關鍵字**（如「黃色衣服」、「藍色貨車」）
   - 再檢查單字關鍵字（如「黃色」、「衣服」）
   - 這樣可以避免誤匹配（例如：先匹配「黃色衣服」就不會再單獨匹配「黃色」和「衣服」）

**關鍵字提取的實際用途**：
- **不是用來計算 embedding**，而是作為**硬篩選條件**（Hard Filter）
- 提取的關鍵字會在資料庫查詢的 `WHERE` 子句中使用：`WHERE message ILIKE '%關鍵字%'`
- 這是在向量搜索**之前**的過濾步驟，用來縮小搜索範圍，提高查詢效率
- **範例**：如果查詢是「給我 1220 的火災影片」，系統會：
  1. 提取關鍵字「火災」
  2. 在資料庫中先過濾：`WHERE message ILIKE '%火災%'`（硬篩選）
  3. 然後對剩餘的記錄進行向量相似度計算和排序（向量搜索）

**Embedding 計算方式**：
- 目前實現是對**整個查詢句子**（去除日期後）進行 embedding 計算
- 例如：「給我 1220 的火災影片」→ 去除日期後 → 「給我 的火災影片」→ 計算 embedding
- **不是**只對關鍵字（如「火災」）進行 embedding 計算

**設計考量：整句 vs 關鍵字 embedding**：
- **整句 embedding 的優勢**：
  - 保留完整的語義上下文（例如「給我火災的影片」vs「火災」）
  - 可以理解查詢的意圖和語氣
  - 對於複雜查詢（如「給我昨天下午發生火災的影片」）效果更好
  
- **關鍵字 embedding 的優勢**：
  - 更聚焦於核心概念，減少噪音
  - 對於簡單查詢（如「火災」）可能更精準
  - 但可能失去語義上下文（例如「不是火災」vs「火災」）

- **目前的混合策略**：
  - 使用關鍵字進行**硬篩選**（精確匹配）
  - 使用整句進行**向量搜索**（語義相似度）
  - 結合兩者優勢：先縮小範圍，再進行語義匹配

3. **相對日期關鍵字（使用字符串包含檢查）**：
   - **方法**：同樣使用 `in` 運算符檢查相對日期關鍵字
   - **範例**：
   ```python
   if "今天" in text or "今日" in text:
       # 返回今天的時間範圍
   if "昨天" in text:
       # 返回昨天的時間範圍
   if "本週" in text or "這週" in text:
       # 返回本週的時間範圍
   ```

**總結**：
- **數字提取**：使用**正則表達式**配合**負向斷言**，精準定位獨立的數字序列，避免誤匹配
- **事件和關鍵字提取**：使用**字符串包含檢查**（`in` 運算符），簡單高效，但需要維護白名單列表
- **相對日期**：使用**字符串包含檢查**，匹配預定義的關鍵字列表

#### 步驟 3：生成查詢向量
1. 從查詢中移除日期相關文字，生成 `clean_query`
2. 使用 `SentenceTransformer` 模型將 `clean_query` 轉換為 384 維向量
3. 向量經過 `normalize_embeddings=True` 正規化

#### 步驟 4：構建混合查詢
使用 SQLAlchemy 構建單一 SQL 查詢，包含：
- **Filter 1（硬篩選）**：時間範圍、事件類型、關鍵字
- **Filter 2（向量搜尋）**：使用 `cosine_distance` 排序

#### 步驟 5：計算分數並格式化結果
1. 將 `cosine_distance` 轉換為相似度分數 `[0, 1]`
2. 過濾低於 `score_threshold` 的結果
3. 返回前 `top_k` 筆結果

---

## 時間和關鍵字提取

### 函式：`_parse_query_filters(question: str)`

此函式負責從用戶查詢中提取所有過濾條件。

#### 1. 日期解析

**調用鏈**：
```
_parse_query_filters()
  └─> parse_date_via_mcp() 或 parse_query_time_window()
      └─> parse_relative_date()  # 優先：相對日期
      └─> parse_yyyymmdd_from_text()  # 次優先：YYYYMMDD 格式
      └─> parse_mmdd_from_text()  # 第三優先：MMDD 格式
      └─> extract_dates()  # 最後：使用 date-extractor 庫（可選）
```

**支援的日期格式**：

| 類型 | 格式範例 | 解析模式 |
|------|---------|---------|
| 相對日期 | "今天"、"昨天"、"本週"、"上週" | `RELATIVE_*` |
| MMDD | "1220"、"12/20"、"12-20" | `MMDD_RULE` |
| YYYYMMDD | "20251220"、"2025-12-20" | `YYYYMMDD_RULE` |
| 自然語言 | "2025年12月20日" | `DATE_EXTRACTOR`（需安裝 date-extractor） |

**日期解析優先順序**：
1. **相對日期關鍵字**（最快、最準確）
   - 支援：今天、昨天、前天、明天、本週、上週、下週
   - 使用正則表達式匹配：`"今天" in text`
2. **YYYYMMDD 格式**（優先於 MMDD）
   - 正則：`r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'` 或 `r'\d{8}'`
3. **MMDD 格式**
   - 正則：`r'(?<!\d)(\d{1,2})[/-](\d{1,2})(?!\d)'` 或 `r'(?<!\d)(\d{4})(?!\d)'`
   - 自動使用當前年份
4. **自然語言日期**（最後嘗試）
   - 使用 `date-extractor` 庫（容器內可能未安裝）

**返回格式**：
```python
{
    "date_filter": date(2025, 12, 20),  # 向後兼容
    "time_start": "2025-12-20T00:00:00+08:00",  # ISO 格式
    "time_end": "2025-12-21T00:00:00+08:00",
    "date_mode": "MMDD_RULE"
}
```

#### 2. 關鍵字提取

**事件關鍵字**（用於 message 欄位過濾）：
```python
event_keywords_in_message = [
    "火災", "倒地", "群聚", "聚眾", "水災", "淹水", 
    "闖入", "遮臉", "吸菸", "停車", "阻塞"
]
```

**描述性關鍵字**（顏色、衣服、車輛）：
```python
descriptive_keywords = [
    # 顏色 + 衣服
    "黃色衣服", "黑色衣服", "白色衣服", "紅色衣服", "藍色衣服", "綠色衣服",
    "深色衣服", "淺色衣服", "灰色衣服",
    # 顏色 + 車輛
    "藍色貨車", "白色貨車", "紅色貨車", "黑色貨車", "綠色貨車", "黃色貨車",
    "藍色卡車", "白色卡車", "紅色卡車", "黑色卡車",
    "藍色汽車", "白色汽車", "紅色汽車", "黑色汽車",
    "藍色機車", "白色機車", "紅色機車", "黑色機車",
    "藍色車", "白色車", "紅色車", "黑色車", "黃色車",
    # 單獨的顏色
    "黃色", "黑色", "白色", "紅色", "藍色", "綠色", "灰色", "深色", "淺色",
    # 衣服相關
    "衣服", "上衣", "褲子", "帽子",
    # 車輛相關
    "貨車", "卡車", "汽車", "機車", "車"
]
```

**地點關鍵字**：
```python
location_keywords = ["路口", "入口", "出口", "停車場", "大門", "側門", "後門"]
```

**提取邏輯**：
- 使用簡單的字符串包含檢查：`if keyword in question`
- 先檢查完整的多字關鍵字（如「黃色衣服」），再檢查單字關鍵字
- 所有匹配的關鍵字會添加到 `filters["message_keywords"]` 列表中

#### 3. 事件類型映射

**事件映射表**（中文關鍵字 → 資料庫欄位）：
```python
event_mapping = {
    "火災": "fire",
    "火": "fire",
    "水災": "water_flood",
    "水": "water_flood",
    "淹水": "water_flood",
    "積水": "water_flood",
    "闖入": "security_door_tamper",
    "突破": "security_door_tamper",
    "安全門": "security_door_tamper",
    "遮臉": "abnormal_attire_face_cover_at_entry",
    "異常著裝": "abnormal_attire_face_cover_at_entry",
    "倒地": "person_fallen_unmoving",
    "倒地不起": "person_fallen_unmoving",
    "併排": "double_parking_lane_block",
    "停車": "double_parking_lane_block",
    "阻塞": "double_parking_lane_block",
    "吸菸": "smoking_outside_zone",
    "抽菸": "smoking_outside_zone",
    "聚眾": "crowd_loitering",
    "逗留": "crowd_loitering",
}
```

**提取邏輯**：
- 將查詢轉為小寫：`question_lower = question.lower()`
- 遍歷映射表，檢查關鍵字是否在查詢中
- 匹配到的事件類型會添加到 `filters["event_types"]` 列表中

---

## 分數計算機制

### 向量相似度分數

**計算公式**：
```python
# cosine_distance 範圍：[0, 2]
# 0 = 完全相同，2 = 完全相反
cosine_distance = Summary.embedding.cosine_distance(query_embedding)

# 轉換為相似度分數 [0, 1]
# 0 = 完全不相關，1 = 完全相關
score = 1 - (cosine_distance / 2.0)

# 確保分數在 [0, 1] 範圍內
score = max(0.0, min(1.0, score))
```

**數學原理**：
- `cosine_distance = 1 - cosine_similarity`
- `cosine_similarity` 範圍：`[-1, 1]`
- `cosine_distance` 範圍：`[0, 2]`
- 為了將距離轉換為 `[0, 1]` 範圍的相似度分數，使用線性映射：`score = 1 - (distance / 2)`

**範例**：
| cosine_distance | score | 意義 |
|----------------|-------|------|
| 0.0 | 1.0 | 完全相關 |
| 0.5 | 0.75 | 高度相關 |
| 1.0 | 0.5 | 中等相關 |
| 1.5 | 0.25 | 低度相關 |
| 2.0 | 0.0 | 完全不相關 |

### 分數過濾

查詢結果會根據 `score_threshold` 參數過濾：
```python
if score < score_threshold:
    continue  # 跳過此結果
```

**預設行為**：
- `score_threshold = 0.0`：不過濾任何結果（僅按分數排序）
- `score_threshold = 0.6`：只返回分數 ≥ 0.6 的結果

---

## 資料庫混合查詢

### PostgreSQL + pgvector 架構

**pgvector 擴展**：
- PostgreSQL 的開源向量擴展，提供 `vector` 資料類型和向量運算函數
- 支援多種距離計算：`cosine_distance`、`L2 distance`、`inner product`
- 本系統使用 `cosine_distance` 進行語義相似度計算

**資料表結構**：
```sql
CREATE TABLE summaries (
    id SERIAL PRIMARY KEY,
    start_timestamp TIMESTAMP,
    message TEXT,
    embedding vector(384),  -- pgvector 類型，384 維度
    fire BOOLEAN,
    water_flood BOOLEAN,
    -- ... 其他事件類型欄位
);
```

**混合查詢實現**：

```python
from sqlalchemy import select, and_, or_
from pgvector.sqlalchemy import Vector

# 步驟 1：構建基礎查詢（硬篩選）
stmt = select(Summary).filter(
    Summary.message.isnot(None),
    Summary.message != "",
    Summary.embedding.isnot(None)  # 只查詢有 embedding 的記錄
)

# 步驟 2：時間範圍過濾（硬篩選）
if query_filters.get("time_start") and query_filters.get("time_end"):
    stmt = stmt.filter(
        Summary.start_timestamp.between(time_start, time_end)
    )

# 步驟 3：事件類型過濾（硬篩選）
if query_filters.get("event_types"):
    event_conditions = []
    for event_type in query_filters["event_types"]:
        event_conditions.append(getattr(Summary, event_type) == True)
    if event_conditions:
        stmt = stmt.filter(or_(*event_conditions))

# 步驟 4：關鍵字過濾（硬篩選）
if query_filters.get("message_keywords"):
    message_conditions = []
    for keyword in query_filters["message_keywords"]:
        message_conditions.append(Summary.message.ilike(f"%{keyword}%"))
    if message_conditions:
        stmt = stmt.filter(or_(*message_conditions))

# 步驟 5：向量搜尋（使用 cosine_distance 排序）
distance_expr = Summary.embedding.cosine_distance(query_embedding)
stmt = stmt.add_columns(
    distance_expr.label('cosine_distance')
).order_by(
    distance_expr  # 按距離升序排列（距離越小越相似）
).limit(top_k * 3)  # 多取一些，後續可以根據分數過濾

# 步驟 6：執行查詢
results = db.execute(stmt).all()
```

**優勢**：
1. **單一查詢**：所有過濾和排序在一個 SQL 查詢中完成，無需多次查詢或合併結果
2. **高效能**：PostgreSQL 的索引和 pgvector 的 HNSW 索引可以大幅提升查詢速度
3. **精確過濾**：硬篩選確保只返回符合時間、事件、關鍵字條件的記錄
4. **語義搜尋**：向量搜尋可以找到語義相關但關鍵字不完全匹配的記錄

---

## 白名單格式說明

### 1. 事件類型映射（字典格式）

**位置**：`backend/src/main.py` 的 `_parse_query_filters()` 函數

**格式**：
```python
event_mapping = {
    "中文關鍵字": "資料庫欄位名稱",
    # 範例
    "火災": "fire",
    "水災": "water_flood",
    "闖入": "security_door_tamper",
}
```

**用途**：將用戶查詢中的中文關鍵字映射到資料庫的 Boolean 欄位

**對應的資料庫欄位**：
- `fire`：火災
- `water_flood`：水災
- `abnormal_attire_face_cover_at_entry`：異常著裝/遮臉入場
- `person_fallen_unmoving`：人員倒地不起
- `double_parking_lane_block`：併排停車/車道阻塞
- `smoking_outside_zone`：違規吸菸
- `crowd_loitering`：聚眾逗留
- `security_door_tamper`：突破安全門

### 2. 關鍵字列表（陣列格式）

**位置**：`backend/src/main.py` 的 `_parse_query_filters()` 函數

**類型 A：事件關鍵字**（用於 message 欄位過濾）
```python
event_keywords_in_message = [
    "火災", "倒地", "群聚", "聚眾", "水災", "淹水", 
    "闖入", "遮臉", "吸菸", "停車", "阻塞"
]
```

**類型 B：描述性關鍵字**（顏色、衣服、車輛）
```python
descriptive_keywords = [
    "黃色衣服", "黑色衣服", "白色衣服", "紅色衣服", "藍色衣服", "綠色衣服",
    "深色衣服", "淺色衣服", "灰色衣服",
    "藍色貨車", "白色貨車", "紅色貨車", "黑色貨車", "綠色貨車", "黃色貨車",
    # ... 更多關鍵字
]
```

**類型 C：地點關鍵字**
```python
location_keywords = ["路口", "入口", "出口", "停車場", "大門", "側門", "後門"]
```

**用途**：從用戶查詢中提取關鍵字，用於在 `message` 欄位中進行模糊匹配（`LIKE '%關鍵字%'`）

### 3. 日期解析關鍵字（正則表達式 + 關鍵字列表）

**位置**：`backend/src/mcp/tools/parse_time.py`

**類型 A：相對日期關鍵字**
```python
# 在 parse_relative_date() 函數中使用簡單的字符串匹配
if "今天" in text or "今日" in text:
    # 返回今天 00:00 ~ 明天 00:00
if "昨天" in text:
    # 返回昨天 00:00 ~ 今天 00:00
if "本週" in text or "這週" in text:
    # 返回本週一 00:00 ~ 下週一 00:00
```

**類型 B：日期格式正則表達式**
```python
# MMDD 格式
r'(?<!\d)(\d{1,2})[/-](\d{1,2})(?!\d)'  # 12/20, 12-20
r'(?<!\d)(\d{4})(?!\d)'  # 1220（4 位數，前兩位是月份）

# YYYYMMDD 格式
r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'  # 2025-12-20, 2025/12/20
r'\d{8}'  # 20251220

# 自然語言日期（由 date-extractor 庫處理）
r'\d{4}年\d{1,2}月\d{1,2}日'  # 2025年12月20日
```

**用途**：從用戶查詢中解析日期和時間範圍

---

## 函式流程圖

### 完整查詢流程

```
用戶查詢: "給我 1225 的火災影片"
    │
    ▼
POST /rag/search
    │
    ├─> 步驟 1: 接收請求
    │   └─> query = "給我 1225 的火災影片"
    │       top_k = 5
    │       score_threshold = 0.0
    │
    ├─> 步驟 2: 解析查詢條件
    │   └─> _parse_query_filters(query)
    │       │
    │       ├─> 日期解析（通過 MCP）
    │       │   └─> parse_date_via_mcp(question)
    │       │       │
    │       │       ├─> MCP 客戶端（mcp_client.py）
    │       │       │   └─> 發送 JSON-RPC 請求: {"method": "tools/call", "params": {"name": "parse_date", "arguments": {"query": "給我 1220 的火災影片"}}}
    │       │       │
    │       │       ├─> MCP 服務器（mcp/server.py）
    │       │       │   └─> 接收請求，調用 handle_tools_call("parse_date", arguments)
    │       │       │
    │       │       ├─> 日期解析工具（mcp/tools/parse_time.py）
    │       │       │   └─> parse_query_time_window(query)
    │       │       │       ├─> parse_relative_date()  # 無匹配
    │       │       │       ├─> parse_yyyymmdd_from_text()  # 無匹配
    │       │       │       └─> parse_mmdd_from_text()  # 匹配 "1220" → (12, 20)
    │       │       │
    │       │       └─> 返回 JSON-RPC 響應: {"result": {"content": [{"text": "{...日期解析結果...}"}]}}
    │       │           └─> 如果 MCP 失敗，回退到直接調用 parse_query_time_window()
    │       │
    │       │       最終返回: time_start="2025-12-20T00:00:00+08:00"
    │       │                 time_end="2025-12-21T00:00:00+08:00"
    │       │                 date_mode="MMDD_RULE"
    │       │
    │       ├─> 事件類型映射
    │       │   └─> "火災" in query → event_types = ["fire"]
    │       │
    │       └─> 關鍵字提取
    │           └─> "火災" in event_keywords_in_message → message_keywords = ["火災"]
    │
    ├─> 步驟 3: 生成查詢向量
    │   └─> clean_query = "給我 的火災影片"  # 移除 "1225"
    │       └─> embedding_model.encode(clean_query)
    │           └─> query_embedding = [0.123, -0.456, ..., 0.789]  # 384 維向量
    │
    ├─> 步驟 4: 構建混合查詢
    │   └─> SQLAlchemy 查詢
    │       │
    │       ├─> Filter 1 (硬篩選)
    │       │   ├─> WHERE start_timestamp BETWEEN '2025-12-20 00:00:00' AND '2025-12-21 00:00:00'
    │       │   ├─> AND fire = TRUE
    │       │   └─> AND message ILIKE '%火災%'
    │       │
    │       └─> Filter 2 (向量搜尋)
    │           └─> ORDER BY embedding <=> query_embedding  # cosine_distance
    │               LIMIT 15  # top_k * 3
    │
    ├─> 步驟 5: 計算分數並格式化
    │   └─> for each result:
    │       ├─> cosine_distance = 0.234
    │       ├─> score = 1 - (0.234 / 2.0) = 0.883
    │       ├─> if score >= score_threshold: 保留
    │       └─> 格式化為 JSON
    │
    └─> 步驟 6: 返回結果
        └─> {
              "backend": "paraphrase-multilingual-MiniLM-L12-v2",
              "hits": [
                {
                  "score": 0.883,
                  "video": "fire_1",
                  "segment": "segment_0000.mp4",
                  "time_range": "00:00:00 - 00:00:08",
                  "events_true": ["fire"],
                  "summary": "畫面中出現火災濃煙...",
                  "reason": "檢測到火災事件"
                },
                # ... 更多結果
              ],
              "date_parsed": {
                "mode": "MMDD_RULE",
                "picked_date": "2025-12-20",
                "time_start": "2025-12-20T00:00:00+08:00",
                "time_end": "2025-12-21T00:00:00+08:00"
              },
              "keywords_found": ["火災"],
              "event_types_found": ["fire"],
              "embedding_query": "給我 的火災影片"
            }
```

### 日期解析流程

```
parse_query_time_window(query: str)
    │
    ├─> 優先級 1: 相對日期
    │   └─> parse_relative_date(query, now)
    │       ├─> if "今天" in query → RELATIVE_TODAY
    │       ├─> if "昨天" in query → RELATIVE_YESTERDAY
    │       ├─> if "本週" in query → RELATIVE_THIS_WEEK
    │       └─> ... 其他相對日期關鍵字
    │
    ├─> 優先級 2: YYYYMMDD 格式
    │   └─> parse_yyyymmdd_from_text(query)
    │       ├─> 正則: r'\d{4}[-/]\d{1,2}[-/]\d{1,2}'  # 2025-12-20
    │       └─> 正則: r'\d{8}'  # 20251220
    │
    ├─> 優先級 3: MMDD 格式
    │   └─> parse_mmdd_from_text(query)
    │       ├─> 正則: r'(?<!\d)(\d{1,2})[/-](\d{1,2})(?!\d)'  # 12/20
    │       └─> 正則: r'(?<!\d)(\d{4})(?!\d)'  # 1220
    │           └─> 檢查: 前兩位是否在 01-12 範圍內
    │
    └─> 優先級 4: 自然語言日期（可選）
        └─> extract_dates(query)  # 使用 date-extractor 庫
            └─> 如果庫未安裝，跳過此步驟
```

### 關鍵字提取流程

```
_parse_query_filters(question: str)
    │
    ├─> 事件關鍵字提取
    │   └─> for keyword in event_keywords_in_message:
    │       └─> if keyword in question:
    │           └─> message_keywords.append(keyword)
    │
    ├─> 描述性關鍵字提取
    │   └─> for keyword in descriptive_keywords:
    │       └─> if keyword in question and keyword not in message_keywords:
    │           └─> message_keywords.append(keyword)
    │
    ├─> 事件類型映射
    │   └─> for keyword, db_field in event_mapping.items():
    │       └─> if keyword in question.lower():
    │           └─> event_types.append(db_field)
    │
    └─> 地點關鍵字提取
        └─> for keyword in location_keywords:
            └─> if keyword in question:
                └─> location_keywords.append(keyword)
```

---

## MCP 流程說明

### 什麼是 MCP？

**MCP（Model Context Protocol）** 是一個標準化的協議，最初設計用於**大型語言模型（LLM）與外部工具之間的通信**。它允許 AI 模型通過標準化接口調用外部工具（如資料庫查詢、API 調用、檔案操作等）。

**在本系統中的實際使用**：

雖然 MCP 的原始設計目的是為 AI 模型提供工具調用接口，但在本系統中：
- **MCP 只是作為工具封裝框架使用**，實際上**沒有調用任何 AI 模型**
- 搜尋功能（`/rag/search`）確實和模型無關，只是借用了 MCP 的協議和架構來封裝日期解析工具
- 使用 MCP 的好處是：
  1. **標準化接口**：使用 JSON-RPC 2.0 協議，易於維護和擴展
  2. **獨立運行**：工具可以獨立測試，不依賴主程式
  3. **未來擴展**：如果未來需要讓 LLM 調用這些工具，已經有標準化的接口

**為什麼搜尋功能使用 MCP？**

搜尋功能本身確實不需要 AI 模型，但使用 MCP 來封裝日期解析工具的原因是：
- **模組化設計**：將日期解析邏輯獨立出來，便於維護和測試
- **協議標準化**：使用標準的 JSON-RPC 協議，而不是自定義的通信方式
- **未來兼容性**：如果未來需要讓 LLM 參與查詢解析（例如理解更複雜的自然語言查詢），已經有現成的接口可以使用

**系統中的模型使用情況**：

本系統有兩個 RAG 端點，它們的區別如下：

1. **`/rag/search`**（搜尋功能）：
   - **不使用 AI 模型**：只進行資料庫查詢和向量相似度計算
   - 使用 MCP 只是為了封裝日期解析工具（純規則匹配，無模型參與）
   - 返回符合條件的影片片段列表

2. **`/rag/answer`**（問答功能）：
   - **使用 AI 模型**（Gemini 或 Ollama）：會調用 LLM 來閱讀檢索到的資料並生成回答
   - 這個功能確實需要模型參與，但 MCP 在這裡仍然只是用於日期解析，不是模型調用

**MCP 與 LLM 的配合關係**：

在 `/rag/answer` 端點中，MCP 和 LLM 是**串聯使用**，但**沒有直接配合**：

```
用戶查詢: "給我 1220 的火災影片，發生了什麼事？"
    │
    ▼
步驟 1: 使用 MCP 解析日期（通過 _parse_query_filters）
    │   └─> MCP 解析出日期：2025-12-20
    │   └─> 提取事件類型：fire
    │   └─> 提取關鍵字：火災
    │
    ▼
步驟 2-4: 資料庫查詢（使用 MCP 解析的結果）
    │   └─> 使用日期範圍過濾資料庫
    │   └─> 使用事件類型過濾
    │   └─> 使用向量搜索找到相關片段
    │   └─> 返回 top_k 筆結果
    │
    ▼
步驟 5: 調用 LLM 生成回答（使用查詢結果）
    │   └─> 將檢索到的片段摘要組裝成 context
    │   └─> 調用 _ollama_chat() 或 Gemini API
    │   └─> LLM 閱讀 context 並生成回答
    │
    ▼
返回結果: {answer: "根據片段 [1] 和 [2]，在 12/20 發生了火災事件..."}
```

**關鍵點**：
1. **MCP 和 LLM 是分離的**：
   - MCP 在**步驟 1**中用於解析用戶查詢（日期、事件、關鍵字）
   - LLM 在**步驟 5**中用於生成回答
   - 它們之間**沒有直接通信**

2. **MCP 的結果用於資料庫查詢，而不是直接傳給 LLM**：
   - MCP 解析出的日期、事件類型、關鍵字用於**過濾資料庫**
   - 資料庫查詢結果（片段摘要）才傳給 LLM
   - LLM 看到的是**已經過濾好的片段摘要**，而不是原始的用戶查詢

3. **為什麼沒有讓 LLM 直接調用 MCP？**：
   - 目前的設計是**規則匹配**（MCP 解析）+ **資料庫查詢** + **LLM 生成回答**
   - 理論上可以讓 LLM 直接調用 MCP 工具（這是 MCP 的原始設計目的），但本系統目前沒有這樣做
   - 原因可能是：
     - 規則匹配已經足夠準確（日期解析、事件識別）
     - 避免 LLM 的額外延遲和成本
     - 保持查詢邏輯的可預測性

**如何讓 LLM 調用 MCP 工具（實現方案）**：

雖然目前沒有實現，但可以通過以下方式讓 LLM 直接調用 MCP 工具：

1. **修改 `_ollama_chat` 函數**：
   - 添加 `tools` 參數，支援傳遞工具定義
   - 在 payload 中添加 `tools` 和 `tool_choice` 欄位
   - 解析響應中的 `tool_calls`，返回工具調用請求

2. **在 `/rag/answer` 中實現工具調用循環**：
   ```python
   # 步驟 1: 讓 LLM 分析查詢，決定是否需要調用工具
   tools = [{
       "type": "function",
       "function": {
           "name": "parse_date",
           "description": "解析中文查詢中的日期時間範圍",
           "parameters": {...}
       }
   }]
   
   response = _ollama_chat(llm_model, messages, tools=tools)
   
   # 步驟 2: 如果 LLM 請求調用工具，執行工具調用
   if isinstance(response, dict) and "tool_calls" in response:
       for tool_call in response["tool_calls"]:
           if tool_call["function"]["name"] == "parse_date":
               # 調用 MCP 工具
               date_result = mcp_client.parse_date(tool_call["function"]["arguments"]["query"])
               
               # 將工具結果返回給 LLM
               messages.append({
                   "role": "assistant",
                   "tool_calls": response["tool_calls"]
               })
               messages.append({
                   "role": "tool",
                   "tool_call_id": tool_call["id"],
                   "content": json.dumps(date_result)
               })
   
   # 步驟 3: 讓 LLM 繼續生成回答（使用工具結果）
   final_response = _ollama_chat(llm_model, messages)
   ```

3. **優勢**：
   - LLM 可以更智能地決定何時需要解析日期
   - 可以處理更複雜的自然語言查詢
   - 符合 MCP 的原始設計目的

4. **注意事項**：
   - 需要確保 LLM 模型支援 function calling（Ollama 的新版本支援）
   - 會增加延遲（需要多次 LLM 調用）
   - 需要處理工具調用失敗的情況

**總結**：
- MCP 和 LLM 在 `/rag/answer` 中是**串聯使用**，但**沒有直接配合**
- MCP 用於**查詢解析**（規則匹配），LLM 用於**回答生成**（閱讀資料）
- 它們通過**資料庫查詢結果**連接：MCP 解析 → 資料庫查詢 → LLM 生成回答

**總結**：
- MCP 在本系統中只是作為**工具封裝框架**，用於日期解析（規則匹配，無模型參與）
- `/rag/search` 端點完全不使用 AI 模型，只是資料庫查詢
- `/rag/answer` 端點使用 AI 模型來生成回答，但 MCP 仍然只用於日期解析，不是模型調用
- 使用 MCP 的好處是代碼組織、標準化接口和未來擴展性

**重要區別：MCP 不是 HTTP API**

MCP 與 FastAPI 的 HTTP API 端點不同：
- **FastAPI 端點**（如 `/rag/search`、`/rag/answer`）：
  - 使用 HTTP 協議（GET、POST 等）
  - 通過網路請求訪問（`http://localhost:8080/rag/search`）
  - 會出現在 FastAPI 的自動生成文檔中（`/docs`）
  - 可以被外部客戶端（瀏覽器、Postman 等）直接調用

- **MCP 服務器**：
  - **不是 HTTP API**，而是通過**子進程（subprocess）**啟動的獨立 Python 程序
  - 使用**標準輸入/輸出（stdin/stdout）**進行通信，不是網路協議
  - 使用 **JSON-RPC 2.0** 協議，但通過進程間通信（IPC），不是 HTTP
  - **不會出現在 FastAPI 的 `/docs` 文檔中**，因為它不是 FastAPI 的端點
  - 只能由主程式內部調用，無法從外部直接訪問

**為什麼使用 MCP 而不是 HTTP API？**
1. **輕量級**：不需要啟動額外的 HTTP 服務器，減少資源消耗
2. **簡單**：進程間通信比網路通信更簡單、更快速
3. **隔離**：工具可以獨立運行，不影響主服務器
4. **標準化**：使用 JSON-RPC 2.0 協議，易於擴展和維護

### MCP 架構

```
主程式 (main.py)
    │
    ├─> _parse_query_filters(query)
    │   └─> parse_date_via_mcp(question)  # MCP 客戶端調用
    │       │
    │       ▼
    │   MCP 客戶端 (mcp_client.py)
    │   │
    │   ├─> 啟動子進程：python3 mcp/server.py
    │   ├─> 發送 JSON-RPC 請求（通過 stdin）
    │   └─> 接收 JSON-RPC 響應（通過 stdout）
    │       │
    │       ▼
    │   MCP 服務器 (mcp/server.py)
    │   │
    │   ├─> 接收 JSON-RPC 請求
    │   ├─> 解析請求：{"method": "tools/call", "params": {...}}
    │   └─> 調用工具：handle_tools_call("parse_date", arguments)
    │       │
    │       ▼
    │   日期解析工具 (mcp/tools/parse_time.py)
    │   │
    │   └─> parse_query_time_window(query)
    │       ├─> parse_relative_date()      # 相對日期
    │       ├─> parse_yyyymmdd_from_text() # YYYYMMDD 格式
    │       └─> parse_mmdd_from_text()     # MMDD 格式
    │
    └─> 返回日期解析結果
```

### MCP 通信流程

1. **初始化階段**：
   - 客戶端啟動 MCP 服務器子進程（`python3 mcp/server.py`）
   - 發送 `initialize` 請求，建立連接

2. **工具調用階段**：
   - 客戶端通過 `stdin.write()` 發送 JSON-RPC 請求（**不是 HTTP POST**）：
     ```json
     {
       "jsonrpc": "2.0",
       "id": 1,
       "method": "tools/call",
       "params": {
         "name": "parse_date",
         "arguments": {
           "query": "給我 1220 的影片"
         }
       }
     }
     ```
   - MCP 服務器從 `sys.stdin` 讀取請求（**不是 HTTP 請求**）
   - 服務器調用 `parse_date` 工具
   - 工具執行日期解析邏輯
   - 服務器通過 `print()` 將結果寫入 `stdout`（**不是 HTTP 響應**）：
     ```json
     {
       "jsonrpc": "2.0",
       "id": 1,
       "result": {
         "content": [
           {
             "type": "text",
             "text": "{\"mode\":\"MMDD_RULE\",\"picked_date\":\"2025-12-20\",...}"
           }
         ]
       }
     }
     ```
   - 客戶端通過 `stdout.readline()` 讀取響應

3. **回退機制**：
   - 如果 MCP 調用失敗（服務器未啟動、通信錯誤等），會直接調用 `parse_query_time_window()` 函數
   - 確保即使 MCP 不可用，日期解析功能仍然可以正常工作

### MCP 的優勢

1. **標準化接口**：使用 JSON-RPC 2.0 協議，易於擴展和維護
2. **獨立運行**：工具可以獨立測試，不依賴主程式
3. **易於擴展**：可以輕鬆添加新的工具（如關鍵字提取、事件識別等）
4. **容錯性**：有回退機制，確保系統穩定性

### 相關檔案

- **客戶端**：`backend/src/mcp_client.py`
- **服務器**：`backend/src/mcp/server.py`
- **工具**：`backend/src/mcp/tools/parse_time.py`
- **配置**：`backend/src/mcp/config.yaml`
- **工具規格**：`backend/src/mcp/schemas/tools.json`

---

## 總結

### 核心優勢

1. **單一查詢混合搜尋**：所有過濾和排序在一個 SQL 查詢中完成，無需多次查詢或合併結果
2. **精確的時間和關鍵字提取**：使用多層次的解析策略（相對日期 → YYYYMMDD → MMDD → 自然語言）
3. **靈活的白名單系統**：使用字典和列表格式，易於擴展和維護
4. **語義搜尋能力**：使用 embedding 向量進行語義相似度計算，可以找到關鍵字不完全匹配但語義相關的記錄

### 擴展建議

1. **添加新的關鍵字**：在 `_parse_query_filters()` 函數中的對應列表中添加
2. **添加新的事件類型**：在 `event_mapping` 字典中添加映射，並確保資料庫中有對應的 Boolean 欄位
3. **優化日期解析**：可以添加更多相對日期關鍵字（如「上個月」、「下個月」）
4. **調整分數計算**：可以根據業務需求調整分數計算公式，或添加額外的加分項（如關鍵字匹配加分）

---

**文件版本**：1.0  
**最後更新**：2025-12-30  
**維護者**：開發團隊

