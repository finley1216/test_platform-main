# 偵測項目管理系統

## 📋 概述

本系統提供了動態的事件偵測項目管理功能，允許透過前端介面新增、修改、刪除偵測項目，系統會自動根據啟用的項目更新 `frame_prompt.md`。

## 🎯 主要功能

### 1. 動態事件管理
- ✅ 透過前端介面管理偵測項目
- ✅ 新增/編輯/刪除偵測項目
- ✅ 啟用/停用偵測項目
- ✅ 自動更新 `frame_prompt.md`

### 2. 資料庫架構更新
- ✅ `DetectionItem` 模型：管理偵測項目
- ✅ `Summary` 模型：改用動態事件記錄
  - `events_en`: 英文事件名稱（逗號分隔）
  - `events_zh`: 中文事件名稱（逗號分隔）
  - `events_json`: JSON 格式的詳細事件資訊

### 3. API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/detection-items` | GET | 列出所有偵測項目 |
| `/detection-items/{id}` | GET | 獲取單個偵測項目 |
| `/detection-items` | POST | 創建偵測項目 |
| `/detection-items/{id}` | PUT | 更新偵測項目 |
| `/detection-items/{id}` | DELETE | 刪除偵測項目 |
| `/detection-items/regenerate-prompt` | POST | 重新生成 prompt |
| `/detection-items/preview-prompt/content` | GET | 預覽 prompt 內容 |

## 🚀 快速開始

### 1. 初始化資料庫

首次使用時，請執行初始化腳本來創建預設的偵測項目：

```bash
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main/backend/src
python init_detection_items.py
```

這會創建以下預設項目：
- 水災 (water_flood)
- 火災 (fire)
- 異常著裝/遮臉入場 (abnormal_attire_face_cover_at_entry)
- 人員倒地不起 (person_fallen_unmoving)
- 併排停車/車道阻塞 (double_parking_lane_block)
- 非管制區吸菸 (smoking_outside_zone)
- 聚眾逗留 (crowd_loitering)
- 突破安全門 (security_door_tamper)

### 2. 啟動服務

```bash
# 後端
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main/backend
python src/start.py

# 前端
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main/frontend
npm start
```

### 3. 使用前端介面管理

1. 登入系統（需要 Admin 權限）
2. 點擊頂部導航欄的「偵測項目管理」按鈕
3. 在彈出的視窗中管理偵測項目

## 📖 使用方式

### 新增偵測項目

1. 點擊「➕ 新增偵測項目」
2. 填寫以下資訊：
   - **唯一識別名稱**：用於資料庫記錄（例如：`fire`）
   - **英文名稱**：用於 prompt 和 API（例如：`fire`）
   - **中文名稱**：用於顯示（例如：`火災`）
   - **偵測標準描述**：用於 prompt 的判斷標準
   - **啟用狀態**：勾選以啟用此項目
3. 點擊「儲存」

### 編輯偵測項目

1. 在偵測項目列表中找到要編輯的項目
2. 點擊「✏️」按鈕
3. 修改資訊後點擊「儲存」

### 啟用/停用偵測項目

- 點擊項目旁的「⏸️」（停用）或「▶️」（啟用）按鈕
- 只有啟用的項目會出現在 `frame_prompt.md` 中

### 刪除偵測項目

1. 點擊項目旁的「🗑️」按鈕
2. 確認刪除

### 預覽 Prompt

點擊「👁️ 預覽 Prompt」可以查看根據當前設定生成的 prompt 內容，而不實際寫入文件。

### 重新生成 Prompt

點擊「🔄 重新生成 Prompt」可以手動觸發 `frame_prompt.md` 的更新。

## 🔄 自動更新機制

每當進行以下操作時，系統會自動更新 `frame_prompt.md`：
- ✅ 新增偵測項目
- ✅ 編輯偵測項目
- ✅ 刪除偵測項目
- ✅ 啟用/停用偵測項目

## 📊 資料庫遷移

如果您已有舊的資料庫，需要進行遷移：

```sql
-- 在 summaries 表中添加新欄位
ALTER TABLE summaries ADD COLUMN events_en TEXT;
ALTER TABLE summaries ADD COLUMN events_zh TEXT;
ALTER TABLE summaries ADD COLUMN events_json TEXT;

-- 可選：移除舊的 boolean 欄位（如果確定不需要了）
-- ALTER TABLE summaries DROP COLUMN water_flood;
-- ALTER TABLE summaries DROP COLUMN fire;
-- ... 其他欄位 ...

-- 創建 detection_items 表
CREATE TABLE detection_items (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    name_en VARCHAR(100) NOT NULL,
    name_zh VARCHAR(100) NOT NULL,
    description TEXT,
    is_enabled BOOLEAN DEFAULT TRUE,
    alert_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_detection_items_name ON detection_items(name);
```

或者直接使用 SQLAlchemy 的 `create_all()`：

```python
from src.database import engine
from src.models import Base

Base.metadata.create_all(bind=engine)
```

## 🎨 前端組件

### DetectionItemsModal

位於 `frontend/src/components/DetectionItemsModal.js`

主要功能：
- 列出所有偵測項目
- 新增/編輯/刪除項目
- 啟用/停用項目
- 預覽和重新生成 prompt

### Navbar 更新

在頂部導航欄添加了「偵測項目管理」按鈕（僅 Admin 可見）。

## 🔒 權限控制

- 偵測項目管理功能僅對 **Admin** 用戶開放
- 前端會檢查 `isAdmin` 狀態來顯示/隱藏相關按鈕
- 後端 API 應配合實施權限驗證（建議添加）

## 📝 範例：偵測項目結構

```json
{
  "name": "fire",
  "name_en": "fire",
  "name_zh": "火災",
  "description": "可見火焰或持續濃煙竄出",
  "is_enabled": true
}
```

## 🔧 故障排除

### 問題 1：Prompt 沒有更新

**解決方案**：
1. 檢查後端日誌是否有錯誤
2. 手動點擊「重新生成 Prompt」
3. 確認 `backend/prompts/frame_prompt.md` 文件權限

### 問題 2：前端無法載入偵測項目

**解決方案**：
1. 檢查後端 API 是否正常運行
2. 檢查瀏覽器控制台的錯誤訊息
3. 確認 API Key 是否有效

### 問題 3：資料庫連接錯誤

**解決方案**：
1. 確認 PostgreSQL 服務正在運行
2. 檢查 `src/config.py` 中的資料庫連接設定
3. 執行資料庫遷移腳本

## 📚 相關文件

- `backend/src/models.py` - 資料庫模型定義
- `backend/src/api/detection_items.py` - API 路由
- `frontend/src/components/DetectionItemsModal.js` - 前端組件
- `frontend/src/services/api.js` - API 服務
- `backend/prompts/frame_prompt.md` - 動態生成的 prompt 文件

## 🎯 未來改進

- [ ] 添加批次匯入/匯出功能
- [ ] 支援偵測項目的排序
- [ ] 添加偵測項目的統計圖表
- [ ] 支援偵測項目的標籤分類
- [ ] 添加偵測項目的版本控制

## 💡 技術細節

### Prompt 生成邏輯

系統會根據啟用的偵測項目自動生成 JSON 格式和判斷標準：

```python
def generate_frame_prompt(db: Session) -> str:
    items = db.query(DetectionItem).filter(
        DetectionItem.is_enabled == True
    ).order_by(DetectionItem.id).all()
    
    # 生成 JSON 欄位
    event_fields = [f'"{item.name_en}": false,' for item in items]
    
    # 生成判斷標準
    event_standards = [
        f"{idx}) {item.name_en}（{item.name_zh}）：{item.description} → **true**。"
        for idx, item in enumerate(items, 1)
    ]
    
    # 組合成完整 prompt...
```

### 事件記錄格式

在 Summary 表中，事件以三種格式記錄：

1. **events_en**：`"fire, water_flood"`
2. **events_zh**：`"火災, 水災"`
3. **events_json**：
```json
[
  {"name_en": "fire", "name_zh": "火災", "detected": true},
  {"name_en": "water_flood", "name_zh": "水災", "detected": true}
]
```

## 📧 支援

如有問題，請查看：
- 系統日誌
- 瀏覽器控制台
- API 文件（`/docs`）

---

**版本**：v2.4  
**最後更新**：2026-01-11  
**作者**：ASE Team
