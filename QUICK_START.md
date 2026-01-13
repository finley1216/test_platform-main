# 快速開始：偵測項目管理系統

## 🚀 三步驟啟動

### 步驟 1：資料庫設置

```bash
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main/backend/src

# 如果是新的資料庫：
python init_detection_items.py

# 如果是從舊版本升級：
python migrate_to_dynamic_events.py
python init_detection_items.py
```

### 步驟 2：啟動服務

```bash
# 終端 1：啟動後端
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main/backend
python src/start.py

# 終端 2：啟動前端（如果需要）
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main/frontend
npm start
```

### 步驟 3：使用前端介面

1. 開啟瀏覽器訪問前端
2. 使用 Admin 帳號登入
3. 點擊頂部導航欄的「**偵測項目管理**」按鈕
4. 開始管理您的偵測項目！

## 📱 介面預覽

### 偵測項目管理視窗

```
┌─────────────────────────────────────────────────────┐
│ 偵測項目管理                                    ×   │
├─────────────────────────────────────────────────────┤
│                                                     │
│ [➕ 新增偵測項目] [👁️ 預覽 Prompt] [🔄 重新生成]  │
│                                                     │
│ 偵測項目列表 (8 個)                                │
│ ┌─────────────────────────────────────────────┐   │
│ │ 火災 (fire)                      ⏸️ ✏️ 🗑️  │   │
│ │ 可見火焰或持續濃煙竄出                        │   │
│ └─────────────────────────────────────────────┘   │
│ ┌─────────────────────────────────────────────┐   │
│ │ 水災 (water_flood)               ⏸️ ✏️ 🗑️  │   │
│ │ 車輛明顯濺水 / 標線被水覆蓋...                │   │
│ └─────────────────────────────────────────────┘   │
│ ...                                                │
│                                                     │
│                                        [關閉]       │
└─────────────────────────────────────────────────────┘
```

## 💡 常用操作

### 新增自定義事件

1. 點擊「➕ 新增偵測項目」
2. 填寫：
   ```
   唯一識別名稱：explosion
   英文名稱：explosion
   中文名稱：爆炸
   描述：可見爆炸火光或聽到爆炸聲響
   ☑ 啟用此偵測項目
   ```
3. 點擊「儲存」

### 暫時停用某個事件

- 點擊事件旁的「⏸️」按鈕即可停用
- 停用的事件不會出現在 `frame_prompt.md` 中

### 預覽生成的 Prompt

- 點擊「👁️ 預覽 Prompt」
- 查看會生成什麼樣的 prompt 內容
- 不會實際寫入文件

## 🔍 驗證安裝

### 1. 檢查 API 端點

訪問：`http://your-backend:8080/docs`

應該能看到以下新端點：
- `/detection-items`
- `/detection-items/regenerate-prompt`
- 等等

### 2. 檢查資料庫

```sql
-- 應該看到 detection_items 表
SELECT * FROM detection_items;

-- 應該看到新的欄位
SELECT events_en, events_zh FROM summaries LIMIT 1;
```

### 3. 檢查前端

- 登入後應該在頂部導航欄看到「偵測項目管理」按鈕
- 點擊後應該能開啟管理視窗

## 🐛 常見問題

### 問題：按鈕沒有出現

**檢查**：
- 確認已使用 Admin 帳號登入
- 檢查瀏覽器控制台是否有錯誤
- 確認前端已重新編譯

### 問題：無法載入偵測項目

**檢查**：
- 後端是否正常運行
- 檢查 `/detection-items` API 是否可訪問
- 執行 `init_detection_items.py` 創建初始資料

### 問題：Prompt 沒有更新

**解決**：
- 點擊「重新生成 Prompt」按鈕
- 檢查 `backend/prompts/frame_prompt.md` 文件權限
- 查看後端日誌

## 📁 檔案位置

```
test_platform-main/
├── backend/
│   ├── src/
│   │   ├── models.py                        # 資料庫模型
│   │   ├── api/
│   │   │   └── detection_items.py           # 偵測項目 API
│   │   ├── init_detection_items.py          # 初始化腳本
│   │   └── migrate_to_dynamic_events.py     # 遷移腳本
│   └── prompts/
│       └── frame_prompt.md                   # 動態生成的 Prompt
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── DetectionItemsModal.js       # 管理介面
│       │   └── Navbar.js                    # 導航欄
│       └── services/
│           └── api.js                        # API 服務
├── DETECTION_ITEMS_README.md                # 詳細文件
└── QUICK_START.md                            # 本文件
```

## 🎯 下一步

- ✅ 開始新增您自己的偵測項目
- ✅ 嘗試啟用/停用不同的事件
- ✅ 預覽和測試生成的 Prompt
- ✅ 執行影片分析，查看新的事件記錄格式

## 📞 需要幫助？

查看詳細文件：`DETECTION_ITEMS_README.md`

---

**祝使用愉快！** 🎉
