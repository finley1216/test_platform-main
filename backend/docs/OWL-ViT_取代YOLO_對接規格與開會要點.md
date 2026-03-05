# OWL-ViT 取代 YOLO 對接規格

---

## 一、必須提供的介面

硬體方需提供可取代 YOLO 的模型物件，支援以下方法：

| 方法 | 說明 |
|------|------|
| `model.to(device)` | device 為 `"cuda"` 或 `"cpu"`，可為空實作 |
| `model.set_classes(labels_list)` | `labels_list` 如 `["person","car"]`，設定偵測類別 |
| `model.predict(images, conf=0.25, verbose=False)` | 輸入 `List[np.ndarray]`（BGR），回傳每幀偵測結果 |

**輸出不變**：每幀結果要有 `res.boxes`，每個 `box` 需有 `cls`（類別索引）、`conf`（信心）、`xyxy`（`[x1,y1,x2,y2]`）。`box.cls` 必須對應 `labels_list` 的索引順序。

---

## 二、開會需確認的事項

1. **載入方式**：同進程 Python 載入，還是 HTTP/gRPC API？離線時權重如何載入？
2. **批次推論**：是否支援一次送多幀？
3. **輸入格式**：要 BGR 還是 RGB？（RGB 我方可用 `cv2.cvtColor` 轉換）是否有解析度限制或自動 resize？
4. **權重路徑**：是否可用環境變數（如 `OWL_WEIGHTS_PATH`）指定？單有權重檔不夠，還需有載入與 predict 的程式或套件。
5. **Adapter 實作**：我方改 `model_loader.py`，還是硬體方提供相容類別？若硬體方無法提供可用的 predict 介面，我方無法直接使用。

### 第 4、5 點說明

**第 4 點：權重路徑與「權重 + 程式」缺一不可**

- 權重檔（`.pt`、`.safetensors` 等）只是模型參數，本身無法執行。
- 必須同時有**載入邏輯**（從路徑讀檔、建立模型）與**predict 介面**（或等同的推理函式）才能使用。
- 對照 YOLO：我們有 `ultralytics` 套件（提供載入 + predict），再加上權重檔，才能直接使用。

**第 5 點：Adapter 責任與「沒有 predict 就無法用」**

- Adapter 是指：把 OWL-ViT 包成符合 YOLO 介面（`to`、`set_classes`、`predict`）的類別或 API 客戶端。
- **硬體方提供**：硬體方給一個 Python 類別或 HTTP 客戶端，我方在 `model_loader.py` 呼叫即可。
- **我方自行改**：我方用 transformers 等套件載入 OWL-ViT，再包一層符合介面的 Adapter。
- 若硬體方**無法**提供可用的 predict 介面（無論是類別還是 API），我方無法直接串接；若要使用，只能由我方投入開發，自行實作 Adapter。

---

