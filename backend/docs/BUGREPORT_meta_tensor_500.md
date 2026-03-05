# 問題報告：segment_pipeline_batch 500 錯誤（meta tensor）

## 現象

- 客戶端：`test_segment_pipeline_rtsp_batch.py` 每 4 段送 API，多數請求回傳 **HTTP 500**。
- 後端 log 出現兩類錯誤：
  1. **Qwen 載入失敗**：`NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.`
  2. **ReID (torchvision ResNet50) 載入失敗**：同上「meta tensor」錯誤，且「copying from a non-meta parameter in the checkpoint to a meta parameter in the current model」。

---

## 原因說明

### 1. Qwen2.5-VL 的 meta tensor 錯誤

- 程式使用 `Qwen2_5_VLForConditionalGeneration.from_pretrained(..., device_map="auto")`。
- `device_map="auto"` 會經由 **accelerate** 做分散載入，過程中會先用 **meta device**（空的 placeholder 張量）建模型再 dispatch 到 GPU。
- 在目前環境下，dispatch 時呼叫的是 `model.to(device)`，而 **meta 張量不能直接用 `.to(device)` 複製**，必須用 `to_empty()`，因此拋出 `NotImplementedError`。

### 2. ReID (torchvision ResNet50) 的 meta tensor 錯誤

- 同一 process 內，若先有其它程式（例如載入 Qwen）透過 accelerate 使用 **meta device**，可能影響後續在其它 thread 建立的模型。
- 當 ReID 使用 `models.resnet50(pretrained=True)` 時，若當時預設/環境處於 meta 狀態，模型參數會變成 meta 張量；之後 `model.to(cuda)` 就會觸發與 Qwen 相同的「Cannot copy out of meta tensor」錯誤。
- log 中「copying from a non-meta parameter in the checkpoint to a meta parameter in the current model」表示：checkpoint 是正常權重，但**當前模型**的參數是 meta，因此無法正常載入。

---

## 已採取的修復

### 1. Qwen 載入（`backend/src/utils/qwen_hf_utils.py`）

- 將 `device_map="auto"` 改為 **`device_map="cuda:0"`**，整顆模型直接放在單一 GPU，不走 accelerate 的 meta + dispatch 流程。
- 加上 **`low_cpu_mem_usage=False`**，避免使用 meta device 的省記憶體載入路徑。
- 預期效果：Qwen 不再出現 meta tensor 的 `NotImplementedError`；單卡 31GB 足以容納 Qwen2.5-VL-7B。

### 2. ReID ResNet50 載入（`backend/src/core/model_loader.py`）

- 在 **CPU** 上建立並載入權重，再 `.to(device)`，避免受其它 meta device 狀態影響：
  - 使用 `with torch.device("cpu"):` 包住 `models.resnet50(...)`。
- 使用新版 API：`weights=models.ResNet50_Weights.IMAGENET1K_V1`；若無此屬性則 fallback 到 `pretrained=True`，以相容舊版 torchvision。
- 預期效果：ReID 不再出現「meta parameter in the current model」的錯誤。

---

## 建議驗證步驟

1. 重啟 backend（建議連同 volume 一併確認）  
   ```bash
   cd test_platform-main && docker compose up -d --force-recreate backend
   ```
2. 再跑一次 batch 測試（例如每 4 段送一次）：  
   ```bash
   python test_platform-main/backend/scripts/test_segment_pipeline_rtsp_batch.py \
     --base http://127.0.0.1:3000/api --streams 10 --duration 1 --batch-size 4
   ```
3. 檢查後端 log：不應再出現「Cannot copy out of meta tensor」或「copying from a non-meta parameter in the checkpoint to a meta parameter」；若有 OOM，可再調小 `--batch-size` 或確認 `WORKERS=1`。

---

## 摘要

| 項目       | 原因概要 | 修復方式 |
|------------|----------|----------|
| Qwen 500   | `device_map="auto"` 使用 meta device，`.to(device)` 不支援 | 改為 `device_map="cuda:0"` + `low_cpu_mem_usage=False` |
| ReID 500   | 同 process 內 meta 狀態導致 ResNet50 建在 meta 上 | CPU 上建模型並載入權重後再 `.to(device)`，並相容舊版 torchvision |
