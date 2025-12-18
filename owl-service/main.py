# ~/ASE/servers/owl_api.py  (merged: OWLv2 image + video)
# - 保留原本 /detect, /detect_file（不縮放）
# - 新增 /video_detect（上傳影片、每 N 秒取一幀、畫框、回存 mp4），/video_ui（簡易頁面可播放結果）
#
# 依賴：torch, transformers[owlv2], pillow, fastapi, uvicorn, requests, opencv-python
# 啟動：python -m uvicorn owl_api:app --host 127.0.0.1 --port 18001

import os, io, json, time, uuid, hashlib
from typing import List
import requests
from PIL import Image
import torch
from contextlib import nullcontext

import cv2, threading, queue
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# -------------------------
# Global torch/runtime tweaks
# -------------------------
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# -------------------------
# Config
# -------------------------
MODEL_ID = os.environ.get("OWL_MODEL_ID", "google/owlv2-base-patch16")
DEVICE = os.environ.get("OWL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
AMP_ON = (DEVICE == "cuda")
AMP_DTYPE = torch.float16
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "15"))
USER_AGENT = os.environ.get("HTTP_UA", "owlv2-fastapi/1.0 (+https://example)")
VERIFY_SSL_DEFAULT = True

# 影片輸出資料夾
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "./media")
os.makedirs(MEDIA_ROOT, exist_ok=True)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="OWLv2 FastAPI (image + video)", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/media", StaticFiles(directory=MEDIA_ROOT), name="media")

# -------------------------
# Load model & processor (once)
# 使用 FastAPI startup event 在後台線程載入模型，不阻塞服務啟動
# -------------------------
processor = None
model = None
MODEL_LOADED = False
import threading

def _load_model_background():
    """在後台線程中載入模型，不阻塞服務啟動"""
    global processor, model, MODEL_LOADED
    max_retries = 3
    retry_delay = 30
    
    for attempt in range(max_retries):
        try:
            print(f"正在載入模型 {MODEL_ID}... (嘗試 {attempt + 1}/{max_retries})")
            
            # 嘗試使用本地緩存
            cache_dir = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            print(f"使用緩存目錄: {cache_dir}")
            
            # 設置環境變數以使用本地緩存
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
            
            processor = Owlv2Processor.from_pretrained(
                MODEL_ID,
                cache_dir=cache_dir,
                local_files_only=False,  # 允許從網絡下載
                resume_download=True
            )
            model = Owlv2ForObjectDetection.from_pretrained(
                MODEL_ID,
                cache_dir=cache_dir,
                local_files_only=False,
                resume_download=True
            ).to(DEVICE).eval()
            
            if AMP_ON:
                try:
                    model = model.to(dtype=AMP_DTYPE)
                except Exception:
                    pass
            
            MODEL_LOADED = True
            print(f"✓ Model {MODEL_ID} loaded successfully")
            
            # 模型載入成功後進行 warmup
            try:
                _warmup()
            except Exception as e:
                print(f"⚠ Warmup failed: {e}")
            
            return  # 成功載入，退出函數
            
        except Exception as e:
            error_msg = str(e)
            print(f"⚠ 嘗試 {attempt + 1}/{max_retries} 失敗: {error_msg[:200]}")
            
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒後重試...")
                import time
                time.sleep(retry_delay)
            else:
                print(f"✗ 所有嘗試都失敗了。模型 {MODEL_ID} 無法載入。")
                print("可能的原因：")
                print("  1. 網絡無法連接到 Hugging Face (huggingface.co)")
                print("  2. 防火牆或代理設置問題")
                print("  3. DNS 解析失敗")
                print("\n解決方案：")
                print("  1. 檢查網絡連接：docker compose exec owl-api python -c \"import requests; requests.get('https://huggingface.co', timeout=10)\"")
                print("  2. 手動下載模型到本地緩存")
                print("  3. 使用其他模型類型（如 qwen 或 gemini）進行影片分析")
                MODEL_LOADED = False
                processor = None
                model = None

def _warmup():
    if not MODEL_LOADED or processor is None or model is None:
        return
    img = Image.new("RGB", (320, 320), (0, 0, 0))
    _inputs = processor(text=["a", "b"], images=img, return_tensors="pt").to(DEVICE)
    ctx = torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if AMP_ON else nullcontext()
    with torch.inference_mode():
        with ctx:
            _ = model(**_inputs)

@app.on_event("startup")
async def startup_event():
    """服務啟動時在後台線程載入模型"""
    print("✓ OWL API service starting... Model will load in background.")
    model_loader_thread = threading.Thread(target=_load_model_background, daemon=True)
    model_loader_thread.start()

# -------------------------
# IO helpers (image)
# -------------------------
def fetch_image_bytes(url: str, insecure_ssl: bool = False) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    verify = False if insecure_ssl else VERIFY_SSL_DEFAULT
    r = requests.get(url, timeout=HTTP_TIMEOUT, headers=headers, verify=verify, allow_redirects=True)
    r.raise_for_status()
    return r.content

def image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

# -------------------------
# Inference (no-resize)
# -------------------------
def infer_no_resize(image: Image.Image, prompts: List[str], score_threshold: float):
    if not MODEL_LOADED or processor is None or model is None:
        raise RuntimeError("Model not loaded. Cannot perform inference.")
    inputs = processor(text=prompts, images=image, return_tensors="pt").to(DEVICE)
    ctx = torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if AMP_ON else nullcontext()
    with torch.inference_mode():
        with ctx:
            outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=DEVICE)
    results = processor.post_process_grounded_object_detection(
        outputs, target_sizes=target_sizes, threshold=score_threshold
    )[0]
    dets = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = [float(v) for v in box]
        idx = int(label)
        name = prompts[idx] if 0 <= idx < len(prompts) else f"class_{idx}"
        dets.append({
            "label": name,
            "score": float(score),
            "bbox_xyxy": [x1, y1, x2, y2],
            "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        })
    return dets

# -------------------------
# Schemas (image routes)
# -------------------------
class DetectReq(BaseModel):
    image_url: str
    prompts: List[str]
    score_threshold: float = 0.15
    insecure_ssl: bool = False

# -------------------------
# Image routes
# -------------------------
@app.get("/")
def root():
    return {
        "ok": True, 
        "model_id": MODEL_ID, 
        "device": DEVICE, 
        "amp": AMP_ON, 
        "resize": False,
        "model_loaded": MODEL_LOADED
    }

@app.post("/detect")
def detect(req: DetectReq):
    t0 = time.time()
    data = fetch_image_bytes(req.image_url, insecure_ssl=req.insecure_ssl)
    img = image_from_bytes(data)
    dets = infer_no_resize(img, req.prompts, req.score_threshold)
    return {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "detections": dets
    }

@app.post("/detect_file")
async def detect_file(
    file: UploadFile = File(...),
    prompts_json: str = Form(...),
    score_threshold: float = Form(0.15),
):
    raw = await file.read()
    try:
        prompts = json.loads(prompts_json)
        if not isinstance(prompts, list):
            raise ValueError("prompts_json must be a JSON list of strings")
    except Exception as e:
        return {"error": f"invalid prompts_json: {e}"}
    t0 = time.time()
    img = image_from_bytes(raw)
    dets = infer_no_resize(img, prompts, score_threshold)
    return {
        "model_id": MODEL_ID,
        "device": DEVICE,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "detections": dets
    }

# -------------------------
# Video helpers (from gateway, in-proc call)
# -------------------------
def downscale_keep_ratio(bgr, target_short):
    if target_short is None or target_short <= 0:
        return bgr
    h, w = bgr.shape[:2]
    short = min(h, w)
    if short <= target_short:
        return bgr
    scale = target_short / float(short)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

def encode_jpg_bytes(bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()

def pil_from_bgr(bgr: np.ndarray) -> Image.Image:
    # 避免慢速轉換，僅於需要推論時轉成 PIL
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def draw_boxes(bgr, dets):
    img = bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["bbox_xyxy"])
        label = d["label"]; score = d.get("score", 0.0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img

# -------------------------
# Video routes
# -------------------------
@app.get("/video_ui", response_class=HTMLResponse)
def video_ui():
    return """
<!doctype html>
<meta charset="utf-8" />
<title>OWLv2 Video Detect</title>
<h3>上傳影片 → 每 N 秒取一幀 → 畫框 → 回傳可播放影片</h3>
<form id="f">
  <input type="file" name="file" accept="video/*" required><br/>
  <label>prompts</label>
  <input type="text" name="prompts" value="person,pedestrian,motorcycle,car,bus,scooter,truck" size="80"><br/>
  <label>score_threshold</label>
  <input type="number" step="0.01" name="score_threshold" value="0.15"><br/>
  <label>every_sec</label>
  <input type="number" step="0.1" name="every_sec" value="2.0"><br/>
  <button type="submit">開始分析</button>
</form>
<div id="status"></div>
<video id="v" controls style="max-width:80vw; border:1px solid #ccc"></video>
<script>
  const f = document.getElementById('f');
  const s = document.getElementById('status');
  const v = document.getElementById('v');
  f.onsubmit = async (e) => {
    e.preventDefault();
    s.textContent = "處理中...";
    v.removeAttribute('src');
    const fd = new FormData(f);
    const res = await fetch('/video_detect', { method:'POST', body: fd });
    const j = await res.json();
    if (j.error) { s.textContent = "失敗：" + j.error; return; }
    s.textContent = "完成！";
    v.src = j.video_url;
    v.load();
    v.play();
  };
</script>
"""

@app.post("/video_detect")
async def video_detect(
    file: UploadFile = File(...),
    prompts: str = Form("person,pedestrian,motorcycle,car,bus,scooter,truck"),
    score_threshold: float = Form(0.15),
    every_sec: float = Form(2.0),
    target_short: int = Form(720),
    jpg_quality: int = Form(80),
    concurrency: int = Form(2),
):
    # 檢查模型是否已加載
    if not MODEL_LOADED or processor is None or model is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model not loaded. Please check network connection and ensure the model can be downloaded from Hugging Face.",
                "model_id": MODEL_ID,
                "hint": "The OWL API service is running but the model failed to load due to network/DNS issues."
            }
        )
    
    # 驗證檔案型別
    if not (file.content_type or "").lower().startswith("video/"):
        raise HTTPException(status_code=400, detail=f"file must be video/*; got {file.content_type}")

    # 先寫到臨時檔
    src_path = os.path.join(MEDIA_ROOT, f"up_{uuid.uuid4().hex}.mp4")
    with open(src_path, "wb") as fobj:
        fobj.write(await file.read())

    # 讀影片
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return JSONResponse(status_code=400, content={"error": "無法開啟影片"})

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    base_fps = fps if fps > 0 else 30.0
    step = max(1, int(round(base_fps * every_sec)))

    # 先讀一張建立輸出
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        return JSONResponse(status_code=400, content={"error": "影片為空"})
    first_down = downscale_keep_ratio(first, target_short)
    h, w = first_down.shape[:2]
    out_name = f"ann_{uuid.uuid4().hex}.mp4"
    out_path = os.path.join(MEDIA_ROOT, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, max(1.0, 1.0 / every_sec), (w, h))

    # 重頭開始順序讀（避免隨機 seek 慢）
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prompts_list = [p.strip() for p in prompts.split(",") if p.strip()]
    q_in: queue.Queue = queue.Queue(maxsize=8 * int(concurrency))
    q_out: queue.Queue = queue.Queue(maxsize=8 * int(concurrency))
    STOP = object()

    def producer():
        idx = 0
        fid = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if idx % step == 0:
                bgr = downscale_keep_ratio(frame, target_short)
                q_in.put((fid, bgr))
                fid += 1
            idx += 1
        for _ in range(int(concurrency)):
            q_in.put(STOP)

    def worker():
        while True:
            item = q_in.get()
            if item is STOP:
                q_out.put(STOP)
                break
            fid, bgr = item
            # 直接在程序內呼叫推論（避免 HTTP 費時）
            pil = pil_from_bgr(bgr)
            dets = infer_no_resize(pil, prompts_list, float(score_threshold))
            q_out.put((fid, bgr, dets))

    def consumer():
        next_id = 0
        buf = {}
        done = 0
        while True:
            item = q_out.get()
            if item is STOP:
                done += 1
                if done == int(concurrency):
                    break
                continue
            fid, bgr, dets = item
            buf[fid] = (bgr, dets)
            while next_id in buf:
                bgr2, dets2 = buf.pop(next_id)
                ann = draw_boxes(bgr2, dets2)
                out.write(ann)
                next_id += 1

    t_prod = threading.Thread(target=producer, daemon=True); t_prod.start()
    workers = [threading.Thread(target=worker, daemon=True) for _ in range(int(concurrency))]
    for t in workers: t.start()
    consumer()
    for t in workers: t.join()
    t_prod.join()

    cap.release()
    out.release()

    return JSONResponse(content={
        "video_url": f"/media/{out_name}",
        "fps_input": fps,
        "every_sec": every_sec,
        "size": [w, h],
    })

