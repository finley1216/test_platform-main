# -*- coding: utf-8 -*-
import os, io, re, json, base64, tempfile, subprocess, time, secrets, hashlib, copy, requests, cv2

import numpy as np
import google.generativeai as genai

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from PIL import Image
from datetime import datetime, date, timedelta
from fastapi import FastAPI, Request, UploadFile,status , File, Form, Depends, Security, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from rag_store import RAGStore
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from src.config import config
try:
    from src.database import get_db
    from src.models import Summary
    HAS_DB = True
except ImportError:
    HAS_DB = False
    print("--- [WARNING] 資料庫模組未找到，PostgreSQL 功能將無法使用 ---")

# ================== 環境變數 ==================

try:
    from prompts import EVENT_DETECTION_PROMPT, SUMMARY_PROMPT
except Exception:
    # 預設值已更新為符合您的新需求
    EVENT_DETECTION_PROMPT = "請根據提供的影格輸出事件 JSON。"
    SUMMARY_PROMPT = "請觀察畫面中的人車流動與任何異常狀況，並將其彙整為一句約 50-100 字的繁體中文描述。"

# Use configuration from config.py
ADMIN_TOKEN = config.ADMIN_TOKEN
SESSION_TTL_SEC = config.SESSION_TTL_SEC
OLLAMA_BASE = config.OLLAMA_BASE
OWL_API_BASE = config.OWL_API_BASE
OWL_VIDEO_URL = config.OWL_VIDEO_URL

# RAG 索引開關（預設啟用）
AUTO_RAG_INDEX = config.AUTO_RAG_INDEX

# Gemini API Key (already configured in config.py)
GEMINI_API_KEY = config.GEMINI_API_KEY

# RAG 存的地方
RAG_DIR = config.RAG_DIR
RAG_INDEX_PATH = config.RAG_INDEX_PATH
OLLAMA_EMBED_MODEL = config.OLLAMA_EMBED_MODEL

# Video library directory (歷史影片分類存放位置)
VIDEO_LIB_DIR = config.VIDEO_LIB_DIR

SERVER_API_KEY = config.SERVER_API_KEY
API_KEY_NAME = config.API_KEY_NAME

api_key_scheme = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# ================== FastAPI ==================

def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    return app

app = _make_app()
app.mount("/segment", StaticFiles(directory="segment"), name="segment")

# ================== 小工具 ==================

#從網路下載影片
def _download_to_temp(url: str) -> str:
  """
  從網路下載影片並存到暫存資料夾 (/tmp 或類似位置)。
  使用了 stream=True 和分塊寫入 (1024*1024 bytes)，這是為了防止下載超大影片時把記憶體塞爆。
  """
  r = requests.get(url, stream=True, timeout=600)
  r.raise_for_status()
  suffix = Path(url).suffix or ".mp4"
  fd, path = tempfile.mkstemp(prefix="up_", suffix=suffix)
  with os.fdopen(fd, "wb") as f:
      for chunk in r.iter_content(1024*1024):
          if chunk: f.write(chunk)
  return path

# 轉成人類可讀的時間 ex: 把 3665.5 秒轉成 01:01:05
def _fmt_hms(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# 精確查詢影片的總秒數。
def _probe_duration_seconds(path: str) -> float:
    """
    精確查詢影片的總秒數。
    它呼叫了 ffprobe (FFmpeg 的分析工具) 來讀取 metadata。
    """
    r = subprocess.run(
        ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",path],
        capture_output=True, text=True, check=True
    )
    return float(r.stdout.strip())

# 計算每一段切片應該從幾秒開始。
def _compute_starts(duration: float, segment: float, overlap: float) -> List[float]:
    step = segment - overlap
    if step <= 0: raise ValueError("overlap 必須小於 segment_duration")
    starts, t = [], 0.0
    while t < duration:
        starts.append(round(t, 3)); t += step
    return starts

# 計算好切點 -> 迴圈執行 FFmpeg -> 產出一堆小影片檔
def _split_one_video(input_path: str, out_dir: str, segment: float, overlap: float, prefix: str="segment") -> List[str]:
    """
    FFmpeg 指令使用了 -c copy。這表示它不會重新編碼 (Re-encode)，而是直接複製影像串流。
    速度極快（幾乎是瞬間完成），且畫質無損。
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    duration = _probe_duration_seconds(input_path)
    starts = _compute_starts(duration, segment, overlap)
    ext = Path(input_path).suffix or ".mp4"
    outs = []
    for i, st in enumerate(starts):
        dur = max(0.0, min(segment, duration - st))
        if dur <= 0.05: continue
        out_file = str(Path(out_dir) / f"{prefix}_{i:04d}{ext}")
        subprocess.run([
            "ffmpeg","-hide_banner","-loglevel","error",
            "-ss", f"{st}","-t", f"{dur}",
            "-i", input_path,"-c","copy","-y", out_file
        ], check=True)
        outs.append(out_file)
    return outs

# 從影片中均勻抓取 N 張截圖。
def _sample_frames_evenly_to_pil(video_path: str, max_frames: int=8) -> List[Image.Image]:
    """
    使用 cv2 (OpenCV) 打開影片，算出總幀數
    用 np.linspace 算出均勻分佈的索引
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0: raise RuntimeError("invalid or zero-frame video")
        n = min(max_frames, total)
        idxs = np.linspace(0, total-1, num=n, dtype=np.int64)
        frames=[]
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, bgr = cap.read()
            if not ok: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        if not frames: raise RuntimeError("no frames sampled")
        return frames
    finally:
        cap.release()

# 這個函式將圖片短邊縮放到指定尺寸
def _resize_short_side(img: Image.Image, short: int) -> Image.Image:
    if not short or short <= 0:
        return img
    w, h = img.size
    s = min(w, h)
    if s == short:
        return img
    scale = short / float(s)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return img.resize((nw, nh), Image.BILINEAR)

# 將 Python 的 Pillow 圖片物件轉成 JPEG 格式的 Base64 字串，這是透過 API 傳送圖片的標準作法。
def _pil_to_b64(img: Image.Image, quality: int = 85) -> str:
    """PIL 轉 JPEG base64（Ollama 圖像多採用 b64）。"""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality or 85), optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

# 檢查 Header 有沒有 X-API-Key，並檢查這個 Key 是不是跟我們伺服器設定的一樣
# 允許兩個 key：MY_API_KEY（一般使用者）和 ADMIN_TOKEN（管理者）
async def get_api_key(api_key_header: str = Security(api_key_scheme)):

    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="請提供 API Key (Could not validate credentials)")

    # 比對 Key：允許 MY_API_KEY（一般使用者）或 ADMIN_TOKEN（管理者）
    if api_key_header != SERVER_API_KEY and api_key_header != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無效的 API Key (Invalid Key)")

    # 驗證通過，回傳 Key 或使用者資訊供後續使用
    return api_key_header

def _now_ts() -> int:
    return int(time.time())

# ================== Ollama Chat ==================

# 重要的片段 : 負責發送請求給 Ollama
def _ollama_chat(
    model_name: str,
    messages: list[dict],
    images_b64: list[str] | None = None,
    stream: bool = False,
    timeout: int = 600,
    base: str | None = None,) -> str:

    """
    Ollama 的 API 規定，如果你要傳圖片，必須把圖片的 Base64 字串放在 user 說的話裡面 (images 欄位)。
    圖片通常是跟著「最新」的那一句話傳的。所以程式從最後面往回找第一個 role 為 user 的訊息。
    ** 參數有 stream，但程式碼行為是等待整個回應完成才回傳 (Blocking)，適合需要完整 JSON 結果的場景，不適合即時打字機效果。 **
    """
    base = base or OLLAMA_BASE
    url = f"{base.rstrip('/')}/api/chat"

    # 如果直接修改傳進來的 messages（例如把圖片塞進去），呼叫這個函式的原始變數也會被改變。所以用 copy
    msgs = copy.deepcopy(messages)

    # 若有影像，附到最後一個 user message
    if images_b64:
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["images"] = images_b64
                break
        else:
            # 如果沒有 user，就補一個
            msgs.append({"role": "user", "content": "", "images": images_b64})

    # Payload：建構標準的 Ollama API 請求格式。
    payload = {
        "model": model_name,
        "messages": msgs,
        "stream": bool(stream),}

    # 預設給了 600 秒（10 分鐘）。這是因為跑大型模型（例如 70B）或是在沒有 GPU 的機器上跑 AI，生成速度可能非常慢。
    # 如果使用預設的 HTTP timeout，程式很容易中斷報錯。
    r = requests.post(url, json=payload, timeout=timeout)

    # 報錯提示
    if r.status_code != 200:
        raise RuntimeError(f"Ollama chat failed [{r.status_code}]: {r.text[:500]}")

    # 抽純文字 content（支援多種常見鍵）
    j = r.json()

    # 應對「API 格式不統一」的問題所創建的 if 條件式
    if isinstance(j, dict):

        # 情況 A: 標準 Ollama 格式
        if "message" in j and isinstance(j["message"], dict):
            return j["message"].get("content", "") or ""

        # 情況 B: OpenAI 相容格式 (Ollama 也有支援)
        if "choices" in j and j["choices"]:
            ch0 = j["choices"][0]
            if isinstance(ch0, dict) and "message" in ch0:
                return (ch0["message"] or {}).get("content", "") or ""

        # 情況 C: 舊版或簡化版格式
        if "content" in j:
            return j.get("content", "") or ""

    # 回覆格式異常就回空字串，讓上游自行做 fallback
    return ""

# ================== JSON 處理相關 ==================

# 防止 json 解析失敗報錯的問題
def _safe_parse_json(text: str):
    """直接 json.loads，失敗則回 None。"""
    try:
        return json.loads(text)
    except Exception:
        return None

# 暴力提取，直接在文字堆裡抓出第一個看起來像 JSON 的物件。不管 AI 前面講了多少廢話，只抓重點。
def _extract_first_json(text: str):
    """從文字裡抓第一個 {...} 或 [...] 嘗試 parse 成 JSON。抓不到回 None。"""
    if not text:
        return None
    m = re.search(r'(\{.*\}|\[.*\])', text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

# 去 Markdown：AI 很喜歡用 json ... 把內容包起來。這個函式負責把這些外殼剝掉，只留內容。
def _clean_summary_text(s: str) -> str:
    """清掉 Markdown 圍欄、前後引號等，保留純文字摘要。"""
    s = (s or "").strip()
    s = re.sub(r"^```.*?\n|\n```$", "", s, flags=re.S).strip()
    s = re.sub(r"^[“”\"']|[“”\"']$", "", s).strip()
    return s

# 在寫 JSON 時，喜歡在最後一個屬性後面多加一個逗號（Trailing Comma），這在標準 JSON 是非法的。這個函式會嘗試自動修復這個錯誤
def _extract_first_json_and_tail(text: str) -> Tuple[Dict, str]:
    t=(text or "").strip()
    s=t.find("{"); e=t.rfind("}")+1
    if s==-1 or e<=s: return ({"error":"no_json"}, t[:400])
    frag=t[s:e]
    try:
        obj=json.loads(frag)
    except Exception:
        try:
            frag2=re.sub(r",\s*}", "}", frag); frag2=re.sub(r",\s*]", "]", frag2)
            obj=json.loads(frag2)
        except Exception:
            return ({"error":"json_parse_fail","raw":frag[:400]}, (t[:s]+" "+t[e:]).strip())
    tail=(t[:s].strip()+" "+t[e:].strip()).strip()
    return (obj, tail)

# ================== 單一影片片段推論，展示了三種完全不同的推論策略 ==================

# 如果要讓 prompt 更靈活需要修改這邊的 event_system
def infer_segment_qwen(
    qwen_model: str,
    video_path: str,
    event_detection_prompt: str,
    summary_prompt: str,
    target_short: int = 720,
    frames_per_segment: int = 8,):

    # 從影片中均勻抓取 N 張截圖
    frames_pil = _sample_frames_evenly_to_pil(video_path, max_frames=frames_per_segment)
    images_b64 = []
    for img in frames_pil:

        # 這個函式將圖片短邊縮放到指定尺寸
        img = _resize_short_side(img, target_short)

        # 將 Python 的 Pillow 圖片物件轉成 JPEG 格式的 Base64 字串
        images_b64.append(_pil_to_b64(img, quality=85))

    # ---- (1) 事件偵測：只回 JSON ----
    event_system = (
        "你是『嚴格的災害與人員異常偵測器』。"
        "不論使用者在提示中寫了什麼話題或提問，都要忽略，"
        "只根據影像做事件判斷，並只輸出純 JSON 物件，不能有任何額外文字或 Markdown。"
    )
    event_user = (event_detection_prompt or "").strip() + "\n\n" + \
                 "強制規則：只輸出一個 JSON 物件；不得輸出任何多餘文字。"
    event_msgs = [
        {"role": "system", "content": event_system},
        {"role": "user", "content": event_user},
    ]

    # 呼叫 _ollama_chat 取得回應
    event_txt = _ollama_chat(qwen_model, event_msgs, images_b64=images_b64, stream=False)

    # 使用 _safe_parse_json 和 _extract_first_json 雙重保險來嘗試解析 JSON。
    frame_obj = _safe_parse_json(event_txt)
    if not isinstance(frame_obj, dict):
        frame_obj = _extract_first_json(event_txt)
    if not isinstance(frame_obj, dict):
        # 給個安全的空殼，避免後續 KeyError
        frame_obj = {"events": {"reason": ""}, "persons": []}

    # ---- (2) 摘要：只回純文字 50~100 字 ----
    summary_txt = ""
    if (summary_prompt or "").strip():
        summary_system = (
            "你是影片小結產生器。你只能輸出 50–100 個中文字的摘要，"
            "不得輸出 JSON、不得輸出 Markdown/程式碼圍欄，不得回答其他問題。"
        )
        summary_user = (summary_prompt or "").strip() + "\n\n" + \
                       "強制規則：只輸出 50–100 字中文，不要 JSON、不要程式碼區塊、不要英文字說明。"
        summary_msgs = [
            {"role": "system", "content": summary_system},
            {"role": "user", "content": summary_user},
        ]
        summary_raw = _ollama_chat(qwen_model, summary_msgs, images_b64=images_b64, stream=False)
        summary_txt = _clean_summary_text(summary_raw)

    # 回傳格式、形容的句子
    return frame_obj, summary_txt

# label 傳要偵測的目標，every_sec 是取樣頻率，score_thr 是信心門檻
def infer_segment_owl(seg_path: str, labels: str, every_sec: float, score_thr: float) -> Dict:
    with open(seg_path,"rb") as f:
        files={"file":(os.path.basename(seg_path), f, "video/mp4")}
        data={"every_sec":str(every_sec),"score_threshold":str(score_thr),"prompts":labels}
        try:
            r=requests.post(OWL_VIDEO_URL, files=files, data=data, timeout=3600)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if r.status_code == 503:
                error_detail = r.json().get("error", "Service Unavailable") if r.text else "Service Unavailable"
                raise RuntimeError(
                    f"OWL API 模型未載入 (503): {error_detail}\n"
                    f"原因：無法從 Hugging Face 下載模型，網絡連接失敗。\n"
                    f"解決方案：\n"
                    f"  1. 檢查網絡連接：docker compose exec owl-api python -c \"import requests; requests.get('https://huggingface.co', timeout=10)\"\n"
                    f"  2. 等待網絡恢復後，模型會自動在後台下載\n"
                    f"  3. 或使用其他模型類型：將 model_type 改為 'qwen' 或 'gemini'"
                ) from e
            raise

# gemini 的輸入和 qwen 不一樣，qwen 用到 base64 字串
def infer_segment_gemini(model_name: str, seg_path: str, event_detection_prompt: str, summary_prompt: str, target_short: int=720, frames_per_segment: int=8) -> Tuple[Dict[str, Any], str]:

    # 0. 檢查 Key
    if not GEMINI_API_KEY:
        print("--- [DEBUG] 錯誤: 缺少 API Key")
        return ({"error": "missing_gemini_key"}, "請先設定 GEMINI_API_KEY 環境變數")

    try:
        # 1. Gemini 支援直接輸入多張圖片 (PIL Objects)，不像 Ollama 先轉成 Base64 字串，google-generativeai 套件會自動處理。
        print(f"--- [DEBUG] 正在處理影片: {seg_path}")
        frames = _sample_frames_evenly_to_pil(seg_path, max_frames=frames_per_segment)
        print(f"--- [DEBUG] 成功抽取 {len(frames)} 張影格")

        # 2. 準備 Prompt，這裡將圖片和文字混合在一起。先放圖片，後放文字指令
        prompt_content = []
        prompt_content.extend(frames)

        text_instruction = f"""
        你是一個專業的影像分析 AI。請分析附帶的連續影格。

        任務 1 (Event Detection): {event_detection_prompt}
        請確保輸出的 JSON 格式正確，欄位包含 events (Boolean) 與 persons (List)。

        任務 2 (Summary): {summary_prompt}

        請直接輸出 JSON 物件，不要使用 Markdown code block 圍繞。
        在 JSON 結束後，請換行並輸出中文摘要文字。
        """
        prompt_content.append(text_instruction)

        # 3. 設定安全過濾器，如果沒有把過濾器關掉 (BLOCK_NONE)，當畫面出現火災或受傷的人時，Gemini 預設的安全機制會觸發，導致你的監控系統失效。
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,}

        # 4. 呼叫 Gemini
        print(f"--- [DEBUG] 正在呼叫模型: {model_name} (等待回應...)")
        model = genai.GenerativeModel(model_name)

        response = model.generate_content(
            prompt_content,
            generation_config=genai.types.GenerationConfig(temperature=0.2),
            safety_settings=safety_settings)

        # 5. 檢查 Prompt 是否因為太色情或暴力直接被擋下。
        print(f"--- [DEBUG]API 回應狀態 (Feedback): {response.prompt_feedback}")

        try:
            raw_text = response.text
            print("--- [DEBUG] 成功取得文字回應，前 500 字元如下:")
            print(raw_text[:500])
            print("---------------------------------------------")
        except Exception as e:

            # 如果沒有 text，代表被擋或是空回應
            print(f"--- [DEBUG] ❌ 無法取得 .text 屬性，原因: {e}")
            if response.candidates:
                print(f"--- [DEBUG] 結束原因 (Finish Reason): {response.candidates[0].finish_reason}")
                print(f"--- [DEBUG] 安全評級 (Safety Ratings): {response.candidates[0].safety_ratings}")
            return ({"error": "no_text_returned", "detail": str(e)}, "無法取得摘要")

        # 6. Markdown 清洗，Gemini 很喜歡用 ```json 包住回傳值，所以這裡會做清洗
        clean_text = raw_text.replace("```json", "").replace("```", "").strip()
        return _extract_first_json_and_tail(clean_text)

    except Exception as e:
        print(f"--- [DEBUG] 發生未預期錯誤: {str(e)}")
        return ({"error": f"gemini_api_error: {str(e)}"}, "")

# ================== 確認狀態的路由 ==================

# Ping 的功能，確認 API 還有在運行
@app.get("/health", tags=["確認連線狀態"])
def health():
    return {"ok": True, "time": _now_ts()}

# 驗證 Key 是否有效 (前端 checkAuth 用)
@app.get("/auth/verify", tags=["驗證 Key 是否有效"])
def auth_verify(api_key: str = Depends(get_api_key)):
    """
    簡單回傳 200 OK，代表 Header 裡的 Key 是正確的。
    如果 Key 錯誤，get_api_key 會直接拋出 401 異常，根本進不到這裡。
    """
    is_admin = api_key == ADMIN_TOKEN
    return {"ok": True, "message": "Key is valid", "is_admin": is_admin}

# ================== 所有業務邏輯 ==================

# 制定資料格式的正確標準，供 /v1/analyze_segment_result、/v1/segment_pipeline_multipart 使用
class SegmentAnalysisRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # 檔案資訊
    segment_path: str
    segment_index: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    # 模型設定
    model_type: str # 'qwen', 'gemini', 'owl'
    qwen_model: str = "qwen2.5-vl:7b"
    frames_per_segment: int = 8
    target_short: int = 720

    # Prompt
    event_detection_prompt: str
    summary_prompt: str

    # OWL 參數
    owl_labels: Optional[str] = None
    owl_every_sec: float = 2.0
    owl_score_thr: float = 0.15

# 處理一個小片段，完全不管檔案是怎麼上傳或切割的
@app.post("/v1/analyze_segment_result", dependencies=[Depends(get_api_key)], tags=["總結"])
def analyze_segment_result(req: SegmentAnalysisRequest):
    """
    這就是您要求的 API：輸入單一片段資訊，輸出該片段的分析結果 (JSON + Summary)
    """
    p = req.segment_path
    tr = f"{_fmt_hms(req.start_time)} - {_fmt_hms(req.end_time)}"
    t1 = time.time()

    # 回傳結構初始化
    result = {
        "segment": Path(p).name,
        "time_range": tr,
        "duration_sec": round(req.end_time - req.start_time, 2),
        "success": False,
        "time_sec": 0.0,
        "parsed": {},
        "raw_detection": None,
        "error": None
    }

    try:
        # ==================== Qwen / Gemini 邏輯 ====================
        if req.model_type in ("qwen", "gemini"):
            # 1. 執行推論
            if req.model_type == "qwen":
                frame_obj, summary_txt = infer_segment_qwen(
                    req.qwen_model, p, req.event_detection_prompt, req.summary_prompt,
                    target_short=req.target_short, frames_per_segment=req.frames_per_segment
                )
            else: # gemini
                # 自動判斷模型名稱
                g_model = req.qwen_model if req.qwen_model.startswith("gemini") else "gemini-2.5-flash"
                frame_obj, summary_txt = infer_segment_gemini(
                    g_model, p, req.event_detection_prompt, req.summary_prompt,
                    req.target_short, req.frames_per_segment
                )

            # 2. 資料清洗與標準化 (Normalization)
            frame_norm = {
                "events": {
                    "water_flood": False, "fire": False,
                    "abnormal_attire_face_cover_at_entry": False,
                    "person_fallen_unmoving": False,
                    "double_parking_lane_block": False,
                    "smoking_outside_zone": False,
                    "crowd_loitering": False,
                    "security_door_tamper": False,
                    "reason": ""
                }
            }

            if isinstance(frame_obj, dict) and "error" not in frame_obj:
                ev = frame_obj.get("events") or {}
                defaults = frame_norm["events"]

                # [動態欄位更新] 支援使用者自訂 Prompt
                for k, v in ev.items():
                    if k == "reason": continue
                    try:
                        defaults[k] = bool(v)
                    except: pass

                # [Reason 排序修正] 刪除再新增，確保排在最後
                reason_text = str(ev.get("reason", "") or "")
                if "reason" in defaults: del defaults["reason"]
                defaults["reason"] = reason_text


            # 填寫成功結果
            result["success"] = ("error" not in (frame_obj or {})) and \
                                (not req.summary_prompt.strip() or len((summary_txt or "").strip()) > 0)
            result["parsed"] = {
                "frame_analysis": frame_norm,
                "summary_independent": (summary_txt or "").strip()
            }

        # ==================== OWL 邏輯 ====================
        elif req.model_type == "owl":
            j = infer_segment_owl(p, labels=req.owl_labels, every_sec=req.owl_every_sec, score_thr=req.owl_score_thr)
            result["success"] = True
            result["raw_detection"] = j

        else:
            raise ValueError("model_type must be qwen, gemini, or owl")

    except Exception as ex:
        result["error"] = str(ex)

    result["time_sec"] = round(time.time() - t1, 2)
    return result

# 它不親自做分析，而是負責調度資源與流程控制。影片，切割，片段影片填入標準格式，片段 API 處理，打包成大的 JSON
@app.post("/v1/segment_pipeline_multipart", dependencies=[Depends(get_api_key)], tags=["總結"])
def segment_pipeline_multipart(
    request: Request,
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db) if HAS_DB else None,
    model_type: str = Form(...),
    file: UploadFile = File(None),
    video_url: str = Form(None),
    video_id: str = Form(None),  # 新增：重新分析已存在的影片
    segment_duration: float = Form(10.0),
    overlap: float = Form(0.0),
    qwen_model: str = Form("qwen3-vl:8b"),
    frames_per_segment: int = Form(8),
    target_short: int = Form(720),
    owl_labels: str = Form("person,pedestrian,motorcycle,car,bus,scooter,truck"),
    owl_every_sec: float = Form(2.0),
    owl_score_thr: float = Form(0.15),
    event_detection_prompt: str = Form(EVENT_DETECTION_PROMPT),
    summary_prompt: str = Form(SUMMARY_PROMPT),
    save_json: bool = Form(True),
    save_basename: str = Form(None),
):

    target_filename = "unknown_video"

    # 1. 下載與儲存 (維持原樣)
    # 如果提供了 video_id，表示要重新分析已存在的影片，跳過下載和切割
    if video_id and video_id.strip():
        # 使用已存在的影片，不需要下載或切割
        local_path = None
        cleanup = False
    elif file is not None:
        # [修正 1] 抓取原始檔名 (例如 "my_video.mp4")
        target_filename = file.filename or "video.mp4"
        fd, tmp = tempfile.mkstemp(prefix="upload_", suffix=Path(file.filename or "video.mp4").suffix)
        with os.fdopen(fd, "wb") as f: f.write(file.file.read())
        local_path, cleanup = tmp, True
    elif video_url:
        # [修正 2] 如果是 URL，從網址抓檔名
        target_filename = Path(video_url).name or "video_url.mp4"
        local_path, cleanup = _download_to_temp(video_url), True
    else:
        raise HTTPException(status_code=422, detail="需要 file、video_url 或 video_id")

    # 2. 切割影片 (如果沒有使用已存在的影片)
    # [修正 3] 使用 "原始檔名" 來當作 ID，而不是用 local_path 的亂碼檔名
    if video_id and video_id.strip():
        video_id_clean = video_id.strip()
        
        # 檢查是否為 video_lib 格式 (category/video_name)
        if "/" in video_id_clean:
            # 從 video 資料夾讀取原始影片
            category, video_name = video_id_clean.split("/", 1)
            
            # 檢查是否已經在 segment 中處理過（使用 {category}_{video_name} 作為 ID）
            stem = f"{category}_{video_name}"  # 使用分類和影片名作為 ID
            seg_dir = Path("segment") / stem
            
            if seg_dir.exists() and list(seg_dir.glob("segment_*.mp4")):
                # 已經處理過，直接使用現有的片段（不從 video 資料夾複製）
                seg_files = sorted(seg_dir.glob("segment_*.mp4"))
                try:
                    json_files = list(seg_dir.glob("*.json"))
                    if json_files:
                        with open(max(json_files, key=lambda p: p.stat().st_mtime), "r", encoding="utf-8") as f:
                            old_data = json.load(f)
                            total_duration = sum(r.get("duration_sec", segment_duration) for r in old_data.get("results", []))
                    else:
                        total_duration = len(seg_files) * segment_duration
                except:
                    total_duration = len(seg_files) * segment_duration
            else:
                # 尚未處理過，需要從 video 資料夾讀取原始影片並切割
                video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
                
                # 嘗試其他擴展名
                if not video_path.exists():
                    for ext in ['.avi', '.mov', '.mkv', '.flv']:
                        video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                        if video_path.exists():
                            break
                
                if not video_path.exists():
                    raise HTTPException(status_code=404, detail=f"Video {video_id_clean} not found in video library")
                
                # 尚未處理過，需要切割影片
                seg_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # 複製影片到 segment 資料夾進行處理
                    import shutil
                    temp_video = seg_dir / video_path.name
                    shutil.copy2(video_path, temp_video)
                    seg_files = _split_one_video(temp_video, seg_dir, segment_duration, overlap, prefix="segment")
                    total_duration = _probe_duration_seconds(temp_video)
                    # 處理完後可以選擇刪除臨時副本（保留原始文件在 video 資料夾）
                    # os.remove(temp_video)  # 可選：刪除臨時副本
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"切割失敗：{e}")
        else:
            # 傳統的 segment 中的影片
            stem = video_id_clean
            seg_dir = Path("segment") / stem
            if not seg_dir.exists():
                raise HTTPException(status_code=404, detail=f"Video {video_id_clean} not found")
            
            # 查找已存在的片段影片
            seg_files_existing = sorted(seg_dir.glob("segment_*.mp4"))
            if not seg_files_existing:
                raise HTTPException(status_code=404, detail=f"No segment files found for video {video_id_clean}")
            
            seg_files = seg_files_existing
            # 估算總時長（從片段數量推斷，或從 JSON 讀取）
            try:
                json_files = list(seg_dir.glob("*.json"))
                if json_files:
                    with open(max(json_files, key=lambda p: p.stat().st_mtime), "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        total_duration = sum(r.get("duration_sec", segment_duration) for r in old_data.get("results", []))
                else:
                    total_duration = len(seg_files) * segment_duration
            except:
                total_duration = len(seg_files) * segment_duration
    else:
        stem = Path(target_filename).stem
        # 建立固定的資料夾 segment/video_1/
        seg_dir = Path("segment") / stem
        seg_dir.mkdir(parents=True, exist_ok=True)
        try:
            seg_files = _split_one_video(local_path, seg_dir, segment_duration, overlap, prefix="segment")
            total_duration = _probe_duration_seconds(local_path)
        except Exception as e:
            if cleanup and os.path.exists(local_path): os.remove(local_path)
            raise HTTPException(status_code=500, detail=f"切割失敗：{e}")

    # 3. 迴圈：Call API 取得結果
    results = []
    t0 = time.time()

    print(f"--- 開始處理 {len(seg_files)} 個片段，呼叫分析 API ---")

    for p in seg_files:
        # 3.1 計算時間區段資訊
        m = re.search(r"(\d+)", Path(p).name)
        idx = int(m.group(1)) if m else 0
        start = idx * (segment_duration - overlap)
        end = min(start + segment_duration, total_duration)

        # 3.2 準備參數 (Request Body)
        req_data = SegmentAnalysisRequest(
            segment_path=str(p), # 傳遞絕對路徑
            segment_index=idx,
            start_time=start,
            end_time=end,
            model_type=model_type,
            qwen_model=qwen_model,
            frames_per_segment=frames_per_segment,
            target_short=target_short,
            event_detection_prompt=event_detection_prompt,
            summary_prompt=summary_prompt,
            owl_labels=owl_labels,
            owl_every_sec=owl_every_sec,
            owl_score_thr=owl_score_thr
        )

        # 3.3 【關鍵步驟】Call API
        # 這裡直接呼叫函式，這等同於透過內部網路呼叫該 API，但更快
        res = analyze_segment_result(req_data)
        results.append(res)

    # 4. 統計與存檔 (維持原樣)
    total_time = time.time() - t0
    ok_count = sum(1 for r in results if r.get("success"))

    resp = {
        "model_type": model_type,
        "total_segments": len(results),
        "success_segments": ok_count,
        "total_time_sec": round(total_time, 2),
        "results": results,
    }

    try:
        if save_json:
            filename = save_basename or f"{stem}.json"

            save_path = seg_dir / filename
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(resp, f, ensure_ascii=False, indent=2)
            resp["save_path"] = str(save_path)

            if AUTO_RAG_INDEX:
                resp["rag_auto_indexed"] = _auto_index_to_rag(resp)
    except Exception: pass

    # 5. 保存分析結果到 PostgreSQL（與 RAG 同步）
    if HAS_DB and db:
        try:
            _save_results_to_postgres(db, results, stem)
        except Exception as e:
            print(f"--- [WARNING] 保存到 PostgreSQL 失敗: {e} ---")
            # 不中斷流程，只記錄警告

    if cleanup and os.path.exists(local_path):
        try: os.remove(local_path)
        except: pass

    return JSONResponse(resp, media_type="application/json; charset=utf-8")

# ================== 前端網頁取得 prompt 的來源 ==================

# 新增一個 GET 路由
@app.get("/prompts/defaults", tags=["回傳 prompt 到前端"])
def get_default_prompts():
    """回傳後端設定的預設 Prompts（動態讀取文件，無需重啟服務）"""
    # 動態讀取 prompt 文件，而不是使用啟動時緩存的變數
    prompts_dir = Path(__file__).parent.parent / "prompts"
    
    def _read_prompt_file(filename: str) -> str:
        """讀取 prompt 文件"""
        file_path = prompts_dir / filename
        try:
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                # 清理：去掉 BOM 與首尾空白
                if content and content[0] == "\ufeff":
                    content = content[1:]
                return content.strip()
            else:
                # 如果文件不存在，回退到緩存的變數
                return EVENT_DETECTION_PROMPT if filename == "frame_prompt.md" else SUMMARY_PROMPT
        except Exception as e:
            # 讀取失敗時回退到緩存的變數
            print(f"[警告] 無法讀取 {filename}：{e}，使用緩存值")
            return EVENT_DETECTION_PROMPT if filename == "frame_prompt.md" else SUMMARY_PROMPT
    
    return {
        "event_prompt": _read_prompt_file("frame_prompt.md"),
        "summary_prompt": _read_prompt_file("summary_prompt.md")
    }

# ================== RAG 路由 ==================

# 將程式碼裡的英文代號（如 water_flood）轉成人類可讀的中文（如 淹水積水）。
def _event_cn_name(k: str) -> str:
    mapping = {
        "water_flood": "淹水積水",
        "fire": "火災濃煙",
        "abnormal_attire_face_cover_at_entry": "門禁遮臉入場",
        "person_fallen_unmoving": "人員倒地不起",
        "double_parking_lane_block": "車道併排阻塞",
        "smoking_outside_zone": "離開吸菸區吸菸",
        "crowd_loitering": "聚眾逗留",
        "security_door_tamper": "安全門破壞/撬動",
    }
    return mapping.get(k, k)

# 將 save_path（segment/upload_群聚/upload_群聚.json）拆分成 video_display、folder_rel
# video_display (segment/upload_群聚)
# folder_rel (upload_群聚)
def _derive_video_and_folder(src_resp: Any) -> Tuple[str, Optional[str]]:

    # 預設
    video_display = str(src_resp.get("video") or src_resp.get("input") or "unknown_video")
    folder_rel: Optional[str] = None

    save_path = src_resp.get("save_path")
    if isinstance(save_path, str) and save_path.strip():

        # 標準化路徑分隔符
        p = Path(save_path).as_posix()

        # [邏輯確認] 假設路徑結構是 segment/影片名/檔名.json
        parts = p.strip("/").split("/")

        if "segment" in parts:
            idx = parts.index("segment")
            # 如果結構是 .../segment/video_name/xxx.json
            if len(parts) > idx + 1:
                folder_name = parts[idx + 1]

                # [關鍵] 這是我們在 RAG 裡面的 Unique Key (Video ID)
                # 只要是同一個 folder_name，就視為同一部影片
                video_display = f"/segment/{folder_name}"
                folder_rel = folder_name

    return video_display, folder_rel

# 把分析完的複雜 JSON 結構（給程式看的），轉換成 「適合 RAG 搜尋的文件格式」
# 輸入參數 src_resp 預期是字典或列表，輸出是一個由字典組成的列表（標準 RAG 文件格式）。
def _results_to_docs(src_resp: Any) -> List[Dict[str, Any]]:

    # 建立一個空列表 docs，用來裝轉換好的文件，最後會回傳它。
    docs: List[Dict[str, Any]] = []

    # 情況 A：完整的 API 回應包 (剛跑完 API 的完整回應)
    if isinstance(src_resp, dict) and "results" in src_resp:

        # src_resp 包含 results 代表他剛跑完
        video_display, folder_rel = _derive_video_and_folder(src_resp)
        items = src_resp.get("results") or []

    # 情況 B：如果輸入直接是一個列表（List），代表它可能只是 results 部分，沒有外層包裝。 (從硬碟讀取的舊存檔)
    elif isinstance(src_resp, list):
        first = src_resp[0] if src_resp else {}
        tmp_src = {"video": (first.get("video") if isinstance(first, dict) else None)}
        video_display, folder_rel = _derive_video_and_folder(tmp_src)
        items = src_resp
    else:
        return docs

    # 遍歷每一個分析片段（Segment)，抓每個 /v1/analyze_segment_result 跑出來的標準小片段格式
    for it in items:
        seg = it.get("segment")
        time_range = it.get("time_range")
        parsed = (it.get("parsed") or {})
        frame = (parsed.get("frame_analysis") or {})
        events = (frame.get("events") or {})
        summary = (parsed.get("summary_independent") or "").strip()

        # 它檢查 events 字典，只保留值為 True 的鍵
        events_true = [k for k, v in events.items() if isinstance(v, bool) and v]
        reason = str(events.get("reason", "") or "")

        # 事件代碼轉成中文。fire -> 火災
        evt_text = "；".join([_event_cn_name(k) for k in events_true]) if events_true else "未見事件"

        # 這段是為了 RAG 搜尋優化，將結構化資料變成一段「自然語言描述」。
        content = (
            f"影片：{video_display}\n"
            f"片段：{seg}（{time_range}）\n"
            f"事件：{evt_text}\n"
            f"說明：{reason}\n"
            f"摘要：{summary}")

        # 為了避免重複索引，將影片名、資料夾、片段名、時間組合成字串，進行 SHA1 雜湊計算，取前 16 碼作為 ID。
        doc_id_base = f"{video_display}|{folder_rel or ''}|{seg}|{time_range}"
        doc_id = hashlib.sha1(doc_id_base.encode("utf-8")).hexdigest()[:16]

        # 這些欄位不會被變成向量（Embed），如果使用者搜尋「火災」，我們可以先用 metadata 過濾 events_true 包含 fire 的文件，再進行語意搜尋。
        meta = {
            "video": video_display,       # 顯示用（/segment/<folder>）
            "folder": folder_rel,         # 給 /rag/search → video_url 用（相對 MEDIA_ROOT）
            "segment": seg,
            "time_range": time_range,
            "duration_sec": it.get("duration_sec"),
            "events_true": events_true,
            "reason": reason,
            "summary": summary,
        }
        docs.append({"id": doc_id, "content": content, "metadata": meta})

    return docs

# 根據 Video ID (路徑識別) 刪除舊的 RAG 紀錄
def _remove_old_rag_records(target_video_id: str):
    """
    讀取 meta.jsonl，把 metadata['video'] == target_video_id 的舊資料通通刪掉。
    """
    meta_path = RAG_DIR / "meta.jsonl"
    if not meta_path.exists():
        return 0

    temp_path = meta_path.with_suffix(".tmp")
    removed_count = 0

    try:
        with open(meta_path, "r", encoding="utf-8") as fin, \
             open(temp_path, "w", encoding="utf-8") as fout:

            for line in fin:
                try:
                    doc = json.loads(line)
                    # 關鍵比對：如果這行資料屬於我們要覆寫的影片，就跳過（刪除）
                    if doc.get("metadata", {}).get("video") == target_video_id:
                        removed_count += 1
                        continue
                    fout.write(line)
                except:
                    fout.write(line)

        # 覆蓋回原檔案
        import shutil
        shutil.move(str(temp_path), str(meta_path))

    except Exception as e:
        print(f"[RAG Clean Error] {e}")
        if temp_path.exists():
            os.remove(temp_path)

    return removed_count

# 當影片分析完成後，順便自動把結果存進向量資料庫。
def _auto_index_to_rag(resp: Dict[str, Any]) -> Dict[str, Any]:

    # 先看全域變數 AUTO_RAG_INDEX 是否為 True。如果關閉就不做。
    if not AUTO_RAG_INDEX:
        return {"enabled": False, "message": "自動 RAG 索引已停用"}

    try:
        # 1. 轉換格式
        docs = _results_to_docs(resp)
        if not docs:
            return {
                "success": False,
                "added": 0,
                "message": "無可索引的文件"
            }

        # 2. [新增] 抓出這次要索引的影片名稱 (Video ID)
        # 我們從第一筆 doc 的 metadata 拿 video 欄位
        target_video_id = docs[0]["metadata"]["video"]

        # 3. 先執行刪除舊資料
        removed = _remove_old_rag_records(target_video_id)

        # 4. 呼叫 RAGStore 加入新資料 (維持原樣)
        store = RAGStore(store_dir=str(RAG_DIR))
        added = store.add_docs(docs)

        # 計算總數
        total = 0
        if store.meta_path.exists():
            with store.meta_path.open("r", encoding="utf-8") as f:
                for _ in f:
                    total += 1

        return {
            "success": True,
            "removed_old": int(removed), # 回傳刪了幾筆
            "added_new": int(added),
            "total": int(total),
            "message": f"✓ RAG 更新完成 (覆蓋 {removed} 筆舊資料，新增 {added} 筆)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"RAG 自動索引失敗：{e}"
        }

# ================== RAG 相關 API ==================

# 將指定的分析結果（JSON）寫入 RAG。
@app.post("/rag/index", tags=["RAG 相關 API"])
async def rag_index(request: Request):

    # 前端直接把分析完的一大包 JSON 陣列傳過來。
    payload = await request.json()
    src_resp = payload.get("results")
    save_path = payload.get("save_path")

    if not src_resp and save_path:
        try:
            src_resp = json.loads(Path(save_path).read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"load save_path failed: {e}")
    if not src_resp:
        raise HTTPException(status_code=422, detail="missing results or save_path")

    docs = _results_to_docs(src_resp)
    if not docs:
        return {"ok": True, "backend": None, "added": 0, "total": 0}

    # 抓出 Video ID 並刪除舊資料
    target_video_id = docs[0]["metadata"]["video"]
    removed = _remove_old_rag_records(target_video_id)

    store = RAGStore(store_dir=RAG_DIR)
    added = store.add_docs(docs)

    # 計算目前總數
    total = 0
    if store.meta_path.exists():
        with store.meta_path.open("r", encoding="utf-8") as f:
            for _ in f: total += 1

    return {
        "ok": True,
        "backend": store.embed_model,
        "removed_old": int(removed),
        "added": int(added),
        "total": int(total)
    }

# ================== 影片管理 API ==================

# 影片事件標籤存儲（簡單的 JSON 文件）
VIDEO_EVENTS_FILE = Path("segment") / "_video_events.json"

def _load_video_events() -> Dict[str, Dict[str, Any]]:
    """載入影片事件標籤"""
    if VIDEO_EVENTS_FILE.exists():
        try:
            with open(VIDEO_EVENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_video_events(events: Dict[str, Dict[str, Any]]):
    """保存影片事件標籤"""
    VIDEO_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VIDEO_EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

def _get_video_lib_categories() -> Dict[str, List[str]]:
    """獲取 video 資料夾中的分類和影片列表"""
    categories = {}
    if VIDEO_LIB_DIR.exists() and VIDEO_LIB_DIR.is_dir():
        for category_dir in VIDEO_LIB_DIR.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                video_files = []
                for video_file in category_dir.iterdir():
                    if video_file.is_file() and video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                        video_files.append(video_file.name)
                if video_files:
                    categories[category_name] = sorted(video_files)
    return categories

@app.get("/v1/videos/list", dependencies=[Depends(get_api_key)], tags=["影片管理"])
def list_videos():
    """獲取已上傳的影片列表（統一管理 segment 和 video 兩個位置）"""
    seg_dir = Path("segment")
    videos = []
    events = _load_video_events()
    video_lib_categories = _get_video_lib_categories()
    
    # 用於追蹤已處理的 video_lib 影片，避免重複顯示
    # key: video_lib 格式的 video_id (例如 "火災生成/Video_火災2")
    # value: segment 中的實際 ID (例如 "火災生成_Video_火災2")
    processed_video_lib = {}  # 改為字典，記錄對應關係
    
    # 1. 從 segment 資料夾讀取已處理的影片
    if seg_dir.exists():
        for video_dir in seg_dir.iterdir():
            if video_dir.is_dir() and not video_dir.name.startswith("_"):
                segment_id = video_dir.name
                # 查找 JSON 文件
                json_files = list(video_dir.glob("*.json"))
                if json_files:
                    # 獲取最新的 JSON 文件
                    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
                    try:
                        with open(latest_json, "r", encoding="utf-8") as f:
                            video_data = json.load(f)
                        
                        # 檢查是否為 video_lib 影片的處理結果（格式：{category}_{video_name}）
                        original_video_id = None
                        category = None
                        display_name = segment_id
                        
                        if "_" in segment_id:
                            # 嘗試還原為 video_lib 格式
                            parts = segment_id.split("_", 1)
                            if len(parts) == 2:
                                potential_category = parts[0]
                                potential_video_name = parts[1]
                                # 檢查是否存在對應的 video_lib 影片
                                if potential_category in video_lib_categories:
                                    for vf in video_lib_categories[potential_category]:
                                        if Path(vf).stem == potential_video_name:
                                            original_video_id = f"{potential_category}/{potential_video_name}"
                                            category = potential_category
                                            display_name = vf  # 使用原始檔案名作為顯示名稱
                                            # 記錄對應關係：video_lib 的 video_id -> segment 的 ID
                                            processed_video_lib[original_video_id] = segment_id
                                            break
                        
                        video_info = {
                            "video_id": original_video_id if original_video_id else segment_id,
                            "display_name": display_name,
                            "source": "segment",  # 標記來源
                            "json_path": str(latest_json.relative_to(Path("."))),
                            "total_segments": video_data.get("total_segments", 0),
                            "success_segments": video_data.get("success_segments", 0),
                            "model_type": video_data.get("model_type", "unknown"),
                            "last_modified": latest_json.stat().st_mtime,
                            "event_label": events.get(original_video_id or segment_id, {}).get("event_label") or (category if category else None),
                            "event_description": events.get(original_video_id or segment_id, {}).get("event_description", ""),
                            "category": category,  # 如果有對應的分類
                            "segment_id": segment_id,  # 保留 segment 中的實際 ID，用於重新分析
                        }
                        videos.append(video_info)
                    except Exception as e:
                        print(f"Warning: Failed to load video info for {segment_id}: {e}")
    
    # 2. 從 video 資料夾讀取歷史影片（按分類），但跳過已經在 segment 中處理過的
    for category_name, video_files in video_lib_categories.items():
        for video_file in video_files:
            video_id = f"{category_name}/{Path(video_file).stem}"
            
            # 如果這個影片已經在 segment 中處理過，跳過（避免重複顯示）
            # processed_video_lib 的 key 是 video_lib 格式的 video_id
            if video_id in processed_video_lib:
                continue
            
            video_path = VIDEO_LIB_DIR / category_name / video_file
            
            # 檢查是否有對應的事件標籤
            event_info = events.get(video_id, {})
            if not event_info.get("event_label"):
                # 如果沒有標籤，使用分類名稱作為預設標籤
                event_info = {"event_label": category_name, "event_description": ""}
            
            video_info = {
                "video_id": video_id,
                "display_name": video_file,
                "source": "video_lib",  # 標記來源
                "json_path": None,  # video_lib 中的影片可能沒有分析結果
                "total_segments": 0,
                "success_segments": 0,
                "model_type": "unknown",
                "last_modified": video_path.stat().st_mtime if video_path.exists() else 0,
                "event_label": event_info.get("event_label", category_name),
                "event_description": event_info.get("event_description", ""),
                "category": category_name,  # 分類名稱
                "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)) if video_path.exists() else None,
            }
            videos.append(video_info)
    
    # 按最後修改時間排序（最新的在前）
    videos.sort(key=lambda x: x["last_modified"], reverse=True)
    
    return {
        "videos": videos,
        "total": len(videos),
        "categories": list(video_lib_categories.keys())  # 返回所有分類
    }

@app.get("/v1/videos/{video_id:path}", dependencies=[Depends(get_api_key)], tags=["影片管理"])
def get_video_info(video_id: str):
    """獲取特定影片的詳細信息（支持 segment 和 video_lib 兩個來源）"""
    # 檢查是否為 video_lib 格式 (category/video_name)
    if "/" in video_id:
        category, video_name = video_id.split("/", 1)
        video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
        # 嘗試其他擴展名
        if not video_path.exists():
            for ext in ['.avi', '.mov', '.mkv', '.flv']:
                video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                if video_path.exists():
                    break
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found in video library")
        
        # 檢查是否已經在 segment 中處理過
        stem = f"{category}_{video_name}"
        seg_dir = Path("segment") / stem
        
        events = _load_video_events()
        event_info = events.get(video_id, {})
        
        # 如果有分析結果，返回分析數據
        json_files = list(seg_dir.glob("*.json")) if seg_dir.exists() else []
        if json_files:
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                return {
                    "video_id": video_id,
                    "display_name": video_path.name,
                    "source": "video_lib",
                    "json_path": str(latest_json.relative_to(Path("."))),
                    "analysis_data": video_data,
                    "event_label": event_info.get("event_label", category),
                    "event_description": event_info.get("event_description", ""),
                    "event_set_by": event_info.get("set_by", ""),
                    "event_set_at": event_info.get("set_at", ""),
                    "category": category,
                    "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load video data: {e}")
        else:
            # 沒有分析結果，只返回基本信息
            return {
                "video_id": video_id,
                "display_name": video_path.name,
                "source": "video_lib",
                "json_path": None,
                "analysis_data": None,
                "event_label": event_info.get("event_label", category),
                "event_description": event_info.get("event_description", ""),
                "event_set_by": event_info.get("set_by", ""),
                "event_set_at": event_info.get("set_at", ""),
                "category": category,
                "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)),
            }
    else:
        # segment 中的影片
        seg_dir = Path("segment") / video_id
        if not seg_dir.exists():
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        # 查找 JSON 文件
        json_files = list(seg_dir.glob("*.json"))
        if not json_files:
            raise HTTPException(status_code=404, detail=f"No analysis result found for {video_id}")
        
        latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest_json, "r", encoding="utf-8") as f:
                video_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load video data: {e}")
        
        events = _load_video_events()
        event_info = events.get(video_id, {})
        
        return {
            "video_id": video_id,
            "display_name": video_id,
            "source": "segment",
            "json_path": str(latest_json.relative_to(Path("."))),
            "analysis_data": video_data,
            "event_label": event_info.get("event_label"),
            "event_description": event_info.get("event_description", ""),
            "event_set_by": event_info.get("set_by", ""),
            "event_set_at": event_info.get("set_at", ""),
            "category": None,
        }

# 獲取 RAG 統計資訊 (前端顯示索引數量用)
@app.get("/rag/stats", dependencies=[Depends(get_api_key)], tags=["RAG 相關 API"])
def rag_stats():
    """
    計算目前 RAG 資料庫裡有多少筆資料。
    透過計算 meta.jsonl 的行數來實現。
    """
    count = 0
    meta_path = RAG_DIR / "meta.jsonl" # 假設您的 RAG Store 是這樣實作的

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                for _ in f:
                    count += 1
        except Exception:
            pass # 讀取失敗就當作 0

    return {
        "count": count,
        "path": str(RAG_DIR.absolute())
    }

# ================== 影片管理 API ==================

# 影片事件標籤存儲（簡單的 JSON 文件）
VIDEO_EVENTS_FILE = Path("segment") / "_video_events.json"

def _load_video_events() -> Dict[str, Dict[str, Any]]:
    """載入影片事件標籤"""
    if VIDEO_EVENTS_FILE.exists():
        try:
            with open(VIDEO_EVENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

@app.post("/v1/videos/{video_id:path}/event", dependencies=[Depends(get_api_key)], tags=["影片管理"])
async def set_video_event(video_id: str, request: Request):
    """設置影片的事件標籤（管理者功能）"""
    # 驗證影片是否存在（支持 segment 和 video_lib 格式）
    video_exists = False
    
    # 檢查是否為 video_lib 格式 (category/video_name)
    if "/" in video_id:
        category, video_name = video_id.split("/", 1)
        video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
        # 嘗試其他擴展名
        if not video_path.exists():
            for ext in ['.avi', '.mov', '.mkv', '.flv']:
                video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                if video_path.exists():
                    break
        video_exists = video_path.exists()
    else:
        # segment 中的影片
        seg_dir = Path("segment") / video_id
        video_exists = seg_dir.exists()
    
    if not video_exists:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    
    payload = await request.json()
    event_label = payload.get("event_label", "").strip()
    event_description = payload.get("event_description", "").strip()
    
    if not event_label:
        raise HTTPException(status_code=422, detail="event_label is required")
    
    events = _load_video_events()
    events[video_id] = {
        "event_label": event_label,
        "event_description": event_description,
        "set_by": "admin",  # 可以從 API key 或 session 獲取
        "set_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    _save_video_events(events)
    
    return {
        "success": True,
        "video_id": video_id,
        "event_label": event_label,
        "event_description": event_description,
        "message": f"影片 {video_id} 已標記為「{event_label}」"
    }

@app.delete("/v1/videos/{video_id:path}/event", dependencies=[Depends(get_api_key)], tags=["影片管理"])
def remove_video_event(video_id: str):
    """移除影片的事件標籤"""
    events = _load_video_events()
    if video_id in events:
        del events[video_id]
        _save_video_events(events)
        return {"success": True, "message": f"已移除影片 {video_id} 的事件標籤"}
    return {"success": False, "message": f"影片 {video_id} 沒有事件標籤"}

@app.get("/v1/videos/categories", dependencies=[Depends(get_api_key)], tags=["影片管理"])
def get_video_categories():
    """獲取 video 資料夾中的所有分類"""
    categories = _get_video_lib_categories()
    return {
        "categories": list(categories.keys()),
        "category_details": {cat: len(videos) for cat, videos in categories.items()}
    }

@app.post("/v1/videos/{video_id:path}/move", dependencies=[Depends(get_api_key)], tags=["影片管理"])
async def move_video_to_category(video_id: str, request: Request):
    """將影片移動到 video 資料夾的指定分類（管理者功能）"""
    # 檢查是否為管理者
    api_key = request.headers.get("X-API-Key", "")
    if api_key != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="此功能僅限管理者使用")
    
    payload = await request.json()
    category = payload.get("category", "").strip()
    event_description = payload.get("event_description", "").strip()
    
    if not category:
        raise HTTPException(status_code=422, detail="category is required")
    
    # 檢查影片是否存在（只支持 segment 中的影片進行移動）
    # 注意：video_lib 中的影片已經在分類資料夾中，不需要移動
    seg_dir = Path("segment") / video_id
    if not seg_dir.exists():
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in segment. Only videos in segment can be moved to categories.")
    
    # 創建目標分類資料夾
    target_category_dir = VIDEO_LIB_DIR / category
    target_category_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找原始影片文件（可能在 segment 目錄中，或需要從片段重建）
    original_video = None
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        potential_file = seg_dir / f"{video_id}{ext}"
        if potential_file.exists():
            original_video = potential_file
            break
    
    # 如果沒有找到原始影片，嘗試從第一個片段推斷
    if not original_video:
        seg_files = sorted(seg_dir.glob("segment_*.mp4"))
        if seg_files:
            # 使用第一個片段作為參考（實際應該合併所有片段，這裡簡化處理）
            original_video = seg_files[0]
    
    if not original_video:
        raise HTTPException(status_code=404, detail=f"找不到影片文件：{video_id}")
    
    # 複製影片到目標分類資料夾
    target_video_path = target_category_dir / original_video.name
    import shutil
    shutil.copy2(original_video, target_video_path)
    
    # 更新事件標籤
    events = _load_video_events()
    new_video_id = f"{category}/{Path(original_video).stem}"
    events[new_video_id] = {
        "event_label": category,
        "event_description": event_description,
        "set_by": "admin",
        "set_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "moved_from": video_id
    }
    _save_video_events(events)
    
    return {
        "success": True,
        "message": f"影片已移動到分類「{category}」",
        "new_video_id": new_video_id,
        "target_path": str(target_video_path)
    }

# 幫你找影片片段，但不負責解釋內容
@app.post("/rag/search", tags=["RAG 相關 API"])
async def rag_search(request: Request, db: Session = Depends(get_db) if HAS_DB else None):
    try:
        payload = await request.json()

        # query: 你要找什麼？
        query = (payload.get("query") or "").strip()
        top_k = int(payload.get("top_k") or 5)

        # [NEW] 新增分數門檻參數 (預設 0.0 代表不過濾，前端可傳 0.6 代表 60%)
        score_threshold = float(payload.get("score_threshold") or 0.0)

        if not query:
            raise HTTPException(status_code=422, detail="missing query")

        # 步驟 1: 解析查詢條件
        query_filters = {}
        sql_results = {}
        
        if HAS_DB and db:
            try:
                query_filters = _parse_query_filters(query)
                # 調用 SQL 查詢（包含日期、事件、關鍵字過濾）
                sql_results = _filter_summaries_by_query(db, query_filters, limit=1000)
                print(f"--- [DEBUG] PostgreSQL 查詢找到 {len(sql_results)} 筆記錄 ---")
            except Exception as e:
                print(f"--- [WARNING] PostgreSQL 查詢失敗: {e} ---")
                import traceback
                traceback.print_exc()
                query_filters = {}
                sql_results = {}

        # 步驟 2: 對 query 做關鍵字擴展，多 query 搜索後取 union
        expanded_queries = _expand_query_keywords(query)
        print(f"--- [DEBUG] 關鍵字擴展後生成 {len(expanded_queries)} 個查詢: {expanded_queries} ---")
        
        # 步驟 3: 固定 vec_pool = 400~800，多 query 搜索後取 union
        store = RAGStore(store_dir=RAG_DIR)
        vec_pool = 600  # 固定向量候選池大小
        
        all_vector_hits = []
        seen_segment_uids = set()
        
        for expanded_query in expanded_queries:
            # 對每個擴展查詢進行搜索
            vector_hits = store.search(expanded_query, top_k=vec_pool, filters={})
            
            # 合併結果（去重）
            for hit in vector_hits:
                metadata = hit.get("metadata", {})
                video = metadata.get("video", "")
                segment = metadata.get("segment", "")
                time_range = metadata.get("time_range", "")
                
                # 生成 segment_uid
                video_str = str(video) if video else ""
                segment_str = str(segment) if segment else ""
                time_range_str = str(time_range) if time_range else ""
                segment_uid = f"{video_str}|{segment_str}|{time_range_str}"
                
                if segment_uid not in seen_segment_uids:
                    seen_segment_uids.add(segment_uid)
                    all_vector_hits.append(hit)
        
        print(f"--- [DEBUG] Vector 搜索後合併得到 {len(all_vector_hits)} 筆結果 ---")
        
        # 步驟 4: 融合排序（保留三類結果：只在 SQL、只在 Vector、兩邊都命中）
        message_keywords = query_filters.get("message_keywords", [])
        merged_results = _merge_and_rank_results(
            all_vector_hits,
            sql_results,
            query_filters,
            message_keywords
        )
        
        print(f"--- [DEBUG] 融合排序後得到 {len(merged_results)} 筆結果 ---")
        
        # 步驟 5: 過濾低於門檻的結果，然後取 top_k
        filtered_results = [
            r for r in merged_results
            if r["final_score"] >= score_threshold
        ]
        
        # 排序後再取 top_k
        final_results = filtered_results[:top_k]
        
        # 步驟 6: 格式化結果
        norm_hits = []
        for r in final_results:
            vector_hit = r.get("vector_hit")
            sql_data = r.get("sql_data")
            
            # 優先使用 vector_hit 的 metadata，如果沒有則使用 sql_data
            if vector_hit:
                m = vector_hit.get("metadata", {})
                norm_hits.append({
                    "score": round(r["final_score"], 4),
                    "video": m.get("video"),
                    "segment": m.get("segment"),
                    "time_range": m.get("time_range"),
                    "events_true": m.get("events_true", []),
                    "summary": m.get("summary", ""),
                    "reason": m.get("reason", ""),
                    "doc_id": vector_hit.get("id"),
                })
            elif sql_data:
                # 只在 SQL 命中的結果
                norm_hits.append({
                    "score": round(r["final_score"], 4),
                    "video": sql_data.get("video"),
                    "segment": sql_data.get("segment"),
                    "time_range": sql_data.get("time_range"),
                    "events_true": [],
                    "summary": sql_data.get("message", ""),
                    "reason": sql_data.get("event_reason", ""),
                    "doc_id": None,
                })
        
        print(f"--- [DEBUG] 最終返回 {len(norm_hits)} 筆結果 ---")
        
        return {"backend": store.embed_model, "hits": norm_hits}

        # [簡化] 如果有日期過濾，PostgreSQL 先過濾日期，然後 RAG 在這些結果中進行向量搜索
        if has_date_filter:
            if filtered_set and len(filtered_set) > 0:
                # PostgreSQL 已過濾日期，RAG 在這些結果中進行向量搜索
                print(f"--- [DEBUG] PostgreSQL 日期過濾找到 {len(filtered_set)} 筆記錄，開始 RAG 向量搜索 ---")
                
                # RAG 正常搜索（不限制數量，因為後面會過濾）
                search_top_k = top_k * 50  # 增加搜尋數量以確保能找到所有匹配的記錄
                raw_hits = store.search(query, top_k=search_top_k, filters=filters)
                
                # 只保留在 PostgreSQL 過濾結果中的記錄
                filtered_hits = []
                # [新增] 構建事件相關關鍵字列表（用於判斷是否相關）
                event_keyword_map = {
                    "fire": ["火災", "火", "fire", "smoke", "濃煙", "煙"],
                    "water_flood": ["水災", "水", "淹水", "積水", "flood", "water"],
                    "person_fallen_unmoving": ["倒地", "倒地不起", "fallen", "unmoving"],
                    "crowd_loitering": ["群聚", "聚眾", "逗留", "crowd", "loitering"],
                    "abnormal_attire_face_cover_at_entry": ["遮臉", "異常著裝", "face", "cover"],
                    "double_parking_lane_block": ["併排", "停車", "阻塞", "parking", "block"],
                    "smoking_outside_zone": ["吸菸", "抽菸", "smoking"],
                    "security_door_tamper": ["闖入", "突破", "安全門", "tamper", "door"],
                }
                # 收集所有需要匹配的關鍵字
                required_keywords = set()
                if query_filters.get("message_keywords"):
                    required_keywords.update(query_filters["message_keywords"])
                if query_filters.get("event_types"):
                    for event_type in query_filters["event_types"]:
                        if event_type in event_keyword_map:
                            required_keywords.update(event_keyword_map[event_type])
                
                if required_keywords:
                    print(f"--- [DEBUG] 需要匹配的關鍵字: {required_keywords} ---")
                
                for h in raw_hits:
                    m = h.get("metadata", {})
                    seg = m.get("segment")
                    tr = m.get("time_range")
                    if seg and tr:
                        # 寬鬆匹配（處理格式差異）
                        seg_base = seg.rsplit('.', 1)[0] if '.' in seg else seg
                        tr_normalized = tr.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
                        # [修改] 檢查是否在 PostgreSQL 過濾結果中（使用 (video, segment, time_range) 三元組）
                        # 從 RAG metadata 中提取 video 名稱
                        video_from_rag = m.get("video", "").replace("/segment/", "").strip("/") if m.get("video") else None
                        folder_from_rag = m.get("folder", "")
                        video_name = folder_from_rag if folder_from_rag else video_from_rag
                        
                        # 匹配邏輯：檢查 (video, segment, time_range) 是否在過濾集合中
                        matched = False
                        for db_video, db_seg, db_tr in filtered_set:
                            # 檢查 segment 和 time_range 是否匹配（寬鬆匹配）
                            seg_match = (seg == db_seg) or (seg_base == db_seg) or (seg == (db_seg.rsplit('.', 1)[0] if '.' in db_seg else db_seg))
                            tr_match = (tr == db_tr) or (tr_normalized == db_tr) or (tr == db_tr.replace(" - ", "-")) or (tr_normalized == db_tr.replace(" - ", "-"))
                            
                            if seg_match and tr_match:
                                # 如果 video 匹配，或者其中一個為 None（舊記錄），則匹配
                                if (video_name and db_video and video_name == db_video) or \
                                   (not video_name and not db_video) or \
                                   (video_name and not db_video) or \
                                   (not video_name and db_video):
                                    matched = True
                                    break
                        
                        if matched:
                            # [修改] 只有當查詢中沒有事件關鍵字時，才給予高分數（1.0）
                            # 如果有事件關鍵字，則檢查是否有事件標記或摘要中包含關鍵字
                            original_score = h.get("score", 0.0)
                            
                            if not required_keywords:
                                # 沒有事件關鍵字：給予高分數（因為已經通過日期過濾）
                                h["score"] = 1.0
                            else:
                                # 有事件關鍵字：檢查是否相關
                                events_true = m.get("events_true", [])
                                summary = str(m.get("summary", "")).lower()
                                summary_original = str(m.get("summary", ""))
                                
                                has_event_match = False
                                has_keyword_match = False
                                
                                # 檢查是否有匹配的事件標記
                                if query_filters.get("event_types"):
                                    for event_type in query_filters["event_types"]:
                                        if event_type in events_true:
                                            has_event_match = True
                                            break
                                
                                # 檢查摘要中是否包含關鍵字
                                if not has_event_match:
                                    for keyword in required_keywords:
                                        keyword_lower = keyword.lower()
                                        if keyword in summary_original or keyword_lower in summary:
                                            has_keyword_match = True
                                            break
                                
                                # 只有當有事件標記或摘要中包含關鍵字時，才給予高分數
                                if has_event_match or has_keyword_match:
                                    h["score"] = 1.0
                                    print(f"--- [DEBUG] 給予高分數: segment={seg}, 事件匹配={has_event_match}, 關鍵字匹配={has_keyword_match}, events_true={events_true} ---")
                                else:
                                    # 保持原始 RAG 分數（不修改）
                                    print(f"--- [DEBUG] 保持原始分數 {original_score:.4f}: segment={seg}, 無事件標記且摘要中無關鍵字, events_true={events_true}, summary前100字={summary_original[:100]}... ---")
                            
                            filtered_hits.append(h)
                
                raw_hits = filtered_hits
                print(f"--- [DEBUG] RAG 向量搜索後，匹配 PostgreSQL 日期過濾的結果: {len(raw_hits)} 筆 ---")
            else:
                # PostgreSQL 日期過濾沒有找到匹配結果：返回空結果
                print(f"--- [DEBUG] PostgreSQL 日期過濾沒有找到匹配結果，返回空結果 ---")
                raw_hits = []
        else:
            # 沒有日期過濾：正常 RAG 搜尋（完全由 RAG 處理）
            # [修改] 當沒有日期過濾時，不限制搜索數量，讓 RAG 進行全面搜索
            # 事件過濾會在搜索後進行，而不是在搜索時使用 events_true_any 過濾器
            # 這樣可以確保即使沒有事件標記，但摘要中包含關鍵字的結果也能被找到
            search_top_k = top_k * 50 if query_filters.get("event_types") or query_filters.get("message_keywords") else top_k
            # [修改] 不使用 events_true_any 過濾器，讓 RAG 進行全面搜索，然後在結果中進行事件過濾
            raw_hits = store.search(query, top_k=search_top_k, filters={})

        # [新增] 如果查詢中有事件關鍵字，進行 RAG 層面的事件過濾（無論是否有日期過濾）
        if query_filters.get("event_types") or query_filters.get("message_keywords"):
            strict_filtered_hits = []
            required_event_types = query_filters.get("event_types", [])
            required_message_keywords = query_filters.get("message_keywords", [])
            # [新增] 保存 has_date_filter 變數，供事件過濾邏輯使用
            # 直接使用 has_date_filter（已在函數開始時定義）
            try:
                has_date_filter_for_event = has_date_filter
            except NameError:
                has_date_filter_for_event = False
            
            # 構建事件關鍵字映射（用於檢查摘要）
            event_keyword_map = {
                "fire": ["火災", "火", "fire", "smoke", "濃煙", "煙"],
                "water_flood": ["水災", "水", "淹水", "積水", "flood", "water"],
                "person_fallen_unmoving": ["倒地", "倒地不起", "fallen", "unmoving"],
                "crowd_loitering": ["群聚", "聚眾", "逗留", "crowd", "loitering"],
                "abnormal_attire_face_cover_at_entry": ["遮臉", "異常著裝", "face", "cover"],
                "double_parking_lane_block": ["併排", "停車", "阻塞", "parking", "block"],
                "smoking_outside_zone": ["吸菸", "抽菸", "smoking"],
                "security_door_tamper": ["闖入", "突破", "安全門", "tamper", "door"],
            }
            
            # 收集所有需要匹配的關鍵字
            all_keywords = set(required_message_keywords)
            for event_type in required_event_types:
                if event_type in event_keyword_map:
                    all_keywords.update(event_keyword_map[event_type])
            
            print(f"--- [DEBUG] RAG 事件過濾: 需要的事件類型={required_event_types}, 關鍵字={all_keywords} ---")
            
            for h in raw_hits:
                m = h.get("metadata", {})
                events_true = m.get("events_true", [])
                summary = str(m.get("summary", "")).lower()
                summary_original = str(m.get("summary", ""))
                
                # 檢查是否有匹配的事件標記
                has_event_match = False
                if required_event_types:
                    for event_type in required_event_types:
                        if event_type in events_true:
                            has_event_match = True
                            break
                
                # 檢查摘要中是否包含關鍵字（更嚴格的匹配）
                has_keyword_match = False
                if all_keywords:
                    for keyword in all_keywords:
                        keyword_lower = keyword.lower()
                        # [修改] 更嚴格的關鍵字匹配：檢查關鍵字是否在摘要中出現
                        # 對於中文關鍵字（如「火災」），直接檢查是否包含
                        # 對於英文關鍵字（如「fire」），使用不區分大小寫的匹配
                        if keyword in summary_original or keyword_lower in summary:
                            # [新增] 額外檢查：確保不是誤匹配
                            # 例如：避免「無火災」被匹配為「火災」
                            # 檢查關鍵字前後是否有否定詞
                            keyword_pos = summary_original.lower().find(keyword_lower)
                            if keyword_pos >= 0:
                                # 檢查關鍵字前後的上下文
                                context_before = summary_original[max(0, keyword_pos-5):keyword_pos].lower()
                                context_after = summary_original[keyword_pos+len(keyword):min(len(summary_original), keyword_pos+len(keyword)+5)].lower()
                                
                                # 如果關鍵字前有否定詞，則不匹配
                                negation_words = ['無', '沒有', '不', '非', '未', 'no', 'not', 'none', 'without']
                                has_negation = any(neg in context_before for neg in negation_words)
                                
                                if not has_negation:
                                    has_keyword_match = True
                                    print(f"--- [DEBUG] 關鍵字匹配: segment={m.get('segment')}, keyword={keyword}, summary片段={summary_original[:100]}... ---")
                                    break
                
                # 如果有事件類型或關鍵字要求，則必須至少匹配其中一個
                # [修改] 如果沒有找到匹配的結果，但 RAG 搜索返回了結果，可能是因為：
                # 1. 摘要中沒有明確寫出關鍵字，但語義相關
                # 2. 向量搜索找到了相關結果，但摘要描述方式不同
                # 在這種情況下，如果 RAG 分數較高（>0.5），且查詢中只有事件關鍵字（沒有日期），
                # 可以考慮放寬過濾條件，允許高分數的結果通過
                if has_event_match or has_keyword_match:
                    strict_filtered_hits.append(h)
                    print(f"--- [DEBUG] 保留符合事件要求的結果: segment={m.get('segment')}, 事件匹配={has_event_match}, 關鍵字匹配={has_keyword_match}, events_true={events_true} ---")
                else:
                    # [新增] 如果沒有日期過濾，且 RAG 分數較高，可以考慮保留（但優先級較低）
                    # 這適用於向量搜索找到了語義相關但摘要中沒有明確關鍵字的情況
                    score = h.get("score", 0.0)
                    # [修改] 檢查是否有日期過濾
                    if not has_date_filter_for_event and score > 0.5:
                        # 檢查摘要中是否有相關的描述（即使沒有明確的關鍵字）
                        # 例如：對於「倒地」，檢查是否有「躺」、「臥」、「不動」等相關詞
                        related_keywords_map = {
                            "person_fallen_unmoving": ["躺", "臥", "不動", "靜止", "趴", "倒", "fall", "lie", "still"],
                            "fire": ["燃燒", "燒", "煙", "火光", "burn", "flame"],
                            "water_flood": ["濕", "水", "積", "淹", "wet", "water"],
                        }
                        
                        has_related_keyword = False
                        for event_type in required_event_types:
                            if event_type in related_keywords_map:
                                for related_kw in related_keywords_map[event_type]:
                                    if related_kw in summary_original.lower() or related_kw in summary:
                                        has_related_keyword = True
                                        print(f"--- [DEBUG] 找到相關關鍵字: segment={m.get('segment')}, related_keyword={related_kw}, score={score:.2f} ---")
                                        break
                                if has_related_keyword:
                                    break
                        
                        if has_related_keyword:
                            strict_filtered_hits.append(h)
                            print(f"--- [DEBUG] 保留高分數且包含相關關鍵字的結果: segment={m.get('segment')}, score={score:.2f} ---")
                        else:
                            print(f"--- [DEBUG] 過濾掉不符合事件要求的結果: segment={m.get('segment')}, events_true={events_true}, score={score:.2f}, summary前100字={summary_original[:100]}... ---")
                    else:
                        print(f"--- [DEBUG] 過濾掉不符合事件要求的結果: segment={m.get('segment')}, events_true={events_true}, summary前100字={summary_original[:100]}... ---")
            
            raw_hits = strict_filtered_hits
            print(f"--- [DEBUG] RAG 事件過濾後剩餘 {len(raw_hits)} 筆結果 ---")

        # 3. [MODIFIED] 過濾與包裝結果，並根據事件和關鍵字進行優先排序
        norm_hits = []
        print(f"--- [DEBUG] 開始處理 {len(raw_hits)} 筆 RAG 結果，分數門檻: {score_threshold} ---")
        for h in raw_hits:
            score = float(h.get("score", 0.0))

            # [NEW] 過濾低於門檻的結果
            if score < score_threshold:
                continue

            m = h.get("metadata", {})
            
            # [新增] 計算優先級：有事件標記或摘要中包含關鍵字的結果優先
            priority = 0
            events_true = m.get("events_true", [])
            summary = str(m.get("summary", "")).lower()
            message_keywords = query_filters.get("message_keywords", [])
            
            # 檢查是否有匹配的事件標記
            if query_filters.get("event_types"):
                for event_type in query_filters["event_types"]:
                    if event_type in events_true:
                        priority = 2  # 有事件標記，最高優先級
                        break
            
            # 檢查摘要中是否包含關鍵字
            if priority < 2 and message_keywords:
                for keyword in message_keywords:
                    if keyword in summary or keyword.lower() in summary:
                        priority = 1  # 摘要中包含關鍵字，次高優先級
                        break
            
            norm_hits.append({
                "score": round(score, 4), # 回傳小數點後四位
                "priority": priority,  # [新增] 優先級，用於排序
                "video": m.get("video"),
                "segment": m.get("segment"),
                "time_range": m.get("time_range"),
                "events_true": m.get("events_true", []),
                "summary": m.get("summary", ""),
                "reason": m.get("reason", ""),
                "doc_id": h.get("id"),
            })
        
        # [新增] 根據優先級和分數排序：優先級高的在前，同優先級按分數降序
        norm_hits.sort(key=lambda x: (-x.get("priority", 0), -x.get("score", 0)))
        
        # 移除 priority 欄位（不需要返回給前端）
        for hit in norm_hits:
            hit.pop("priority", None)
        
        print(f"--- [DEBUG] 最終返回 {len(norm_hits)} 筆結果 ---")

        return {"backend": store.embed_model, "hits": norm_hits}

    except Exception as e:
        print(f"--- [RAG Search Error] ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"RAG Search Failed: {str(e)}", "detail": str(e)}
        )

# 它不僅幫你找資料，還會 「閱讀資料並回答問題」。
@app.post("/rag/answer", tags=["RAG 相關 API"])
async def rag_answer(request: Request, db: Session = Depends(get_db) if HAS_DB else None):
    try:
        payload = await request.json()
        question = (payload.get("query") or "").strip()
        if not question:
            raise HTTPException(status_code=422, detail="missing query")

        top_k = int(payload.get("top_k") or 5)

        # [NEW] 分數門檻 (建議 RAG 回答時可以設高一點，例如 0.5)
        score_threshold = float(payload.get("score_threshold") or 0.0)

        # 指定用哪個 LLM 來回答
        llm_model = (payload.get("model") or "qwen2.5vl:latest").strip()

        # 步驟 1: 解析查詢條件（與 /rag/search 統一）
        query_filters = {}
        sql_results = {}
        
        if HAS_DB and db:
            try:
                query_filters = _parse_query_filters(question)
                # 調用 SQL 查詢（包含日期、事件、關鍵字過濾）
                sql_results = _filter_summaries_by_query(db, query_filters, limit=1000)
                print(f"--- [DEBUG] PostgreSQL 查詢找到 {len(sql_results)} 筆記錄 ---")
            except Exception as e:
                print(f"--- [WARNING] PostgreSQL 查詢失敗: {e} ---")
                import traceback
                traceback.print_exc()
                query_filters = {}
                sql_results = {}

        # 步驟 2: 對 query 做關鍵字擴展，多 query 搜索後取 union
        expanded_queries = _expand_query_keywords(question)
        print(f"--- [DEBUG] 關鍵字擴展後生成 {len(expanded_queries)} 個查詢: {expanded_queries} ---")
        
        # 步驟 3: 固定 vec_pool = 400~800，多 query 搜索後取 union
        store = RAGStore(store_dir=RAG_DIR)
        vec_pool = 600  # 固定向量候選池大小
        
        all_vector_hits = []
        seen_segment_uids = set()
        
        for expanded_query in expanded_queries:
            # 對每個擴展查詢進行搜索
            vector_hits = store.search(expanded_query, top_k=vec_pool, filters={})
            
            # 合併結果（去重）
            for hit in vector_hits:
                metadata = hit.get("metadata", {})
                video = metadata.get("video", "")
                segment = metadata.get("segment", "")
                time_range = metadata.get("time_range", "")
                
                # 生成 segment_uid
                video_str = str(video) if video else ""
                segment_str = str(segment) if segment else ""
                time_range_str = str(time_range) if time_range else ""
                segment_uid = f"{video_str}|{segment_str}|{time_range_str}"
                
                if segment_uid not in seen_segment_uids:
                    seen_segment_uids.add(segment_uid)
                    all_vector_hits.append(hit)
        
        print(f"--- [DEBUG] Vector 搜索後合併得到 {len(all_vector_hits)} 筆結果 ---")
        
        # 步驟 4: 融合排序（保留三類結果：只在 SQL、只在 Vector、兩邊都命中）
        message_keywords = query_filters.get("message_keywords", [])
        merged_results = _merge_and_rank_results(
            all_vector_hits,
            sql_results,
            query_filters,
            message_keywords
        )
        
        print(f"--- [DEBUG] 融合排序後得到 {len(merged_results)} 筆結果 ---")
        
        # 步驟 5: 過濾低於門檻的結果，然後取 top_k
        filtered_results = [
            r for r in merged_results
            if r["final_score"] >= score_threshold
        ]
        
        # 排序後再取 top_k
        final_results = filtered_results[:top_k]
        
        # 步驟 6: 格式化結果
        norm_hits = []
        for r in final_results:
            vector_hit = r.get("vector_hit")
            sql_data = r.get("sql_data")
            
            # 優先使用 vector_hit 的 metadata，如果沒有則使用 sql_data
            if vector_hit:
                m = vector_hit.get("metadata", {})
                norm_hits.append({
                    "score": round(r["final_score"], 4),
                    "video": m.get("video"),
                    "segment": m.get("segment"),
                    "time_range": m.get("time_range"),
                    "events_true": m.get("events_true", []),
                    "summary": m.get("summary", ""),
                    "reason": m.get("reason", ""),
                    "doc_id": vector_hit.get("id"),
                })
            elif sql_data:
                # 只在 SQL 命中的結果
                norm_hits.append({
                    "score": round(r["final_score"], 4),
                    "video": sql_data.get("video"),
                    "segment": sql_data.get("segment"),
                    "time_range": sql_data.get("time_range"),
                    "events_true": [],
                    "summary": sql_data.get("message", ""),
                    "reason": sql_data.get("event_reason", ""),
                    "doc_id": None,
                })
        
        print(f"--- [DEBUG] 最終返回 {len(norm_hits)} 筆結果 ---")
        
        # 2. 組裝 Context (A) - 使用 norm_hits 作為上下文
        if not norm_hits:
            # 如果因為門檻過濾後導致沒有資料，也視為找不到
            # [NEW] 如果 Ollama 失敗，直接返回空結果而不是報錯
            try:
                msgs = [
                    {"role": "system", "content": "你只能根據系統提供的資料回答。現在沒有資料可用，請直接說你找不到答案。"},
                    {"role": "user", "content": question},
                ]
                rj = _ollama_chat(llm_model, msgs, timeout=1800)
                msg = ""
                if isinstance(rj, dict):
                    msg = (rj.get("message") or {}).get("content", "").strip()
                elif isinstance(rj, str):
                    msg = rj
            except Exception as ollama_error:
                # Ollama 失敗時，返回空結果而不是報錯
                print(f"--- [WARNING] Ollama 失敗，返回空結果: {ollama_error} ---")
                msg = "目前索引到的片段裡找不到答案（或是相似度過低）。LLM 服務暫時無法使用。"

            return {
                "backend": {"embed_model": store.embed_model, "llm": llm_model},
                "hits": [],
                "answer": msg or "目前索引到的片段裡找不到答案（或是相似度過低）。",
            }

        # 組裝 Context (A) - 使用已經格式化好的 norm_hits
        context_blocks = []
        for i, hit in enumerate(norm_hits, start=1):
            summary = hit.get("summary", "")
            video = hit.get("video")
            time_range = hit.get("time_range")
            
            # 在 context 中加入分數資訊讓 LLM 參考也不錯，但這邊先保持簡潔
            context_blocks.append(
                f"[{i}] 影片: {video}  時間: {time_range}\n摘要: {summary}"
            )

        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "你是工廠監控影片說明助理，必須嚴格根據提供的片段摘要回答問題。"
            "如果資料中沒有答案，就回答「我在目前索引到的片段裡找不到相關資訊」。"
        )
        user_prompt = (
            f"使用下面這些片段摘要回答問題：\n\n{context_text}\n\n"
            f"問題：{question}\n\n"
            "請用繁體中文回答，並在回答中附上你參考的片段編號（例如 [1]、[2]）。"
        )

        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 3. 呼叫 LLM (G) - 如果失敗，返回搜尋結果
        try:
            ans_content = _ollama_chat(llm_model, msgs, timeout=1800)
            # 防呆處理
            if isinstance(ans_content, dict):
                answer = (ans_content.get("message") or {}).get("content", "").strip()
            else:
                answer = str(ans_content).strip()
        except Exception as ollama_error:
            # [NEW] Ollama 失敗時，返回搜尋結果而不是報錯
            print(f"--- [WARNING] Ollama 失敗，返回搜尋結果: {ollama_error} ---")
            answer = f"抱歉，LLM 服務暫時無法使用（錯誤：{str(ollama_error)[:100]}）。以下是根據您的查詢找到的相關片段：\n\n"
            # 將搜尋結果轉換為文字描述
            for i, hit in enumerate(norm_hits, 1):
                answer += f"[{i}] 影片: {hit.get('video', 'N/A')}  時間: {hit.get('time_range', 'N/A')}\n"
                answer += f"    摘要: {hit.get('summary', 'N/A')[:100]}...\n\n"

        return {
            "backend": {"embed_model": store.embed_model, "llm": llm_model},
            "hits": norm_hits,
            "answer": answer,
        }

    except Exception as e:
        print(f"--- [RAG Answer Error] ---")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"RAG Answer Failed: {str(e)}", "detail": str(e)}
        )

# ================== PostgreSQL 保存與過濾 ==================

def _parse_time_range(time_range_str: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    解析時間範圍字串，例如 "00:00:00 - 00:00:08"
    返回 (start_time, end_time) 作為 datetime 對象
    """
    try:
        if not time_range_str or not isinstance(time_range_str, str):
            return None, None
            
        if " - " in time_range_str:
            start_str, end_str = time_range_str.split(" - ", 1)
            start_str = start_str.strip()
            end_str = end_str.strip()
            
            # 解析 HH:MM:SS 格式
            start_time_obj = datetime.strptime(start_str, "%H:%M:%S").time()
            end_time_obj = datetime.strptime(end_str, "%H:%M:%S").time()
            
            # 創建 datetime 對象（使用今天的日期作為基準）
            today = datetime.now().date()
            
            start_time = datetime.combine(today, start_time_obj)
            end_time = datetime.combine(today, end_time_obj)
            
            return start_time, end_time
    except ValueError as e:
        print(f"Warning: Could not parse time range '{time_range_str}': {e}")
    except Exception as e:
        print(f"Warning: Unexpected error parsing time range '{time_range_str}': {e}")
    
    return None, None


def _save_results_to_postgres(db: Session, results: List[Dict[str, Any]], video_stem: str):
    """
    將分析結果保存到 PostgreSQL 資料庫
    與 migrate_segments_to_db.py 的邏輯一致：影片相同則更新，新的則新增
    
    Args:
        db: 資料庫 session
        results: 分析結果列表（來自 segment_pipeline_multipart 的 results）
        video_stem: 影片名稱（用於識別，例如 "fire_1" 或 "火災生成_Video_火災"）
    """
    if not HAS_DB:
        return
    
    # [修改] 採用「更新或新增」的邏輯，與 migrate_segments_to_db.py 保持一致
    # 不再先刪除舊記錄，而是檢查是否存在，存在則更新，不存在則新增
    saved_count = 0
    updated_count = 0
    inserted_count = 0
    
    for result in results:
        # 只處理成功的結果
        if not result.get("success", False):
            continue
        
        # 獲取摘要文字
        parsed = result.get("parsed", {})
        summary_text = parsed.get("summary_independent", "")
        
        # 如果沒有摘要，跳過
        if not summary_text or not summary_text.strip():
            continue
        
        # 解析時間範圍
        time_range = result.get("time_range", "")
        start_time, end_time = _parse_time_range(time_range)
        
        # 獲取其他欄位
        segment = result.get("segment", "")
        duration_sec = result.get("duration_sec")
        time_sec = result.get("time_sec")
        
        # 獲取事件檢測資料
        frame_analysis = parsed.get("frame_analysis", {})
        events = frame_analysis.get("events", {})
        
        # 檢查記錄是否已存在（根據 video、segment 和 time_range）
        # 與 migrate_segments_to_db.py 的邏輯一致：影片相同則更新，新的則新增
        # [修改] 加入 video 欄位判斷，避免不同影片的相同 segment 和 time_range 被誤判為同一筆記錄
        existing = db.query(Summary).filter(
            Summary.video == video_stem,
            Summary.segment == segment,
            Summary.time_range == time_range
        ).first()
        
        if existing:
            # 更新現有記錄（影片相同則更新）
            existing.start_timestamp = start_time if start_time else datetime.now()
            existing.end_timestamp = end_time
            existing.video = video_stem  # [新增] 確保 video 欄位也被更新
            existing.message = summary_text.strip()
            existing.duration_sec = float(duration_sec) if duration_sec is not None else None
            existing.time_sec = float(time_sec) if time_sec is not None else None
            # 更新事件檢測欄位
            existing.water_flood = bool(events.get("water_flood", False))
            existing.fire = bool(events.get("fire", False))
            existing.abnormal_attire_face_cover_at_entry = bool(events.get("abnormal_attire_face_cover_at_entry", False))
            existing.person_fallen_unmoving = bool(events.get("person_fallen_unmoving", False))
            existing.double_parking_lane_block = bool(events.get("double_parking_lane_block", False))
            existing.smoking_outside_zone = bool(events.get("smoking_outside_zone", False))
            existing.crowd_loitering = bool(events.get("crowd_loitering", False))
            existing.security_door_tamper = bool(events.get("security_door_tamper", False))
            existing.event_reason = events.get("reason", "") if events.get("reason") else None
            # 更新 updated_at 時間戳
            existing.updated_at = datetime.now()
            saved_count += 1
            updated_count += 1
        else:
            # 新增記錄（新的則新增）
            summary = Summary(
                start_timestamp=start_time if start_time else datetime.now(),
                end_timestamp=end_time,
                location=None,  # 之後可以從其他地方填入
                camera=None,    # 之後可以從其他地方填入
                video=video_stem,  # [新增] 保存影片名稱，用於區分不同影片的相同 segment
                message=summary_text.strip(),
                segment=segment if segment else None,
                time_range=time_range if time_range else None,
                duration_sec=float(duration_sec) if duration_sec is not None else None,
                time_sec=float(time_sec) if time_sec is not None else None,
                # 事件檢測欄位
                water_flood=bool(events.get("water_flood", False)),
                fire=bool(events.get("fire", False)),
                abnormal_attire_face_cover_at_entry=bool(events.get("abnormal_attire_face_cover_at_entry", False)),
                person_fallen_unmoving=bool(events.get("person_fallen_unmoving", False)),
                double_parking_lane_block=bool(events.get("double_parking_lane_block", False)),
                smoking_outside_zone=bool(events.get("smoking_outside_zone", False)),
                crowd_loitering=bool(events.get("crowd_loitering", False)),
                security_door_tamper=bool(events.get("security_door_tamper", False)),
                event_reason=events.get("reason", "") if events.get("reason") else None,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            try:
                db.add(summary)
                saved_count += 1
                inserted_count += 1
            except Exception as e:
                print(f"Warning: Failed to add summary to session: {e}")
                continue
    
    # 批量提交
    if saved_count > 0:
        try:
            db.commit()
            print(f"--- [PostgreSQL] 成功保存/更新 {saved_count} 筆分析結果到資料庫 (video: {video_stem}, 新增: {inserted_count}, 更新: {updated_count}) ---")
        except Exception as e:
            db.rollback()
            print(f"--- [PostgreSQL ERROR] 提交失敗: {e} ---")
    else:
        # 即使沒有新記錄，也要提交（可能只有更新操作）
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            print(f"--- [PostgreSQL ERROR] 提交失敗: {e} ---")


# ================== Prompt 解析與 PostgreSQL 過濾 ==================

def _parse_query_filters(question: str) -> Dict[str, Any]:
    """
    從用戶問題中解析出過濾條件：
    - 日期/時間（如 "1219" -> 2025-12-19, "給我 1219 的影片" -> 2025-12-19）
    - 地點（如 "路口" -> location 欄位）
    - 事件類型（如 "火災"、"水災"、"闖入"）
    
    返回一個字典，包含：
    - date_filter: Optional[date] - 日期過濾
    - location_keywords: List[str] - 地點關鍵字
    - event_types: List[str] - 事件類型（對應到資料庫欄位）
    """
    filters = {
        "date_filter": None,
        "location_keywords": [],
        "event_types": [],
    }
    
    # 事件類型映射（中文 -> 資料庫欄位）
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
    
    # 解析日期（格式：MMDD 或 YYYYMMDD）
    # [FIX] 改進日期解析，支援 "給我 1219 的影片"、"給我 1220 的影片" 這種格式
    date_patterns = [
        r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
        r'(\d{2})(\d{2})',          # MMDD (例如 1219, 1220)
    ]
    
    for pattern in date_patterns:
        matches = list(re.finditer(pattern, question))
        for match in matches:
            if len(match.groups()) == 3:  # YYYYMMDD
                year, month, day = match.groups()
                try:
                    filters["date_filter"] = date(int(year), int(month), int(day))
                    print(f"--- [DEBUG] 解析到日期 (YYYYMMDD): {filters['date_filter']} ---")
                    break
                except ValueError:
                    continue
            elif len(match.groups()) == 2:  # MMDD
                month, day = match.groups()
                try:
                    # 假設是當前年份
                    current_year = datetime.now().year
                    filters["date_filter"] = date(current_year, int(month), int(day))
                    print(f"--- [DEBUG] 解析到日期 (MMDD): {filters['date_filter']} (年份: {current_year}) ---")
                    break
                except ValueError:
                    continue
        if filters["date_filter"]:
            break
    
    # [NEW] message 關鍵字過濾：如果查詢中包含事件相關關鍵字，也在 message 中搜尋
    # 這可以幫助找到 message 中提到相關事件的記錄（例如：火災、倒地、群聚等）
    event_keywords_in_message = ["火災", "倒地", "群聚", "聚眾", "水災", "淹水", "闖入", "遮臉", "吸菸", "停車", "阻塞"]
    message_keywords_found = []
    for keyword in event_keywords_in_message:
        if keyword in question:
            message_keywords_found.append(keyword)
    
    # [NEW] 添加描述性關鍵字（顏色、衣服、車輛等）
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
        # 單獨的顏色（用於匹配摘要中的顏色描述）
        "黃色", "黑色", "白色", "紅色", "藍色", "綠色", "灰色", "深色", "淺色",
        # 衣服相關
        "衣服", "上衣", "褲子", "帽子",
        # 車輛相關
        "貨車", "卡車", "汽車", "機車", "車"
    ]
    
    # 先檢查完整的多字關鍵字（如「黃色衣服」、「藍色貨車」）
    for keyword in descriptive_keywords:
        if keyword in question and keyword not in message_keywords_found:
            message_keywords_found.append(keyword)
    
    if message_keywords_found:
        # 將找到的關鍵字添加到 filters 中，用於後續的 message 過濾
        filters["message_keywords"] = message_keywords_found
        print(f"--- [DEBUG] 找到 message 關鍵字: {message_keywords_found} ---")
    
    # 解析事件類型
    question_lower = question.lower()
    for keyword, db_field in event_mapping.items():
        if keyword in question_lower:
            if db_field not in filters["event_types"]:
                filters["event_types"].append(db_field)
    
    # 解析地點關鍵字（簡單關鍵字匹配）
    location_keywords = ["路口", "入口", "出口", "停車場", "大門", "側門", "後門"]
    for keyword in location_keywords:
        if keyword in question:
            filters["location_keywords"].append(keyword)
    
    return filters


def _expand_query_keywords(query: str) -> List[str]:
    """
    對 query 做關鍵字擴展（例如：積水 / 水災 / 淹水）
    返回擴展後的查詢列表
    """
    expanded_queries = [query]  # 原始查詢
    
    # 關鍵字擴展映射
    keyword_expansions = {
        "積水": ["積水", "水災", "淹水", "水"],
        "水災": ["水災", "積水", "淹水", "水"],
        "淹水": ["淹水", "積水", "水災", "水"],
        "火災": ["火災", "火", "煙", "濃煙"],
        "火": ["火", "火災", "煙", "濃煙"],
        "倒地": ["倒地", "倒地不起", "躺", "臥"],
        "群聚": ["群聚", "聚眾", "逗留"],
        "闖入": ["闖入", "突破", "安全門"],
        "遮臉": ["遮臉", "異常著裝"],
        "吸菸": ["吸菸", "抽菸"],
        "停車": ["停車", "併排", "阻塞"],
    }
    
    # 檢查查詢中是否包含可擴展的關鍵字
    for keyword, expansions in keyword_expansions.items():
        if keyword in query:
            for expanded_keyword in expansions:
                if expanded_keyword not in query:
                    # 替換關鍵字生成新的查詢
                    expanded_query = query.replace(keyword, expanded_keyword)
                    if expanded_query not in expanded_queries:
                        expanded_queries.append(expanded_query)
    
    return expanded_queries


def _filter_summaries_by_query(
    db: Session,
    filters: Dict[str, Any],
    limit: int = 1000
) -> Dict[str, Dict[str, Any]]:
    """
    根據過濾條件從 PostgreSQL 查詢 summaries，返回符合條件的記錄
    
    返回: Dict[segment_uid] -> sql_payload
    segment_uid 格式: f"{video}|{segment}|{time_range}"
    sql_payload 包含: start_timestamp, video, segment, time_range, event_reason, message
    """
    if not HAS_DB:
        return {}
    
    # 查詢需要的欄位
    query = db.query(
        Summary.start_timestamp,
        Summary.video,
        Summary.segment,
        Summary.time_range,
        Summary.event_reason,
        Summary.message
    ).filter(
        Summary.message.isnot(None),
        Summary.message != ""
    )
    
    # [關鍵] 日期過濾 - 使用 start_timestamp 做 range filter
    if filters.get("date_filter"):
        target_date = filters["date_filter"]
        # 計算時間範圍：從當天 00:00:00 到次日 00:00:00
        t0 = datetime.combine(target_date, datetime.min.time())
        next_day = target_date + timedelta(days=1)
        t1 = datetime.combine(next_day, datetime.min.time())
        query = query.filter(
            Summary.start_timestamp >= t0,
            Summary.start_timestamp < t1
        )
        print(f"--- [DEBUG] 應用日期過濾 (使用 start_timestamp range): {target_date} ({t0} ~ {t1}) ---")
    
    # [關鍵] 事件類型過濾 - 必須嚴格匹配事件 boolean 欄位
    if filters.get("event_types"):
        event_conditions = []
        for event_type in filters["event_types"]:
            if event_type == "fire":
                event_conditions.append(Summary.fire == True)
            elif event_type == "water_flood":
                event_conditions.append(Summary.water_flood == True)
            elif event_type == "abnormal_attire_face_cover_at_entry":
                event_conditions.append(Summary.abnormal_attire_face_cover_at_entry == True)
            elif event_type == "person_fallen_unmoving":
                event_conditions.append(Summary.person_fallen_unmoving == True)
            elif event_type == "double_parking_lane_block":
                event_conditions.append(Summary.double_parking_lane_block == True)
            elif event_type == "smoking_outside_zone":
                event_conditions.append(Summary.smoking_outside_zone == True)
            elif event_type == "crowd_loitering":
                event_conditions.append(Summary.crowd_loitering == True)
            elif event_type == "security_door_tamper":
                event_conditions.append(Summary.security_door_tamper == True)
        
        if event_conditions:
            query = query.filter(or_(*event_conditions))
            print(f"--- [DEBUG] 應用事件過濾: {filters['event_types']} ---")
    
    # [NEW] message 關鍵字過濾 - 在 SQL 查詢中過濾
    if filters.get("message_keywords"):
        message_conditions = []
        for keyword in filters["message_keywords"]:
            message_conditions.append(Summary.message.ilike(f"%{keyword}%"))
        if message_conditions:
            query = query.filter(or_(*message_conditions))
            print(f"--- [DEBUG] 應用 message 關鍵字過濾: {filters['message_keywords']} ---")
    
    # 執行查詢
    results = query.limit(limit).all()
    print(f"--- [DEBUG] PostgreSQL 查詢返回 {len(results)} 筆原始結果 ---")
    
    # 構建 Dict[segment_uid] -> sql_payload
    sql_results = {}
    for row in results:
        start_ts, video, segment, time_range, event_reason, message = row
        
        # 生成 segment_uid
        video_str = str(video) if video else ""
        segment_str = str(segment) if segment else ""
        time_range_str = str(time_range) if time_range else ""
        segment_uid = f"{video_str}|{segment_str}|{time_range_str}"
        
        # 構建 sql_payload
        sql_results[segment_uid] = {
            "start_timestamp": start_ts,
            "video": video_str,
            "segment": segment_str,
            "time_range": time_range_str,
            "event_reason": str(event_reason) if event_reason else "",
            "message": str(message) if message else ""
        }
    
    print(f"--- [DEBUG] 過濾後的有效結果: {len(sql_results)} 筆 ---")
    return sql_results


def _calculate_sql_score(
    sql_data: Dict[str, Any],
    query_filters: Dict[str, Any],
    message_keywords: List[str]
) -> float:
    """
    計算 SQL 結果的分數
    
    規則：
    - 符合時間：+0.4
    - 符合事件 boolean：+0.5
    - summary / message 命中關鍵字：+0.1
    """
    score = 0.0
    
    # 符合時間：+0.4（如果有日期過濾，已經在 SQL 查詢中過濾了，所以這裡直接給分）
    if query_filters.get("date_filter"):
        score += 0.4
    
    # 符合事件 boolean：+0.5
    # 注意：如果 query_filters 中有 event_types，且 sql_data 存在，
    # 表示這個記錄已經通過了 SQL 的事件 boolean 過濾，所以直接給分
    if query_filters.get("event_types"):
        score += 0.5
    
    # summary / message 命中關鍵字：根據匹配程度給予不同分數
    if message_keywords:
        message = sql_data.get("message", "") or ""
        event_reason = sql_data.get("event_reason", "") or ""
        combined_text = f"{message} {event_reason}".lower()
        
        matched_keywords = []
        for keyword in message_keywords:
            keyword_lower = keyword.lower()
            # 檢查關鍵字是否在文本中
            if keyword_lower in combined_text:
                matched_keywords.append(keyword)
                # 精確匹配（完整關鍵字）：+0.4
                score += 0.4
            # 檢查關鍵字的部分是否在文本中（例如「黃色衣服」中的「黃色」）
            elif len(keyword) >= 2:
                # 對於多字關鍵字，檢查每個字是否在文本中
                keyword_parts = [part for part in keyword_lower.split() if len(part) >= 2]
                matched_parts = [part for part in keyword_parts if part in combined_text]
                if matched_parts:
                    matched_keywords.append(keyword)
                    # 部分匹配：+0.1（每個匹配的部分）
                    score += 0.1 * len(matched_parts)
        
        # 如果匹配到多個關鍵字，給予額外分數
        if len(matched_keywords) > 1:
            score += 0.15 * (len(matched_keywords) - 1)  # 每多一個關鍵字 +0.15
        
        # 如果完全沒有匹配到任何關鍵字，但有關鍵字要求，大幅減分
        if len(matched_keywords) == 0 and len(message_keywords) > 0:
            score -= 0.2
    
    return score


def _merge_and_rank_results(
    vector_hits: List[Dict[str, Any]],
    sql_results: Dict[str, Dict[str, Any]],
    query_filters: Dict[str, Any],
    message_keywords: List[str]
) -> List[Dict[str, Any]]:
    """
    融合排序：合併 Vector 和 SQL 的結果
    
    保留三類結果：
    1. 只在 SQL 命中
    2. 只在 Vector 命中（但如果有日期過濾，則不保留）
    3. 兩邊都命中（加分）
    
    最終排序規則：
    final_score = 0.65 * vector_score + 0.35 * sql_score + 0.15 bonus if in_both
    """
    merged = {}
    has_date_filter = bool(query_filters.get("date_filter"))
    
    # 處理 Vector 結果
    for vector_hit in vector_hits:
        metadata = vector_hit.get("metadata", {})
        video = metadata.get("video", "")
        segment = metadata.get("segment", "")
        time_range = metadata.get("time_range", "")
        
        # 生成 segment_uid（與 SQL 一致）
        video_str = str(video) if video else ""
        segment_str = str(segment) if segment else ""
        time_range_str = str(time_range) if time_range else ""
        segment_uid = f"{video_str}|{segment_str}|{time_range_str}"
        
        vector_score = float(vector_hit.get("score", 0.0))
        
        # [NEW] 檢查 Vector 結果中是否包含關鍵字，如果有則提高分數
        if message_keywords:
            summary = str(metadata.get("summary", "")).lower()
            message = str(metadata.get("message", "")).lower()
            combined_text = f"{summary} {message}".lower()
            
            keyword_match_bonus = 0.0
            matched_count = 0
            
            for keyword in message_keywords:
                keyword_lower = keyword.lower()
                # 精確匹配（完整關鍵字）
                if keyword_lower in combined_text:
                    keyword_match_bonus += 0.3  # 每個精確匹配 +0.3
                    matched_count += 1
                # 部分匹配（例如「黃色衣服」中的「黃色」）
                elif len(keyword) >= 2:
                    keyword_parts = [part for part in keyword_lower.split() if len(part) >= 2]
                    matched_parts = [part for part in keyword_parts if part in combined_text]
                    if matched_parts:
                        keyword_match_bonus += 0.15 * len(matched_parts)  # 每個部分匹配 +0.15
                        matched_count += 1
            
            # 如果匹配到多個關鍵字，給予額外分數
            if matched_count > 1:
                keyword_match_bonus += 0.2 * (matched_count - 1)
            
            # 將關鍵字匹配分數加到 vector_score
            vector_score = min(1.0, vector_score + keyword_match_bonus)  # 限制在 1.0 以內
        
        # 檢查是否也在 SQL 結果中
        sql_data = sql_results.get(segment_uid)
        if sql_data:
            # 兩邊都命中：計算 SQL 分數並加分
            sql_score = _calculate_sql_score(sql_data, query_filters, message_keywords)
            final_score = 0.65 * vector_score + 0.35 * sql_score + 0.15  # bonus for in_both
            in_both = True
        else:
            # 只在 Vector 命中
            # [FIX] 如果有日期過濾，則不保留只在 Vector 命中的結果（因為它們不符合日期條件）
            if has_date_filter:
                continue  # 跳過不符合日期條件的 Vector 結果
            
            sql_score = 0.0
            final_score = 0.65 * vector_score + 0.35 * sql_score
            in_both = False
        
        merged[segment_uid] = {
            "segment_uid": segment_uid,
            "vector_score": vector_score,
            "sql_score": sql_score,
            "final_score": final_score,
            "in_both": in_both,
            "vector_hit": vector_hit,
            "sql_data": sql_data
        }
    
    # 處理只在 SQL 命中的結果
    for segment_uid, sql_data in sql_results.items():
        if segment_uid not in merged:
            # 只在 SQL 命中
            sql_score = _calculate_sql_score(sql_data, query_filters, message_keywords)
            final_score = 0.65 * 0.0 + 0.35 * sql_score  # vector_score = 0
            merged[segment_uid] = {
                "segment_uid": segment_uid,
                "vector_score": 0.0,
                "sql_score": sql_score,
                "final_score": final_score,
                "in_both": False,
                "vector_hit": None,
                "sql_data": sql_data
            }
    
    # 按 final_score 排序
    sorted_results = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)
    
    return sorted_results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8080, reload=True)