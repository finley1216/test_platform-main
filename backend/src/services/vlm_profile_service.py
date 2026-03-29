# -*- coding: utf-8 -*-
"""
VLM 執行設定檔：持久化使用者選擇的推論後端（Ollama / vLLM×2），並可選透過 docker compose 停啟服務以釋放 GPU。
預設 profile 與 docker-compose 預設一致：vLLM Qwen2.5-VL-7B-AWQ。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config import config

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_ID = "vllm_qwen25"

# 與前端 model_type + qwen_model 對齊
VLM_PROFILES: Dict[str, Dict[str, Any]] = {
    "ollama_qwen25": {
        "label": "Ollama · qwen2.5vl:latest",
        "model_type": "qwen",
        "qwen_model": "qwen2.5vl:latest",
        "stop_services": ("vllm", "vllm-qwen3"),
        "start_services": ("ollama",),
        "ready_check": "ollama_qwen",
    },
    "vllm_qwen25": {
        "label": "vLLM · Qwen2.5-VL-7B-Instruct-AWQ",
        "model_type": "vllm_qwen",
        "qwen_model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "stop_services": ("vllm-qwen3",),
        "start_services": ("vllm",),
        "ready_check": "vllm_main",
    },
    "vllm_qwen3": {
        "label": "vLLM · Qwen3-VL-8B-Instruct-FP8",
        "model_type": "vllm_qwen",
        "qwen_model": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "stop_services": ("vllm",),
        "start_services": ("vllm-qwen3",),
        "ready_check": "vllm_qwen3",
    },
}

_switch_lock = threading.Lock()
_switch_state: Dict[str, Any] = {
    "phase": "idle",
    "message": "",
    "since": None,
    "target_profile": None,
    "last_error": None,
}


def _profile_path() -> Path:
    return Path(config.SEGMENT_DIR) / ".vlm_profile.json"


def read_selected_profile_id() -> str:
    path = _profile_path()
    if not path.is_file():
        return DEFAULT_PROFILE_ID
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = data.get("profile_id") or DEFAULT_PROFILE_ID
        return pid if pid in VLM_PROFILES else DEFAULT_PROFILE_ID
    except Exception:
        return DEFAULT_PROFILE_ID


def write_selected_profile_id(profile_id: str) -> None:
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"profile_id": profile_id, "updated_at": int(time.time())}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def profile_id_from_ui(model_type: str, qwen_model: str) -> Optional[str]:
    if model_type == "qwen" and (qwen_model or "").strip() in ("qwen2.5vl:latest",):
        return "ollama_qwen25"
    if model_type == "vllm_qwen":
        m = (qwen_model or "").strip()
        if "Qwen3" in m or "qwen3" in m.lower():
            return "vllm_qwen3"
        return "vllm_qwen25"
    return None


def ui_from_profile_id(profile_id: str) -> Tuple[str, str]:
    p = VLM_PROFILES.get(profile_id) or VLM_PROFILES[DEFAULT_PROFILE_ID]
    return p["model_type"], p["qwen_model"]


def _vllm_headers() -> Dict[str, str]:
    h: Dict[str, str] = {}
    key = getattr(config, "VLLM_API_KEY", None) or os.getenv("VLLM_API_KEY")
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _friendly_request_error(
    exc: Exception,
    endpoint_label: str,
    url_hint: str = "",
) -> str:
    """將 requests/urllib3 長錯誤轉成可讀說明（常見：Docker DNS 被覆寫導致無法解析服務名）。"""
    msg = str(exc)
    low = msg.lower()
    hint = f" {url_hint}" if url_hint else ""
    if "failed to resolve" in low or "nameresolutionerror" in low or "name or service not known" in low:
        return (
            f"{endpoint_label}：無法解析主機名（DNS）。"
            "若 backend 在 Docker 內：請移除 compose 中 backend 的整段 dns: 設定（勿用 8.8.8.8 覆寫），"
            "重建容器後服務名 vllm 才可解析；或於 .env 將 VLLM_BASE 改為可連線的位址。"
            "若在本機直跑 backend：VLLM_BASE 請用 http://127.0.0.1:<埠>，勿使用 vllm 主機名。"
        )
    if "connection refused" in low or "errno 111" in low:
        if endpoint_label == "vLLM":
            return (
                f"vLLM 連線被拒{hint}。"
                "請 docker compose up -d vllm（Qwen3 用 vllm-qwen3）並等模型載入；若要用 Ollama 請改選「Qwen (Multimodal via Ollama)」。"
            )
        return f"{endpoint_label}：連線被拒{hint}（服務未啟動或埠錯誤）。請確認 Ollama / 對應容器已啟動。"
    if "timed out" in low or "timeout" in low:
        if endpoint_label == "vLLM":
            return f"vLLM 連線逾時{hint}。模型可能仍在載入，請稍候或查看 vllm 容器日誌。"
        return f"{endpoint_label}：連線逾時{hint}。"
    return f"{endpoint_label}：{msg[:400]}"


def probe_ollama_tags() -> Dict[str, Any]:
    base = (config.OLLAMA_BASE or "").rstrip("/")
    out: Dict[str, Any] = {"ok": False, "base": base, "models": [], "error": None}
    if not base:
        out["error"] = "OLLAMA_BASE 未設定"
        return out
    try:
        r = requests.get(f"{base}/api/tags", timeout=5)
        r.raise_for_status()
        data = r.json() or {}
        names: List[str] = []
        for m in data.get("models") or []:
            if isinstance(m, dict) and m.get("name"):
                names.append(str(m["name"]))
        out["ok"] = True
        out["models"] = names
    except Exception as e:
        out["error"] = _friendly_request_error(e, "Ollama", f"→ {base}/api/tags")
    return out


def probe_vllm_models(base_url: str) -> Dict[str, Any]:
    base = (base_url or "").rstrip("/")
    out: Dict[str, Any] = {"ok": False, "base": base, "model_ids": [], "error": None}
    if not base:
        out["error"] = "base 空白"
        return out
    try:
        r = requests.get(f"{base}/v1/models", headers=_vllm_headers(), timeout=8)
        r.raise_for_status()
        data = r.json() or {}
        ids: List[str] = []
        for item in data.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                ids.append(str(item["id"]))
        out["ok"] = True
        out["model_ids"] = ids
    except Exception as e:
        out["error"] = _friendly_request_error(e, "vLLM", f"→ {base}/v1/models")
    return out


def _ollama_unload_model(model_name: str) -> None:
    """盡量卸載 Ollama 記憶體中的模型：官方慣用 keep_alive=0（欄位為 model，非 name）。"""
    base = (config.OLLAMA_BASE or "").rstrip("/")
    if not base or not model_name:
        return
    try:
        r = requests.post(
            f"{base}/api/generate",
            json={"model": model_name, "prompt": " ", "keep_alive": 0, "stream": False},
            timeout=30,
        )
        logger.info("Ollama unload /api/generate model=%s http=%s", model_name, r.status_code)
        requests.post(
            f"{base}/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": " "}],
                "keep_alive": 0,
                "stream": False,
            },
            timeout=30,
        )
    except Exception as e:
        logger.warning("Ollama unload failed model=%s: %s", model_name, e)


def unload_ollama_vlm_for_vllm_switch() -> None:
    """切到 vLLM 時盡量釋放 Ollama 上的 VLM 權重（保留 embed 模型與否視 Ollama 行為而定）。"""
    for name in (
        getattr(config, "OLLAMA_LLM_MODEL", None) or "qwen2.5vl:latest",
        "qwen2.5vl:latest",
        "qwen3-vl:8b",
    ):
        _ollama_unload_model(name)


def _compose_bin() -> Optional[List[str]]:
    """回傳 compose 指令前綴，例如 ['docker-compose'] 或 ['docker', 'compose']。
    Dockerfile 安裝的是靜態 docker 二進位（通常不含 compose 外掛）＋獨立 docker-compose，
    若優先使用 docker compose 可能出現 unknown flag -p；故先嘗試 docker-compose。
    """
    if shutil.which("docker-compose"):
        try:
            subprocess.run(
                ["docker-compose", "version"],
                capture_output=True,
                timeout=5,
                check=True,
            )
            return ["docker-compose"]
        except Exception:
            pass
    if shutil.which("docker"):
        try:
            subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                timeout=5,
                check=True,
            )
            return ["docker", "compose"]
        except Exception:
            pass
    return None


def orchestration_configured() -> bool:
    if not getattr(config, "VLM_ORCHESTRATION_ENABLED", False):
        return False
    cf = (getattr(config, "VLM_COMPOSE_FILE", None) or "").strip()
    if not cf or not os.path.isfile(cf):
        return False
    return True


def _compose_project_dir() -> str:
    pd = (getattr(config, "VLM_COMPOSE_PROJECT_DIR", None) or "").strip()
    if pd:
        return pd
    cf = (getattr(config, "VLM_COMPOSE_FILE", None) or "").strip()
    return os.path.dirname(cf) or "."


def _compose_file_and_project_flags() -> List[str]:
    """docker compose 共用參數：-p 專案名（若設）+ -f + --project-directory。"""
    cf = (getattr(config, "VLM_COMPOSE_FILE", None) or "").strip()
    pd = _compose_project_dir()
    pname = (getattr(config, "VLM_COMPOSE_PROJECT_NAME", None) or "").strip()
    out: List[str] = []
    if pname:
        out.extend(["-p", pname])
    out.extend(["-f", cf, "--project-directory", pd])
    return out


def run_compose(args: List[str], timeout: int = 180) -> Tuple[int, str, str]:
    cf = (getattr(config, "VLM_COMPOSE_FILE", None) or "").strip()
    prefix = _compose_bin()
    if not cf or not prefix:
        return -1, "", "docker compose 未設定或無 docker CLI"
    cmd = prefix + _compose_file_and_project_flags() + args
    logger.info("run_compose: %s", " ".join(cmd))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            logger.warning("run_compose failed rc=%s stderr=%s", p.returncode, (p.stderr or "")[:2000])
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "docker compose 逾時"
    except Exception as e:
        return -1, "", str(e)


def run_compose_stop_service(svc: str) -> Tuple[bool, str]:
    """
    docker compose stop <svc>。若 rc≠0 但該 service 已不在專案 running 清單內（常見：從未 up 過、
    或舊版 compose 對「無容器可停」仍回傳錯誤），視為成功以免中斷整段編排。
    """
    rc, out, err = run_compose(["stop", svc])
    if rc == 0:
        return True, ""
    detail = ((out or "").strip() + "\n" + (err or "").strip()).strip()
    running = _compose_running_service_names()
    if running is not None and svc not in running:
        logger.warning(
            "compose stop %s 回 rc=%s，但專案內無此 service 在 running，視為可繼續。輸出: %s",
            svc,
            rc,
            detail[:800] or "(empty)",
        )
        return True, detail
    low = detail.lower()
    for frag in (
        "no such service",
        "unknown service",
        "no container",
        "no resource found to stop",
        "nothing to stop",
    ):
        if frag in low:
            logger.warning("compose stop %s 失敗訊息含「%s」，視為可忽略: %s", svc, frag, detail[:600])
            return True, detail
    return False, detail or f"docker compose stop {svc} 失敗 rc={rc}"


def probe_docker_cli() -> Tuple[bool, str]:
    """
    確認 backend 容器內可執行 docker（含 docker.sock 掛載）。
    回傳 (成功, 說明或錯誤摘要)。
    """
    if not docker_socket_present():
        return False, "/var/run/docker.sock 不存在（未掛載 docker.sock）"
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False, "PATH 內找不到 docker 執行檔（Command not found）"
    try:
        p = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if p.returncode != 0:
            err = (p.stderr or p.stdout or "").strip()
            return False, f"docker ps 失敗 rc={p.returncode}: {err[:500]}"
        lines = (p.stdout or "").strip().splitlines()
        head = lines[0] if lines else "(empty)"
        logger.info("probe_docker_cli: OK, docker ps 首行: %s", head[:200])
        return True, "docker ps 成功"
    except PermissionError as e:
        return False, f"Permission denied（docker.sock 權限）: {e}"
    except FileNotFoundError:
        return False, "docker 執行檔不存在"
    except Exception as e:
        return False, str(e)


def _compose_running_service_names() -> Optional[set]:
    """目前 compose 專案中仍為 running 的 service 名稱集合；若無法查詢則回傳 None。"""
    cf = (getattr(config, "VLM_COMPOSE_FILE", None) or "").strip()
    prefix = _compose_bin()
    if not cf or not prefix:
        return set()
    cmd = prefix + _compose_file_and_project_flags() + [
        "ps",
        "--status",
        "running",
        "--format",
        "{{.Service}}",
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        if p.returncode != 0:
            logger.warning(
                "compose ps --status running failed rc=%s err=%s",
                p.returncode,
                (p.stderr or "")[:800],
            )
            return None
        return {x.strip() for x in (p.stdout or "").splitlines() if x.strip()}
    except Exception as e:
        logger.warning("compose ps running: %s", e)
        return None


def wait_until_compose_services_not_running(
    service_names: Tuple[str, ...],
    timeout_sec: int = 180,
) -> Tuple[bool, str]:
    """在 stop 之後輪詢，直到列出的 service 皆不在 running 狀態（利於釋放 VRAM）。"""
    want = set(service_names)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        running = _compose_running_service_names()
        if running is None:
            logger.warning("VLM orchestration: 無法取得 compose ps，2s 後重試…")
            time.sleep(2)
            continue
        still = want & running
        if not still:
            return True, "ok"
        logger.info("VLM orchestration: 等待容器停止，仍 running: %s", still)
        time.sleep(1.5)
    running = _compose_running_service_names()
    if running is None:
        return False, "逾時：compose ps 持續失敗，無法確認容器狀態"
    still = want & running
    return False, f"逾時：以下服務仍未退出 running: {still}"


def docker_socket_present() -> bool:
    return os.path.exists("/var/run/docker.sock")


def readiness_from_probes(
    selected_profile_id: str,
    ollama_p: Dict[str, Any],
    vllm_main: Dict[str, Any],
    vllm_q3: Dict[str, Any],
) -> Tuple[bool, str]:
    """與 profile_ready 邏輯一致，但使用已快取的探測結果（避免 build_status_payload 重複 HTTP）。"""
    spec = VLM_PROFILES.get(selected_profile_id)
    if not spec:
        return False, "未知 profile"
    check = spec["ready_check"]
    if check == "ollama_qwen":
        if not ollama_p.get("ok"):
            return False, ollama_p.get("error") or "Ollama 無法連線"
        want = spec["qwen_model"]
        models = ollama_p.get("models") or []
        if any(want in n or n.endswith(want) for n in models):
            return True, "Ollama 已就緒（含目標模型標籤）"
        return False, f"Ollama 可用但尚未見到模型標籤: {want}"
    if check == "vllm_main":
        if not vllm_main.get("ok"):
            return False, vllm_main.get("error") or "vLLM(主服務)無法連線"
        return True, "vLLM(8440/bridge) 已回應 /v1/models"
    if check == "vllm_qwen3":
        if not vllm_q3.get("ok"):
            return False, vllm_q3.get("error") or "vLLM(Qwen3)無法連線"
        return True, "vLLM(Qwen3) 已回應 /v1/models"
    return False, "未實作 ready_check"


def profile_ready(profile_id: str) -> Tuple[bool, str]:
    spec = VLM_PROFILES.get(profile_id)
    if not spec:
        return False, "未知 profile"
    check = spec["ready_check"]
    if check == "ollama_qwen":
        pr = probe_ollama_tags()
        if not pr["ok"]:
            return False, pr.get("error") or "Ollama 無法連線"
        want = spec["qwen_model"]
        if any(want in n or n.endswith(want) for n in pr["models"]):
            return True, "Ollama 已就緒（含目標模型標籤）"
        return False, f"Ollama 可用但尚未見到模型標籤: {want}"
    if check == "vllm_main":
        pr = probe_vllm_models(config.VLLM_BASE)
        if not pr["ok"]:
            return False, pr.get("error") or "vLLM(主服務)無法連線"
        return True, "vLLM(8440/bridge) 已回應 /v1/models"
    if check == "vllm_qwen3":
        base = getattr(config, "QWEN3_VLLM_BASE", None) or config.VLLM_BASE
        pr = probe_vllm_models(base)
        if not pr["ok"]:
            return False, pr.get("error") or "vLLM(Qwen3)無法連線"
        return True, "vLLM(Qwen3) 已回應 /v1/models"
    return False, "未實作 ready_check"


def get_switch_state() -> Dict[str, Any]:
    with _switch_lock:
        return dict(_switch_state)


def _set_switch(phase: str, message: str = "", target: Optional[str] = None, err: Optional[str] = None) -> None:
    global _switch_state
    with _switch_lock:
        _switch_state["phase"] = phase
        _switch_state["message"] = message
        _switch_state["since"] = int(time.time()) if phase == "loading" else _switch_state.get("since")
        _switch_state["target_profile"] = target
        _switch_state["last_error"] = err


def _switch_worker(profile_id: str) -> None:
    spec = VLM_PROFILES.get(profile_id)
    if not spec:
        _set_switch("error", "未知 profile", None, "unknown profile")
        return
    _set_switch("loading", f"正在切換至 {spec['label']}…", profile_id, None)
    write_selected_profile_id(profile_id)
    try:
        if profile_id.startswith("vllm_"):
            unload_ollama_vlm_for_vllm_switch()
        if orchestration_configured():
            cf = (getattr(config, "VLM_COMPOSE_FILE", None) or "").strip()
            pd = _compose_project_dir()
            logger.info(
                "VLM orchestration 開始: VLM_COMPOSE_FILE=%s VLM_COMPOSE_PROJECT_DIR=%s (實際 project-directory=%s)",
                cf,
                getattr(config, "VLM_COMPOSE_PROJECT_DIR", None) or "",
                pd,
            )
            ok_docker, docker_msg = probe_docker_cli()
            if not ok_docker:
                logger.error("VLM orchestration: docker 探測失敗（將仍嘗試 compose）: %s", docker_msg)
            else:
                logger.info("VLM orchestration: docker 探測成功: %s", docker_msg)

            stop_services = list(spec["stop_services"])
            if (
                profile_id.startswith("vllm_")
                and getattr(config, "VLM_ORCHESTRATION_STOP_OLLAMA_ON_VLLM", False)
                and "ollama" not in stop_services
            ):
                stop_services.append("ollama")
                logger.info(
                    "VLM orchestration: VLM_ORCHESTRATION_STOP_OLLAMA_ON_VLLM 已啟用，將一併停止 ollama 以釋放 GPU"
                )

            for svc in stop_services:
                ok_stop, stop_msg = run_compose_stop_service(svc)
                if not ok_stop:
                    _set_switch("error", f"停止 {svc} 失敗", profile_id, stop_msg[:2000])
                    return
                ok_wait, wmsg = wait_until_compose_services_not_running((svc,), timeout_sec=120)
                if not ok_wait:
                    _set_switch("error", f"停止 {svc} 後仍偵測到 running：{wmsg}", profile_id, wmsg[:2000])
                    return
                logger.info("VLM orchestration: %s 已非 running，可繼續", svc)

            ok_all, wmsg2 = wait_until_compose_services_not_running(tuple(stop_services), timeout_sec=60)
            if not ok_all:
                logger.warning("VLM orchestration: 啟動前複查 stop 狀態: %s", wmsg2)
            time.sleep(2)

            for svc in spec["start_services"]:
                rc, out, err = run_compose(["up", "-d", svc])
                if rc != 0:
                    _set_switch("error", f"啟動 {svc} 失敗", profile_id, (err or out)[:2000])
                    return
            deadline = time.time() + 240
            while time.time() < deadline:
                ok, detail = profile_ready(profile_id)
                if ok:
                    _set_switch("idle", detail, None, None)
                    return
                time.sleep(5)
            _set_switch(
                "error",
                "服務已送出啟動指令，但在時限內仍未偵測到就緒（請檢查 GPU / 日誌）",
                profile_id,
                "timeout",
            )
        else:
            _set_switch(
                "idle",
                "已儲存偏好設定；未啟用 Docker 編排（請設定 VLM_ORCHESTRATION_ENABLED 與 VLM_COMPOSE_FILE，"
                "並掛載 /var/run/docker.sock），或於主機手動 docker compose stop/start 對應服務。",
                None,
                None,
            )
    except Exception as e:
        _set_switch("error", str(e), profile_id, str(e))


def request_profile_switch(profile_id: str) -> Tuple[bool, str]:
    if profile_id not in VLM_PROFILES:
        return False, "無效的 profile_id"
    with _switch_lock:
        if _switch_state["phase"] == "loading":
            return False, "已有切換作業進行中，請稍候"
    logger.info("request_profile_switch: profile_id=%s orchestration=%s", profile_id, orchestration_configured())
    ok_d, msg_d = probe_docker_cli()
    logger.info("request_profile_switch: docker_cli_probe ok=%s detail=%s", ok_d, msg_d)
    t = threading.Thread(target=_switch_worker, args=(profile_id,), daemon=True)
    t.start()
    return True, "已接受切換請求"


def build_status_payload() -> Dict[str, Any]:
    selected = read_selected_profile_id()
    q3_base = getattr(config, "QWEN3_VLLM_BASE", None) or config.VLLM_BASE
    # 選定 vLLM 時略過 Ollama /api/tags，避免單卡環境下不必要喚醒 Ollama；Ollama profile 仍會探測
    if selected.startswith("vllm_"):
        ollama_p = {
            "ok": False,
            "base": (config.OLLAMA_BASE or "").rstrip("/"),
            "models": [],
            "error": None,
            "probe_skipped": True,
            "skipped_reason": "目前選定 vLLM profile，已略過 Ollama 探測",
        }
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_vllm_main = pool.submit(probe_vllm_models, config.VLLM_BASE)
            f_vllm_q3 = pool.submit(probe_vllm_models, q3_base)
            vllm_main = f_vllm_main.result()
            vllm_q3 = f_vllm_q3.result()
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_ollama = pool.submit(probe_ollama_tags)
            f_vllm_main = pool.submit(probe_vllm_models, config.VLLM_BASE)
            f_vllm_q3 = pool.submit(probe_vllm_models, q3_base)
            ollama_p = f_ollama.result()
            vllm_main = f_vllm_main.result()
            vllm_q3 = f_vllm_q3.result()
    ready, ready_detail = readiness_from_probes(selected, ollama_p, vllm_main, vllm_q3)
    sw = get_switch_state()
    if sw.get("phase") == "loading":
        ready = False
        ready_detail = sw.get("message") or "正在切換 VLM 後端…"
    elif sw.get("phase") == "error" and sw.get("last_error"):
        ready = False
        ready_detail = sw.get("message") or str(sw.get("last_error"))
    profiles = []
    for pid, spec in VLM_PROFILES.items():
        profiles.append(
            {
                "id": pid,
                "label": spec["label"],
                "model_type": spec["model_type"],
                "qwen_model": spec["qwen_model"],
            }
        )
    return {
        "default_profile_id": DEFAULT_PROFILE_ID,
        "selected_profile_id": selected,
        "profiles": profiles,
        "probes": {
            "ollama": ollama_p,
            "vllm_main": vllm_main,
            "vllm_qwen3": vllm_q3,
        },
        "readiness": {
            "profile_id": selected,
            "ready": ready,
            "detail": ready_detail,
        },
        "switch": sw,
        "orchestration": {
            "enabled": getattr(config, "VLM_ORCHESTRATION_ENABLED", False),
            "compose_file_set": bool((getattr(config, "VLM_COMPOSE_FILE", None) or "").strip()),
            "compose_file_exists": os.path.isfile((getattr(config, "VLM_COMPOSE_FILE", None) or "").strip())
            if (getattr(config, "VLM_COMPOSE_FILE", None) or "").strip()
            else False,
            "compose_project_name_set": bool((getattr(config, "VLM_COMPOSE_PROJECT_NAME", None) or "").strip()),
            "docker_socket_present": docker_socket_present(),
            "compose_cli_available": _compose_bin() is not None,
            "stop_ollama_on_vllm": getattr(config, "VLM_ORCHESTRATION_STOP_OLLAMA_ON_VLLM", False),
        },
    }
