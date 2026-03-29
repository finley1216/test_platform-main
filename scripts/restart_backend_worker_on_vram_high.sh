#!/usr/bin/env bash
set -euo pipefail

# Restart only the backend "worker" process when its VRAM (per-process) stays
# above a threshold, but DO NOT interrupt active requests.
#
# Behavior:
# - Continuously poll GPU usage via `nvidia-smi`.
# - When a backend python process exceeds THRESHOLD_MB, mark it as pending.
# - Wait until /v1/system/status reports active_requests == 0.
# - Then kill the pending pid (worker will be respawned by gunicorn).
#
# Usage:
#   ./scripts/restart_backend_worker_on_vram_high.sh [THRESHOLD_MB] [INTERVAL_SEC] [COOLDOWN_SEC]
#
# Example:
#   ./scripts/restart_backend_worker_on_vram_high.sh 1600 5 120

THRESHOLD_MB="${1:-1600}"
INTERVAL_SEC="${2:-1}"
COOLDOWN_SEC="${3:-15}"

COMPOSE_FILE="docker-compose.yml"
BACKEND_SERVICE="backend"

pending_pid=""
last_restart_ts=0

log() {
  echo "[$(date '+%F %T')] $*"
}

get_active_requests() {
  # 預設：用 docker compose exec 直接查容器內的 http://127.0.0.1:8080/v1/system/status
  # 由於你的 backend 8080 可能未對外暴露，主機端預設無法直連，因此不做預設 host 直撥。
  #
  # 若你確定 backend 端點可以從主機直連，請用環境變數 ACTIVE_URL 覆蓋。
  local active_url="${ACTIVE_URL:-}"
  if [[ -n "${active_url}" ]]; then
    python3 - "${active_url}" <<'PY' 2>/dev/null || echo -1
import json, urllib.request, sys
url = sys.argv[1]
data = json.loads(urllib.request.urlopen(url, timeout=1.5).read().decode("utf-8"))
print(int(data.get("active_requests", 0)))
PY
    return
  fi

  docker compose -f "${COMPOSE_FILE}" exec -T "${BACKEND_SERVICE}" python3 -c \
    "import json,urllib.request; url='http://127.0.0.1:8080/v1/system/status'; data=json.loads(urllib.request.urlopen(url, timeout=1.5).read().decode('utf-8')); print(int(data.get('active_requests', 0)))" \
    2>/dev/null || echo -1
}

get_max_python_pid_over_threshold() {
  # Output: pid,used_mb,process_name (or 0,0,)
  local threshold_mb="$1"

  # nvidia-smi output lines like: "814725, /usr/local/bin/python, 1618"
  local smi
  smi="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ -z "${smi}" ]] && echo "0,0," && return

  python3 - "${threshold_mb}" "${smi}" <<'PY'
import sys, csv, io

threshold = int(sys.argv[1])
raw = sys.argv[2].strip()
if not raw:
    print("0,0,")
    raise SystemExit(0)

rows = []
f = io.StringIO(raw)
reader = csv.reader(f, skipinitialspace=True)
for r in reader:
    if not r or len(r) < 3:
        continue
    try:
        pid = int(r[0])
        proc = (r[1] or "").strip()
        used = int(r[2])
    except Exception:
        continue
    rows.append((pid, proc, used))

def is_python_proc(proc: str) -> bool:
    p = (proc or "").lower()
    return ("python" in p) and ("vllm" not in p) and ("enginecore" not in p)

cands = [x for x in rows if x[2] > threshold and is_python_proc(x[1])]
if not cands:
    print("0,0,")
else:
    pid, proc, used = max(cands, key=lambda t: t[2])
    print(f"{pid},{used},{proc}")
PY
}

pid_alive() {
  local pid="$1"
  kill -0 "${pid}" 2>/dev/null
}

while true; do
  sleep "${INTERVAL_SEC}"

  max_info="$(get_max_python_pid_over_threshold "${THRESHOLD_MB}" || echo "0,0,")"
  max_pid="${max_info%%,*}"
  rest="${max_info#*,}"
  max_mb="${rest%%,*}"
  # max_proc is the remainder after second comma
  max_proc="${max_info#*,}"
  max_proc="${max_proc#*,}"

  # active_requests 可能相對耗時，只有需要判斷「是否可殺」時才查
  log "max_python_pid=${max_pid} vram=${max_mb}MB pending=${pending_pid:-none} (${max_proc})"

  now_ts="$(date +%s)"
  if [[ "${max_pid}" != "0" ]]; then
    # Mark pending (update to the latest over-threshold pid)
    pending_pid="${max_pid}"
  fi

  # Only restart when:
  # - we have a pending pid
  # - cooldown passed
  if [[ -n "${pending_pid}" ]]; then
    active_requests="$(get_active_requests || echo -1)"
    [[ "${active_requests}" == "-1" ]] && continue

    # - system is idle (active_requests == 0)
    if [[ "${active_requests}" -ne 0 ]]; then
      continue
    fi

    if (( now_ts - last_restart_ts < COOLDOWN_SEC )); then
      continue
    fi

    if pid_alive "${pending_pid}"; then
      log "VRAM above threshold and idle now -> killing worker pid=${pending_pid} ..."
      kill -TERM "${pending_pid}" >/dev/null 2>&1 || true
      last_restart_ts="${now_ts}"
      pending_pid=""
      log "worker killed. gunicorn should respawn it."
    else
      # worker already gone; clear pending
      pending_pid=""
    fi
  fi
done

