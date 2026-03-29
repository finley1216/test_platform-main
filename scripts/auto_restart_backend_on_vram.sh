#!/usr/bin/env bash
set -euo pipefail

# Auto-restart backend when any backend worker's GPU memory
# exceeds the threshold while no request is running.
#
# Usage:
#   ./scripts/auto_restart_backend_on_vram.sh [THRESHOLD_MB] [CHECK_INTERVAL_SEC] [COOLDOWN_SEC]
#
# Example:
#   ./scripts/auto_restart_backend_on_vram.sh 1500 5 120

THRESHOLD_MB="${1:-1500}"
CHECK_INTERVAL_SEC="${2:-5}"
COOLDOWN_SEC="${3:-120}"

COMPOSE_FILE="docker-compose.yml"
BACKEND_SERVICE="backend"
BACKEND_CONTAINER="test_platform-main-backend-1"

LAST_RESTART_TS=0

log() {
  echo "[$(date '+%F %T')] $*"
}

get_active_requests() {
  docker compose -f "${COMPOSE_FILE}" exec -T "${BACKEND_SERVICE}" /bin/sh -lc \
    'python - <<'"'"'PY'"'"'
import json
import urllib.request
try:
    data = json.loads(urllib.request.urlopen("http://127.0.0.1:8080/v1/system/status", timeout=2).read().decode("utf-8"))
    print(int(data.get("active_requests", 0)))
except Exception:
    print(-1)
PY' 2>/dev/null | tr -d '\r'
}

get_backend_gpu_max_mb() {
  # 1) backend container host pids
  local pids
  pids="$(docker top "${BACKEND_CONTAINER}" -eo pid,comm 2>/dev/null | awk 'NR>1 {print $1}')"
  if [[ -z "${pids}" ]]; then
    echo 0
    return
  fi

  # 2) compute-process gpu usage from nvidia-smi: pid, used_mb
  local smi
  smi="$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${smi}" ]]; then
    echo 0
    return
  fi

  # 3) pick max GPU memory among backend pids
  python - "$pids" "$smi" <<'PY'
import sys

pid_lines = sys.argv[1].splitlines()
smi_lines = sys.argv[2].splitlines()

backend_pids = set()
for line in pid_lines:
    line = line.strip()
    if not line:
        continue
    try:
        backend_pids.add(int(line))
    except Exception:
        pass

max_mb = 0
for row in smi_lines:
    parts = [x.strip() for x in row.split(",")]
    if len(parts) < 2:
        continue
    try:
        pid = int(parts[0])
        used = int(parts[1])
    except Exception:
        continue
    if pid in backend_pids:
        max_mb = max(max_mb, used)

print(max_mb)
PY
}

log "start monitor: threshold=${THRESHOLD_MB}MB interval=${CHECK_INTERVAL_SEC}s cooldown=${COOLDOWN_SEC}s"

while true; do
  sleep "${CHECK_INTERVAL_SEC}"

  # Backend may be restarting.
  if ! docker compose -f "${COMPOSE_FILE}" ps "${BACKEND_SERVICE}" --status running >/dev/null 2>&1; then
    continue
  fi

  active_requests="$(get_active_requests || echo -1)"
  max_backend_mb="$(get_backend_gpu_max_mb || echo 0)"
  now_ts="$(date +%s)"

  log "active_requests=${active_requests} max_backend_gpu=${max_backend_mb}MB"

  # -1 means status endpoint temporarily unavailable; skip this round.
  if [[ "${active_requests}" == "-1" ]]; then
    continue
  fi

  # Safety: never restart while handling requests.
  if (( active_requests > 0 )); then
    continue
  fi

  # Threshold check + cooldown.
  if (( max_backend_mb > THRESHOLD_MB )) && (( now_ts - LAST_RESTART_TS >= COOLDOWN_SEC )); then
    log "threshold exceeded, restarting backend..."
    docker compose -f "${COMPOSE_FILE}" restart "${BACKEND_SERVICE}" >/dev/null
    LAST_RESTART_TS="$(date +%s)"
    log "backend restarted."
  fi
done

