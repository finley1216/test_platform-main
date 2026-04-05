#!/usr/bin/env bash
# 驗證 VLM 編排三點：專案名、compose 路徑（主機 + backend 容器內）、docker 可用性
# 用法：在 test_platform-main 目錄執行  ./scripts/verify_vlm_orchestration_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "========== 0) 工作目錄 =========="
echo "$ROOT"

echo ""
echo "========== 1) 專案名：.env 的 VLM_COMPOSE_PROJECT_NAME vs docker compose ls =========="
ENV_NAME=""
if [[ -f .env ]]; then
  ENV_NAME=$(grep -E '^VLM_COMPOSE_PROJECT_NAME=' .env | cut -d= -f2- | tr -d '\r' || true)
fi
echo "VLM_COMPOSE_PROJECT_NAME (from .env) = ${ENV_NAME:-<unset>}"
echo "--- docker compose ls ---"
docker compose ls 2>&1 || { echo "FAIL: docker compose ls"; exit 1; }
if [[ -n "${ENV_NAME:-}" ]]; then
  if docker compose ls 2>/dev/null | awk 'NR>1 {print $1}' | grep -qxF "$ENV_NAME"; then
    echo "OK: 專案名「$ENV_NAME」出現在 docker compose ls"
  else
    echo "WARN: 專案名「$ENV_NAME」未出現在 docker compose ls（請確認 stack 是否用此名啟動）"
  fi
fi

echo ""
echo "========== 2) compose 檔與 project-directory（主機路徑，等同後端在容器內用掛載路徑）=========="
COMPOSE_REL="docker-compose.yml"
if [[ ! -f "$COMPOSE_REL" ]]; then
  echo "FAIL: 找不到 $COMPOSE_REL"
  exit 1
fi
echo "主機 compose 檔: $ROOT/$COMPOSE_REL"
echo "--- 服務名稱（須含 vllm、vllm-qwen3、vllm-qwen3-awq、ollama、backend）---"
docker compose -p "${ENV_NAME:-test_platform-main}" -f "$COMPOSE_REL" --project-directory "$ROOT" config --services 2>&1

echo "--- ps（-p 與 .env 一致時）---"
docker compose -p "${ENV_NAME:-test_platform-main}" -f "$COMPOSE_REL" --project-directory "$ROOT" ps -a 2>&1

echo ""
echo "========== 3) Docker CLI / daemon =========="
docker ps >/dev/null 2>&1 && echo "OK: docker ps" || { echo "FAIL: docker ps"; exit 1; }
docker info >/dev/null 2>&1 && echo "OK: docker info" || echo "WARN: docker info 失敗"

echo ""
echo "========== 4) backend 容器內（若存在）：/vlm-compose-host 與 docker-compose =========="
# 後端 Dockerfile 為靜態 docker（通常無 compose 外掛）＋獨立 docker-compose；勿用「docker compose -p」測試
BACKEND_CTN=""
BACKEND_CTN=$(docker compose -p "${ENV_NAME:-test_platform-main}" -f "$COMPOSE_REL" --project-directory "$ROOT" ps -q backend 2>/dev/null | head -1 || true)
if [[ -z "${BACKEND_CTN:-}" ]]; then
  echo "SKIP: 找不到 backend 容器（請先 docker compose up -d backend）"
else
  echo "backend container id: ${BACKEND_CTN:0:12}..."
  docker exec "$BACKEND_CTN" sh -c 'echo -n "compose file: "; test -f /vlm-compose-host/docker-compose.yml && echo OK || echo MISSING; echo -n "docker-compose: "; command -v docker-compose 2>/dev/null || echo "not found"; ls -la /var/run/docker.sock 2>&1 || true' 2>&1
  echo "--- docker ps（失敗時會印 stderr）---"
  docker exec "$BACKEND_CTN" sh -c 'docker ps 2>&1' 2>&1 | head -5
  echo "--- 在 backend 內用 docker-compose（與 vlm_profile_service._compose_bin 優先順序一致，僅 ps）---"
  docker exec "$BACKEND_CTN" sh -c "docker-compose -p ${ENV_NAME:-test_platform-main} -f /vlm-compose-host/docker-compose.yml --project-directory /vlm-compose-host ps -a 2>&1" 2>&1 | head -30 || echo "WARN: backend 內 docker-compose 失敗"
fi

echo ""
echo "========== 完成 =========="
