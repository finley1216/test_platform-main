#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
RESTART_BACKEND="${2:-yes}"

if [[ -z "${MODE}" ]]; then
  echo "用法: ./scripts/switch_vllm.sh <qwen25|qwen3|qwen3-awq> [yes|no]"
  echo "例子:"
  echo "  ./scripts/switch_vllm.sh qwen25"
  echo "  ./scripts/switch_vllm.sh qwen3"
  echo "  ./scripts/switch_vllm.sh qwen3-awq"
  echo "  ./scripts/switch_vllm.sh qwen3 no   # 只切 vLLM，不重啟 backend"
  exit 1
fi

if [[ "${RESTART_BACKEND}" != "yes" && "${RESTART_BACKEND}" != "no" ]]; then
  echo "第二個參數只能是 yes 或 no"
  exit 1
fi

COMPOSE_FILE="docker-compose.yml"

case "${MODE}" in
  qwen25)
    TARGET_SERVICE="vllm"
    STOP_SERVICES=("vllm-qwen3" "vllm-qwen3-awq")
    TARGET_BASE="http://vllm:8440"
    ;;
  qwen3)
    TARGET_SERVICE="vllm-qwen3"
    STOP_SERVICES=("vllm" "vllm-qwen3-awq")
    TARGET_BASE="http://host.docker.internal:8441"
    ;;
  qwen3-awq)
    TARGET_SERVICE="vllm-qwen3-awq"
    STOP_SERVICES=("vllm" "vllm-qwen3")
    TARGET_BASE="http://host.docker.internal:8442"
    ;;
  *)
    echo "不支援的模式: ${MODE}"
    echo "可用模式: qwen25 | qwen3 | qwen3-awq"
    exit 1
    ;;
esac

echo "[1/4] 停用非目標服務: ${STOP_SERVICES[*]} ..."
docker compose -f "${COMPOSE_FILE}" stop "${STOP_SERVICES[@]}" >/dev/null || true

echo "[2/4] 啟用 ${TARGET_SERVICE} ..."
docker compose -f "${COMPOSE_FILE}" up -d "${TARGET_SERVICE}"

if [[ "${RESTART_BACKEND}" == "yes" ]]; then
  echo "[3/4] 重啟 backend 並套用 VLLM_BASE=${TARGET_BASE} ..."
  VLLM_BASE="${TARGET_BASE}" docker compose -f "${COMPOSE_FILE}" up -d --force-recreate backend
else
  echo "[3/4] 跳過 backend 重啟"
fi

echo "[4/4] 目前服務狀態："
docker compose -f "${COMPOSE_FILE}" ps vllm vllm-qwen3 vllm-qwen3-awq backend

echo
echo "完成。當前模式: ${MODE}"
echo "建議檢查："
echo "  - docker compose -f ${COMPOSE_FILE} logs -f ${TARGET_SERVICE}"
echo "  - backend 使用的 VLLM_BASE: ${TARGET_BASE}"
