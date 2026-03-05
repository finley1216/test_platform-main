#!/bin/bash
# 監控 RAM 和 CPU 使用量，每秒輸出一次，同時寫入日誌檔
# 用法：在另一個終端執行 ./monitor_resources.sh，然後執行你的測試腳本

LOG="monitor_$(date +%Y%m%d_%H%M%S).log"
echo "監控中... 輸出同時寫入 $LOG (Ctrl+C 結束)"
echo ""

while true; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "=== $ts ==="
  # RAM: 總量、已用、可用
  free -h | grep -E "Mem|Swap"
  # CPU: 整體使用率 + 前 5 個最耗資源的進程
  echo "--- Top 5 進程 (CPU%) ---"
  ps aux --sort=-%cpu | head -6
  echo "--- Top 5 進程 (RAM%) ---"
  ps aux --sort=-%mem | head -6
  echo ""
  sleep 1
done 2>&1 | tee "$LOG"
