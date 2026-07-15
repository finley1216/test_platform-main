# -*- coding: utf-8 -*-
"""
LLR 管線硬門檻覆寫（不修改 path_enum_scoring.py 原文）
====================================================
日期：2026-07-15

修正一依據（GT 更正後，人員 0507，CALIB_SOURCE=GT_20260507）：
  - emb|same ≈ Normal(μ=0.917, σ=0.023)
  - emb|diff ≈ Normal(μ=0.874, σ=0.029)（見 calibration_gt0507_report）
  - 舊 EMB_EDGE_MIN=0.91 ≈ μ_same − 0.3σ，構造性拒絕約 38% 真轉移
    （瓶頸例：22_22→07_112 emb=0.859；07_112→01_50 emb=0.897）
  - 改為 0.80 ≈ μ_diff − 2.5σ：僅作粗理智檢查，外觀鑑別交給 LLR_emb

執行時以 apply_llr_emb_gates() 覆寫 pes.EMB_EDGE_MIN / pes.EMB_HIST_MIN。
舊法 path_enum_scoring 主程式仍使用原檔預設 0.91/0.90。

---
dt 軟計分（2026-07-15）：
  tau／通行時間無本場景實測；transit 邊 LLR_dt 可經 --dt-scoring off 停用。
  硬規則（時間順序／MIN_TRANSIT／DT_MAX）不受影響。
"""

from __future__ import annotations

import path_enum_scoring as pes

# 原始預設（僅供對照／還原）
ORIGINAL_EMB_EDGE_MIN = 0.91
ORIGINAL_EMB_HIST_MIN = 0.90

# 修正一
LLR_EMB_EDGE_MIN = 0.80
LLR_EMB_HIST_MIN = 0.80

RATIONALE = (
    "2026-07-15：emb|same N(0.917,0.023)；EMB_EDGE_MIN=0.91 為 μ−0.3σ，"
    "構造性拒真轉移；改 0.80≈μ_diff−2.5σ，鑑別交給 LLR_emb。"
)

DT_SCORING_RATIONALE = (
    "2026-07-15：tau 無本場景實測來源；transit 邊 LLR_dt 軟證據自即日起可停用"
    "（--dt-scoring off）。硬規則不動；handoff 本來不算 LLR_dt。"
)


def apply_llr_emb_gates(enabled: bool = True) -> dict:
    """覆寫或還原 pes 的 emb 硬門檻。回傳生效值。"""
    if enabled:
        pes.EMB_EDGE_MIN = float(LLR_EMB_EDGE_MIN)
        pes.EMB_HIST_MIN = float(LLR_EMB_HIST_MIN)
    else:
        pes.EMB_EDGE_MIN = float(ORIGINAL_EMB_EDGE_MIN)
        pes.EMB_HIST_MIN = float(ORIGINAL_EMB_HIST_MIN)
    return {
        "enabled": bool(enabled),
        "EMB_EDGE_MIN": float(pes.EMB_EDGE_MIN),
        "EMB_HIST_MIN": float(pes.EMB_HIST_MIN),
        "rationale": RATIONALE if enabled else "restored path_enum_scoring defaults",
    }
