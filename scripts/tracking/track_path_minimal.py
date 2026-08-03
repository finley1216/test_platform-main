# -*- coding: utf-8 -*-
"""
路徑枚舉增量實驗入口（M1–M9；實驗用）
====================================
鐵則：不修改 track_path.py（= M0）／query_filter／config／calibration。
交付請用根目錄 `track_path_m9.py`；本檔僅保留 M1–M8 對照與舊流程。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

import track_path_m0 as tp


MINIMAL_RULES_M1 = [
    "時間順序：v.t_start >= u.t_end（重疊鏡頭對沿用既有 OVERLAP_PAIRS / TOL）",
    "拓撲可達：hop <= 2",
    "超節點物理合併：幾何制 H<95px；名單制 emb > 全圖邊 emb 中位數",
    "計分：邊 z_emb；節點分=0；路徑分=Σ邊",
    "極大路徑+分段假設同池、矛盾作廢、softmax",
]

M2_RULES = [
    "時間順序（含既有重疊交接容許）",
    "拓撲可達 hop <= 2",
    "超節點物理合併（H<95px；名單制 emb > 全圖邊 emb 中位數）",
    "可選 DT_MAX=130（--dtmax on|off）",
    "邊分 = z_emb + z_time + hop_pen（handoff 的 dt 以 0 計入同池標準化）",
    "  z_emb=(emb−μ_emb)/σ_emb；z_time=−(dt−μ_dt)/σ_dt；hop_pen=0(hop≤1) / −1.0(hop2，1σ單位)",
    "可選節點分 z_sim=(sim−μ_sim)/σ_sim（--node-score on|off；相對化，不查校準）",
    "路徑分 = Σ節點 + Σ邊；極大路徑+分段同池、矛盾作廢、softmax",
]

# 相容舊名
MINIMAL_RULES = MINIMAL_RULES_M1

DT_MAX_M2_ON = 130.0  # 與現行版 DT_MAX 相同；僅 M2 --dtmax on 使用

M3_RULES = [
    "時間順序（含既有重疊交接容許）",
    "拓撲可達 hop <= 2",
    "超節點物理合併（H<95px；名單制 emb > 全圖邊 emb 中位數）",
    "無 DT_MAX；無 emb 底線",
    "邊分 = z_emb + z_time + hop_pen（與 M2 同定義：全圖合法邊自標準化；hop_pen=0/0/−1）",
    "  z_emb=(emb−μ)/σ；z_time=−(dt−μ_dt)/σ_dt；hop_pen=0(hop≤1)/−1.0(hop2)",
    "節點分=0（無節點證據）",
    "路徑分=Σ邊；極大路徑+分段同池、矛盾作廢、softmax",
]

M4_RULES = [
    "時間順序（含既有重疊交接容許）",
    "拓撲可達 hop <= 2",
    "超節點物理合併（H<95px；名單制 emb > 全圖邊 emb 中位數）",
    "無 DT_MAX；無 emb 底線；無節點分；無全圖統計；無手調常數",
    "A = ln( ((w_u+w_v)/2) / d_uv )："
    "  w = kept crops 兩兩 cosine distance 平均；"
    "  d_uv = 代表向量（kept 平均 emb）cosine distance（1−cos）",
    "C = ln(N_u·P(v|u)) + ln(N_v_pred·P(u|v))；P 為對 A 的 softmax",
    "M = ln(1+miss)；miss = 空檔內其他合法下家數 + (hop2?1:0)",
    "邊分 = A + C − M；路徑分 = Σ邊；極大路徑+分段同池、矛盾作廢、softmax",
]

M4B_RULES = M4_RULES + [
    "M4b 假設層（計分不變）：Σ=A+C−M < 0 的邊為「可斷點」",
    "  對高分路徑產生「在該邊斷開」的分段假設（不收負分邊）；"
    "  各斷法與全縫版同池競價；斷不斷由總分裁決（零常數）",
]

M5_RULES = [
    "時間順序 only（重疊鏡頭對沿用交接容許；無 hop 限制、無 DT_MAX、無門檻）",
    "超節點物理合併照舊（H<95px；名單制 emb > 全圖邊 emb 中位數）",
    "不用 C、不用 M；邊分 = A + T",
    "A = ln(((w_u+w_v)/2)/d_uv)（同 M4 自量尺）",
    "T_u = t_end−t_start（超節點=成員聯集時長）；T̄=(T_u+T_v)/2",
    "  單幀/極短（時長 < 1 幀間隔）以另一端時長替代；兩端皆退化 → T=0 並標記",
    "R = dt / (hop_count × T̄)；hop_count=拓撲最短站數（重疊/同鏡交接=1）",
    "T_score = −max(0, ln R)；路徑分=Σ邊；極大路徑+分段同池、矛盾作廢、softmax",
]

M6_RULES = [
    "時間順序 only（無 hop 門檻、無 DT_MAX）；hop 不參與任何計分",
    "超節點物理合併照舊；極大路徑+分段同池、矛盾作廢、softmax",
    "邊分 = A + C + S（不用 M）",
    "A、C 同 M4（A 自量尺；C 雙向競爭）",
    "S = ln(1 − Σ P(w|u))；Σ 取 u 下家中 t_start < v.t_start 且 ≠v 者",
    "  Σ≥1 → 該邊 degenerate、路徑不採用",
    "排名附 min-A（路徑邊 A 最小值；不參與排名）",
]

M7_RULES = [
    "其餘全同 M6（時間順序建邊、極大路徑、矛盾作廢）",
    "邊分 = ln(emb/(1−emb)) + C + S（C、S 公式同 M6；競爭 softmax 對 logit）",
    "emb≤0 → logit=−∞；emb≥1 → logit=+∞（無額外常數）",
    "排名附 min-logit（路徑邊 logit 最小值；不參與排名）",
]

M8_RULES = [
    "其餘全同 M6（時間順序建邊、極大路徑、矛盾作廢）",
    "邊分 = C + S（無 A／無 logit；C、S 公式同 M6）",
    "競爭 softmax 改對裸 emb（cosine similarity）",
]

M9_RULES = [
    "其餘全同 M6（時間順序建邊、極大路徑、矛盾作廢、同池排名）",
    "邊分 = LLR + C + S；LLR = ln(f_same(emb)/f_diff(emb))",
    "密度取自 calibration_gt0507.pkl 的 emb_same / emb_diff",
    "不乘 shrink_w（單尺度下 w 為共同倍率，不影響排序）",
    "C、S 公式同 M6；競爭 softmax 餵各下家的 LLR；hop 不計分",
]


@dataclass
class RunConfig:
    scoring: str = "m9"  # 交付預設 m9；可選 m1|m2|m3|m4|m4b|m5|m6|m7|m8|m9
    node_score: bool = False
    dt_max: float | None = None  # None=off
    hop_pen: float = -1.0
    sim_min: float = 0.85
    variant_tag: str = ""
    calibration_path: str | None = None  # M9：emb LLR 校準；None→預設 pkl


# ============================================================
# 建邊（時間順序 + hop<=2；可選 DT_MAX；無 emb 底線）
# ============================================================

def edge_check_minimal(u: tp.Track, v: tp.Track, *, dt_max: float | None = None):
    """回傳 (ok, reason, dt, hop, emb, h_dist)。"""
    dt_raw = v.t_start - u.t_end
    key = tuple(sorted((u.cam, v.cam)))
    tol = tp.OVERLAP_PAIRS.get(key, tp.TOL)

    h_ok, h_dist = tp.same_object_h(u, v)
    if dt_raw < -tol:
        if not (h_ok or tp.corridor_prefers(u, v)):
            return False, f"時間順序（重疊 {-dt_raw:.1f}s 超過容許 {tol:.1f}s）", dt_raw, None, 0.0, h_dist
    dt = max(dt_raw, 0.0)

    hop = tp.hop_count(u.cam, v.cam)
    if hop is None:
        if h_ok and tuple(sorted((u.cam, v.cam))) in tp.ADJACENT:
            hop = 1
        else:
            return False, "拓撲不可達", dt, hop, 0.0, h_dist

    if dt_max is not None and dt > float(dt_max):
        return False, f"斷太久（dt={dt:.1f}s > DT_MAX={dt_max}）", dt, hop, 0.0, h_dist

    emb = tp.emb_sim(u, v)
    return True, "", dt, hop, emb, h_dist


def collect_track_edge_embs(tracks: list) -> list[float]:
    """track 層級、僅時間+hop 的合法邊 emb（供名單制中位數；不受 DT_MAX）。"""
    embs = []
    for u, v in itertools.permutations(tracks, 2):
        ok, _, _, _, emb, _ = edge_check_minimal(u, v, dt_max=None)
        if ok:
            embs.append(float(emb))
    return embs


def median_edge_emb(tracks: list) -> float:
    embs = collect_track_edge_embs(tracks)
    if not embs:
        return 0.0
    return float(np.median(np.asarray(embs, dtype=np.float64)))


def _best_member_edge_minimal(
    sa: tp.SuperNode, sb: tp.SuperNode, *, dt_max: float | None = None
):
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            key = tuple(sorted((u.cam, v.cam)))
            tol = tp.OVERLAP_PAIRS.get(key, tp.TOL)
            h_ok, h_dist = tp.same_object_h(u, v)

            if dt_raw < -tol:
                if not (h_ok or tp.corridor_prefers(u, v)):
                    rejects.append(
                        (
                            u.tid,
                            v.tid,
                            f"時間順序（聯集重疊 {-dt_raw:.1f}s 超過容許 {tol:.1f}s）",
                        )
                    )
                    continue

            hop = tp.hop_count(u.cam, v.cam)
            if hop is None:
                if h_ok and tuple(sorted((u.cam, v.cam))) in tp.ADJACENT:
                    hop = 1
                else:
                    rejects.append((u.tid, v.tid, "拓撲不可達"))
                    continue

            if dt_max is not None and dt > float(dt_max):
                rejects.append(
                    (u.tid, v.tid, f"斷太久（聯集dt={dt:.1f}s > DT_MAX={dt_max}）")
                )
                continue

            emb = tp.emb_sim(u, v)
            cand = (u, v, dt, hop, emb, h_dist)
            if best is None or emb > best[4]:
                best = cand
    return best, rejects


def _build_succ_minimal(
    nodes: list, *, dt_max: float | None = None
) -> tuple[list, list, int, list[dict]]:
    """回傳 succ、rejects、n_legal、edge_meta_list（每邊 emb/dt/hop/handoff）。"""
    n = len(nodes)
    succ = [[] for _ in range(n)]
    rejected_edges = []
    n_legal = 0
    edge_metas: list[dict] = []
    for i, j in itertools.permutations(range(n), 2):
        sa, sb = nodes[i], nodes[j]
        if dt_max is not None and sb.t_end < sa.t_start - float(dt_max):
            continue
        best, rejects = _best_member_edge_minimal(sa, sb, dt_max=dt_max)
        if best is not None:
            u, v, dt, hop, emb, h_dist = best
            handoff = tp.is_handoff_edge(u, v, dt, h_dist)
            dt_z = 0.0 if handoff else float(dt)
            succ[i].append((j, u, v, dt, hop, emb, h_dist, handoff, dt_z))
            n_legal += 1
            edge_metas.append(
                {
                    "emb": float(emb),
                    "dt": float(dt),
                    "dt_z": dt_z,
                    "hop": hop,
                    "handoff": handoff,
                }
            )
        else:
            for r in rejects[:3]:
                rejected_edges.append(r)
    return succ, rejected_edges, n_legal, edge_metas


# ============================================================
# M4：A + C − M（無單位、無常數、無全圖統計）
# ============================================================

def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    """L2-normalized 向量的 cosine distance = 1 − cos。"""
    return float(1.0 - np.dot(a, b))


def _pairwise_mean_dist(embs: list) -> float:
    """kept crops 兩兩 cosine distance 平均；n<2 → 0（無可觀測內部波動）。"""
    n = len(embs)
    if n < 2:
        return 0.0
    s = 0.0
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += _cos_dist(embs[i], embs[j])
            c += 1
    return s / c


def attach_crop_embs(tracks: list, merge_dir: Path) -> None:
    """在 Track.meta 寫入 crop_embs / w_intra（不改 track_path.load_tracks）。"""
    cache_by_cam: dict = {}
    for t in tracks:
        cam, tid_s = t.tid.rsplit("_", 1)
        tid = int(tid_s)
        if cam not in cache_by_cam:
            cache_by_cam[cam] = tp._load_emb_cache(merge_dir, cam)
        cache = cache_by_cam[cam]
        filt = (
            merge_dir
            / "filter_results"
            / f"{merge_dir.name}_{cam}_crop08_botsort09"
            / f"track_{tid}_filter_result.json"
        )
        names: list[str] = []
        if filt.is_file():
            kept = json.loads(filt.read_text(encoding="utf-8")).get("kept") or []
            names = [Path(x).name for x in kept]
        embs = []
        for name in names:
            e = tp._cache_lookup(cache, name)
            if e is not None:
                embs.append(tp._l2_normalize(np.asarray(e, dtype=np.float64)))
        if not embs and t.emb is not None:
            embs = [np.asarray(t.emb, dtype=np.float64)]
        t.meta["crop_embs"] = embs
        t.meta["w_intra"] = _pairwise_mean_dist(embs)


def m4_A(u: tp.Track, v: tp.Track) -> float:
    """
    A = ln( ((w_u+w_v)/2) / d_uv )
    d_uv：代表向量（Track.emb = kept 平均）cosine distance。
    """
    w_u = float(u.meta.get("w_intra", 0.0))
    w_v = float(v.meta.get("w_intra", 0.0))
    num = 0.5 * (w_u + w_v)
    if u.emb is None or v.emb is None:
        raise RuntimeError(f"M4 異常：缺 embedding {u.tid} / {v.tid}")
    d_uv = _cos_dist(np.asarray(u.emb), np.asarray(v.emb))
    if d_uv == 0.0:
        # 代表向量完全相同：比值 → +∞；對稱 0/0 → 0
        return 0.0 if num == 0.0 else float("inf")
    if num == 0.0:
        return float("-inf")
    return float(math.log(num / d_uv))


def _softmax_probs(scores: list[float]) -> list[float]:
    """對 A 做 softmax；+∞ 均分、全 −∞ 均分；無額外常數。"""
    n = len(scores)
    if n == 0:
        return []
    pos_inf = [i for i, s in enumerate(scores) if s == float("inf")]
    if pos_inf:
        p = [0.0] * n
        share = 1.0 / len(pos_inf)
        for i in pos_inf:
            p[i] = share
        return p
    finite = [i for i, s in enumerate(scores) if math.isfinite(s)]
    if not finite:
        return [1.0 / n] * n
    mx = max(scores[i] for i in finite)
    exps = [0.0] * n
    for i in finite:
        exps[i] = math.exp(scores[i] - mx)
    s = sum(exps)
    return [e / s for e in exps]


def _best_member_edge_m4(sa: tp.SuperNode, sb: tp.SuperNode):
    """合法成員對中取 A 最大者（建邊仍僅時間+hop）。"""
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            key = tuple(sorted((u.cam, v.cam)))
            tol = tp.OVERLAP_PAIRS.get(key, tp.TOL)
            h_ok, h_dist = tp.same_object_h(u, v)
            if dt_raw < -tol:
                if not (h_ok or tp.corridor_prefers(u, v)):
                    rejects.append((u.tid, v.tid, "時間順序"))
                    continue
            hop = tp.hop_count(u.cam, v.cam)
            if hop is None:
                if h_ok and tuple(sorted((u.cam, v.cam))) in tp.ADJACENT:
                    hop = 1
                else:
                    rejects.append((u.tid, v.tid, "拓撲不可達"))
                    continue
            a = m4_A(u, v)
            emb = tp.emb_sim(u, v)
            cand = (u, v, dt, hop, emb, h_dist, a)
            if best is None:
                best = cand
            else:
                # 取 A 較大；inf 優於 finite；同 A 取 emb 較大
                ba = best[6]
                if a == float("inf") and ba != float("inf"):
                    best = cand
                elif math.isfinite(a) and math.isfinite(ba) and (
                    a > ba or (a == ba and emb > best[4])
                ):
                    best = cand
                elif math.isfinite(a) and ba == float("-inf"):
                    best = cand
    return best, rejects


def _build_succ_m4(nodes: list) -> tuple[list, list, int, dict]:
    """
    建合法超節點邊 → 算 A → 雙向 C → M → score=A+C−M。
    回傳 succ（含完整 edge dict）、rejects、n_legal、meta。
    """
    n = len(nodes)
    # 1) 合法邊 + A
    raw = {}  # (i,j) -> (u,v,dt,hop,emb,h_dist,A)
    rejected = []
    for i, j in itertools.permutations(range(n), 2):
        best, rejects = _best_member_edge_m4(nodes[i], nodes[j])
        if best is None:
            for r in rejects[:2]:
                rejected.append(r)
            continue
        raw[(i, j)] = best

    # 2) succ / pred 索引
    succ_idx = [[] for _ in range(n)]
    pred_idx = [[] for _ in range(n)]
    for (i, j) in raw:
        succ_idx[i].append(j)
        pred_idx[j].append(i)

    # 3) 對每個起點，對下家 A 做 softmax → P(v|u)
    P_fwd = {}  # (i,j) -> p
    for i in range(n):
        js = succ_idx[i]
        if not js:
            continue
        As = [raw[(i, j)][6] for j in js]
        ps = _softmax_probs(As)
        for j, p in zip(js, ps):
            P_fwd[(i, j)] = p

    # 4) 對每個終點，對前家 A 做 softmax → P(u|v)
    P_bwd = {}  # (i,j) -> p
    for j in range(n):
        is_ = pred_idx[j]
        if not is_:
            continue
        As = [raw[(i, j)][6] for i in is_]
        ps = _softmax_probs(As)
        for i, p in zip(is_, ps):
            P_bwd[(i, j)] = p

    # 5) 組邊
    succ = [[] for _ in range(n)]
    n_legal = 0
    for (i, j), (u, v, dt, hop, emb, h_dist, A) in raw.items():
        N_u = len(succ_idx[i])
        N_v_pred = len(pred_idx[j])
        p_fwd = P_fwd.get((i, j), 0.0)
        p_bwd = P_bwd.get((i, j), 0.0)

        def _ln_NP(N, p):
            if N <= 0 or p <= 0.0:
                return float("-inf")
            return float(math.log(N * p))

        C_fwd = _ln_NP(N_u, p_fwd)
        C_bwd = _ln_NP(N_v_pred, p_bwd)
        if C_fwd == float("-inf") or C_bwd == float("-inf"):
            C = float("-inf")
        else:
            C = C_fwd + C_bwd

        # miss：u 結束後、v 開始前出現的其他合法下家
        t0 = float(nodes[i].t_end)
        t1 = float(nodes[j].t_start)
        miss_others = 0
        missed_labels = []
        for k in succ_idx[i]:
            if k == j:
                continue
            tk = float(nodes[k].t_start)
            if t0 < tk < t1:
                miss_others += 1
                missed_labels.append(nodes[k].label)
        hop_extra = 1 if int(hop) == 2 else 0
        miss = miss_others + hop_extra
        M = float(math.log(1 + miss))

        if A == float("-inf") or C == float("-inf"):
            score = float("-inf")
        elif A == float("inf"):
            score = float("inf") if C != float("-inf") else float("-inf")
        else:
            score = float(A + C - M)

        e = {
            "from": u.tid,
            "to": v.tid,
            "from_super": nodes[i].label,
            "to_super": nodes[j].label,
            "from_members": nodes[i].tids,
            "to_members": nodes[j].tids,
            "dt": float(dt),
            "dt_z": float(dt),
            "hop": hop,
            "emb": float(emb),
            "h_dist": None if h_dist is None else float(h_dist),
            "handoff": bool(tp.is_handoff_edge(u, v, dt, h_dist)),
            "score": score,
            "A": float(A) if math.isfinite(A) or A in (float("inf"), float("-inf")) else float(A),
            "C": float(C) if math.isfinite(C) or C in (float("inf"), float("-inf")) else float(C),
            "C_fwd": float(C_fwd) if math.isfinite(C_fwd) or C_fwd == float("-inf") else C_fwd,
            "C_bwd": float(C_bwd) if math.isfinite(C_bwd) or C_bwd == float("-inf") else C_bwd,
            "M": float(M),
            "miss": int(miss),
            "miss_others": int(miss_others),
            "miss_hop2": int(hop_extra),
            "missed_labels": missed_labels,
            "N_u": int(N_u),
            "N_v_pred": int(N_v_pred),
            "P_fwd": float(p_fwd),
            "P_bwd": float(p_bwd),
            "w_u": float(u.meta.get("w_intra", 0.0)),
            "w_v": float(v.meta.get("w_intra", 0.0)),
            "d_uv": _cos_dist(np.asarray(u.emb), np.asarray(v.emb)),
            "z_emb": 0.0,
            "z_time": 0.0,
            "hop_pen": 0.0,
            "scoring": "m4_A_C_minus_M",
            "dt_source": "super_union",
            "hist_emb": None,
        }
        # JSON 安全：inf → 字串標記
        for key in ("score", "A", "C", "C_fwd", "C_bwd"):
            val = e[key]
            if val == float("inf"):
                e[key] = 1e300  # 排序用大正數；報告時可還原
                e[f"{key}_inf"] = True
            elif val == float("-inf"):
                e[key] = -1e300
                e[f"{key}_inf"] = True

        succ[i].append((j, e))
        n_legal += 1

    meta = {
        "scoring": "m4",
        "d_uv_note": "代表向量（kept 平均 emb）cosine distance = 1−cos",
        "w_note": "kept crops 兩兩 cosine distance 平均；n_emb<2 → w=0",
        "constants": [],
        "n_legal_edges": n_legal,
    }
    return succ, rejected, n_legal, meta


def enumerate_paths_m4(
    tracks: list,
    merge_dir: Path,
    *,
    use_supernode: bool = True,
    beam_width: int = tp.DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = tp.DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
):
    attach_crop_embs(tracks, merge_dir)
    coexist_median = median_edge_emb(tracks)
    if use_supernode:
        nodes, super_report = tp.build_supernodes(
            tracks, overlap_emb_min=coexist_median
        )
    else:
        nodes, super_report = tp._build_nodes(tracks, False)
    super_report["minimal_coexistence_emb_median"] = coexist_median
    super_report["coexistence_overlap_emb_min"] = coexist_median

    succ_raw, rejected_edges, n_legal_edges, m4_meta = _build_succ_m4(nodes)
    # succ_raw[i] = [(j, edge_dict), ...]
    stats = {
        "emb": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "dt": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "sim": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "hop_pen": 0.0,
        "scoring": "m4",
        "node_score": False,
        "dt_max": None,
        "m4": m4_meta,
        "constants": [],
    }
    n = len(nodes)
    use_beam = (not force_full) and (n_legal_edges > tp.FULL_ENUM_EDGE_CAP)
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_max": None,
        "note": (
            f"M4 合法邊={n_legal_edges} > {tp.FULL_ENUM_EDGE_CAP}：beam"
            if use_beam
            else "M4 全量 DFS（含前綴）"
        ),
    }
    super_report["score_stats"] = stats
    super_report["z_stats"] = stats["emb"]

    # 轉成與既有 dfs 相容的 succ 結構：(j,u,v,dt,hop,emb,h_dist,handoff,dt_z) 不夠用
    # 改用 edge dict 直掛
    all_paths = []

    if not use_beam:
        def dfs(idx, path_idx, edges_info):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, e in succ_raw[idx]:
                if j in path_idx:
                    continue
                edges_info.append(e)
                path_idx.append(j)
                dfs(j, path_idx, edges_info)
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [])
    else:
        leaves = []
        for s in range(n):
            beam = [(0.0, [s], [])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info in beam:
                    idx = path_idx[-1]
                    extended = False
                    for j, e in succ_raw[idx]:
                        if j in path_idx:
                            continue
                        extended = True
                        nxt.append((sc + e["score"], path_idx + [j], edges_info + [e]))
                    if not extended:
                        leaves.append((path_idx, edges_info))
                if not nxt:
                    break
                nxt.sort(key=lambda x: -x[0])
                beam = nxt[:beam_width]
                if len(leaves) >= beam_max_leaves:
                    break
            if len(leaves) >= beam_max_leaves:
                break
        seen = set()
        for path_idx, edges_info in leaves:
            key = tuple(path_idx)
            if key in seen:
                continue
            seen.add(key)
            all_paths.append((path_idx, edges_info))
        super_report["enumeration"]["n_beam_leaves"] = len(all_paths)

    # 把 succ_raw 轉成供 segmented 重算用的形式
    return (
        all_paths,
        rejected_edges,
        n_legal_edges,
        nodes,
        super_report,
        stats,
        succ_raw,
    )


def _score_paths_on_nodes_m4(nodes: list, tracks: list) -> tuple[list, int, dict]:
    """分段成長用：在剩餘超節點子圖上重算 A/C/M。"""
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}
    succ_raw, _, n_legal, m4_meta = _build_succ_m4(nodes)
    n = len(nodes)
    use_beam = n_legal > tp.FULL_ENUM_EDGE_CAP
    all_paths = []
    if not use_beam:
        def dfs(idx, path_idx, edges_info):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, e in succ_raw[idx]:
                if j in path_idx:
                    continue
                edges_info.append(e)
                path_idx.append(j)
                dfs(j, path_idx, edges_info)
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [])
    else:
        leaves = []
        for s in range(n):
            beam = [(0.0, [s], [])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info in beam:
                    idx = path_idx[-1]
                    extended = False
                    for j, e in succ_raw[idx]:
                        if j in path_idx:
                            continue
                        extended = True
                        nxt.append((sc + e["score"], path_idx + [j], edges_info + [e]))
                    if not extended:
                        leaves.append((path_idx, edges_info))
                if not nxt:
                    break
                nxt.sort(key=lambda x: -x[0])
                beam = nxt[: tp.DEFAULT_BEAM_WIDTH]
                if len(leaves) >= tp.DEFAULT_BEAM_MAX_LEAVES:
                    break
            if len(leaves) >= tp.DEFAULT_BEAM_MAX_LEAVES:
                break
        seen = set()
        for path_idx, edges_info in leaves:
            key = tuple(path_idx)
            if key in seen:
                continue
            seen.add(key)
            all_paths.append((path_idx, edges_info))

    scored = []
    for path_idx, edges_info in all_paths:
        score = float(sum(e["score"] for e in edges_info))
        scored.append(
            {
                "tids": tp.expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": [],
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    maximal = tp.attach_softmax(tp.maximal_paths(scored))
    return maximal, n_legal, {"n_legal_edges": n_legal, "mode": "beam" if use_beam else "full", "m4": m4_meta}


# ============================================================
# M5：A + T（自量尺 + 自時鐘；時間順序 only）
# ============================================================

def _estimate_frame_interval(tracks: list) -> float:
    """由 foots 連續時間差中位數估 1 幀間隔；不足則 1/3s（FPS≈3 觀測）。"""
    gaps: list[float] = []
    for t in tracks:
        ts = sorted(float(f[0]) for f in (t.foots or []) if f)
        for a, b in zip(ts, ts[1:]):
            d = float(b - a)
            if d > 1e-9:
                gaps.append(d)
    if len(gaps) >= 3:
        return float(np.median(np.asarray(gaps, dtype=np.float64)))
    return 1.0 / 3.0


def _topo_shortest_hops(cam_u: str, cam_v: str) -> int | None:
    """鏡頭鄰接圖 BFS 最短站數（邊數）。同鏡=0。"""
    if cam_u == cam_v:
        return 0
    adj: dict[str, set[str]] = {}
    for a, b in tp.ADJACENT:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    from collections import deque

    q = deque([(cam_u, 0)])
    seen = {cam_u}
    while q:
        c, d = q.popleft()
        for nb in adj.get(c, ()):
            if nb in seen:
                continue
            if nb == cam_v:
                return d + 1
            seen.add(nb)
            q.append((nb, d + 1))
    return None


def _hop_count_for_R(u: tp.Track, v: tp.Track, dt: float, h_dist) -> int:
    """
    R 分母用 hop_count：拓撲最短站數；重疊/同鏡交接 = 1。
    不可達 → 異常停下。
    """
    key = tuple(sorted((u.cam, v.cam)))
    if u.cam == v.cam:
        return 1
    if key in tp.OVERLAP_PAIRS:
        return 1
    if tp.is_handoff_edge(u, v, dt, h_dist):
        return 1
    hops = _topo_shortest_hops(u.cam, v.cam)
    if hops is None:
        raise RuntimeError(
            f"M5 異常：時間序邊拓撲不可達 {u.tid}({u.cam})→{v.tid}({v.cam})，"
            f"無法定義 hop_count"
        )
    return max(int(hops), 1)


def _sn_duration(sn: tp.SuperNode) -> float:
    """超節點聯集時長 = t_end − t_start。"""
    return float(max(0.0, sn.t_end - sn.t_start))


def m5_T(
    sa: tp.SuperNode,
    sb: tp.SuperNode,
    dt: float,
    hop_count: int,
    frame_interval: float,
) -> dict:
    """
    T̄ 處理退化後算 R、T_score = −max(0, ln R)。
    兩端皆退化 → T_score=0 並標記。
    """
    Tu_raw = _sn_duration(sa)
    Tv_raw = _sn_duration(sb)
    deg_u = Tu_raw < float(frame_interval)
    deg_v = Tv_raw < float(frame_interval)
    Tu, Tv = Tu_raw, Tv_raw
    both_deg = False
    if deg_u and deg_v:
        both_deg = True
        return {
            "T_u_raw": Tu_raw,
            "T_v_raw": Tv_raw,
            "T_u": Tu_raw,
            "T_v": Tv_raw,
            "T_bar": 0.0,
            "deg_u": True,
            "deg_v": True,
            "both_degenerate": True,
            "hop_count": int(hop_count),
            "R": None,
            "T_score": 0.0,
            "frame_interval": float(frame_interval),
        }
    if deg_u:
        Tu = Tv
    if deg_v:
        Tv = Tu
    T_bar = 0.5 * (Tu + Tv)
    if T_bar <= 0.0:
        raise RuntimeError(
            f"M5 異常：T̄≤0 於 {sa.label}→{sb.label} "
            f"(Tu_raw={Tu_raw}, Tv_raw={Tv_raw}, fi={frame_interval})"
        )
    if hop_count <= 0:
        raise RuntimeError(
            f"M5 異常：hop_count={hop_count} ≤0 於 {sa.label}→{sb.label}"
        )
    R = float(dt) / (float(hop_count) * T_bar)
    if R <= 0.0:
        # dt=0 → lnR→−∞ → max(0,·)=0
        T_score = 0.0
    else:
        T_score = float(-max(0.0, math.log(R)))
    return {
        "T_u_raw": Tu_raw,
        "T_v_raw": Tv_raw,
        "T_u": Tu,
        "T_v": Tv,
        "T_bar": T_bar,
        "deg_u": deg_u,
        "deg_v": deg_v,
        "both_degenerate": both_deg,
        "hop_count": int(hop_count),
        "R": R,
        "T_score": T_score,
        "frame_interval": float(frame_interval),
    }


def _best_member_edge_m5(sa: tp.SuperNode, sb: tp.SuperNode):
    """時間順序合法成員對中取 A 最大者（無 hop / DT_MAX 門檻）。"""
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            key = tuple(sorted((u.cam, v.cam)))
            tol = tp.OVERLAP_PAIRS.get(key, tp.TOL)
            h_ok, h_dist = tp.same_object_h(u, v)
            if dt_raw < -tol:
                if not (h_ok or tp.corridor_prefers(u, v)):
                    rejects.append((u.tid, v.tid, "時間順序"))
                    continue
            a = m4_A(u, v)
            emb = tp.emb_sim(u, v)
            hop_r = _hop_count_for_R(u, v, dt, h_dist)
            cand = (u, v, dt, hop_r, emb, h_dist, a)
            if best is None:
                best = cand
            else:
                ba = best[6]
                if a == float("inf") and ba != float("inf"):
                    best = cand
                elif math.isfinite(a) and math.isfinite(ba) and (
                    a > ba or (a == ba and emb > best[4])
                ):
                    best = cand
                elif math.isfinite(a) and ba == float("-inf"):
                    best = cand
    return best, rejects


def _build_succ_m5(nodes: list, frame_interval: float) -> tuple[list, list, int, dict]:
    """建時間序邊 → A + T；score = A + T_score。"""
    n = len(nodes)
    succ = [[] for _ in range(n)]
    rejected = []
    n_legal = 0
    n_deg = 0
    for i, j in itertools.permutations(range(n), 2):
        best, rejects = _best_member_edge_m5(nodes[i], nodes[j])
        if best is None:
            for r in rejects[:2]:
                rejected.append(r)
            continue
        u, v, dt, hop_r, emb, h_dist, A = best
        tinfo = m5_T(nodes[i], nodes[j], dt, hop_r, frame_interval)
        if tinfo["both_degenerate"]:
            n_deg += 1
        T_score = float(tinfo["T_score"])
        if A == float("-inf"):
            score = float("-inf")
        elif A == float("inf"):
            score = float("inf")
        else:
            score = float(A + T_score)

        e = {
            "from": u.tid,
            "to": v.tid,
            "from_super": nodes[i].label,
            "to_super": nodes[j].label,
            "from_members": nodes[i].tids,
            "to_members": nodes[j].tids,
            "dt": float(dt),
            "dt_z": float(dt),
            "hop": int(hop_r),
            "hop_count": int(hop_r),
            "emb": float(emb),
            "h_dist": None if h_dist is None else float(h_dist),
            "handoff": bool(tp.is_handoff_edge(u, v, dt, h_dist)),
            "score": score,
            "A": float(A),
            "T_score": T_score,
            "R": tinfo["R"],
            "T_u": tinfo["T_u"],
            "T_v": tinfo["T_v"],
            "T_u_raw": tinfo["T_u_raw"],
            "T_v_raw": tinfo["T_v_raw"],
            "T_bar": tinfo["T_bar"],
            "deg_u": tinfo["deg_u"],
            "deg_v": tinfo["deg_v"],
            "both_degenerate": tinfo["both_degenerate"],
            "frame_interval": tinfo["frame_interval"],
            "w_u": float(u.meta.get("w_intra", 0.0)),
            "w_v": float(v.meta.get("w_intra", 0.0)),
            "d_uv": _cos_dist(np.asarray(u.emb), np.asarray(v.emb)),
            "z_emb": 0.0,
            "z_time": 0.0,
            "hop_pen": 0.0,
            "scoring": "m5_A_plus_T",
            "dt_source": "super_union",
            "hist_emb": None,
        }
        for key in ("score", "A"):
            val = e[key]
            if val == float("inf"):
                e[key] = 1e300
                e[f"{key}_inf"] = True
            elif val == float("-inf"):
                e[key] = -1e300
                e[f"{key}_inf"] = True
        succ[i].append((j, e))
        n_legal += 1

    meta = {
        "scoring": "m5",
        "frame_interval": float(frame_interval),
        "n_both_degenerate": int(n_deg),
        "constants": [],
        "n_legal_edges": n_legal,
        "note": "建邊僅時間順序；A 同 M4；T=−max(0,ln R)",
    }
    return succ, rejected, n_legal, meta


def enumerate_paths_m5(
    tracks: list,
    merge_dir: Path,
    *,
    use_supernode: bool = True,
    beam_width: int = tp.DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = tp.DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
):
    attach_crop_embs(tracks, merge_dir)
    coexist_median = median_edge_emb(tracks)
    frame_interval = _estimate_frame_interval(tracks)
    if use_supernode:
        nodes, super_report = tp.build_supernodes(
            tracks, overlap_emb_min=coexist_median
        )
    else:
        nodes, super_report = tp._build_nodes(tracks, False)
    super_report["minimal_coexistence_emb_median"] = coexist_median
    super_report["coexistence_overlap_emb_min"] = coexist_median
    super_report["m5_frame_interval"] = frame_interval

    succ_raw, rejected_edges, n_legal_edges, m5_meta = _build_succ_m5(
        nodes, frame_interval
    )
    stats = {
        "emb": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "dt": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "sim": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "hop_pen": 0.0,
        "scoring": "m5",
        "node_score": False,
        "dt_max": None,
        "m5": m5_meta,
        "constants": [],
    }
    n = len(nodes)
    use_beam = (not force_full) and (n_legal_edges > tp.FULL_ENUM_EDGE_CAP)
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_max": None,
        "note": (
            f"M5 合法邊={n_legal_edges} > {tp.FULL_ENUM_EDGE_CAP}：beam"
            if use_beam
            else "M5 全量 DFS（含前綴）"
        ),
    }
    super_report["score_stats"] = stats
    super_report["z_stats"] = stats["emb"]

    all_paths = []
    if not use_beam:
        def dfs(idx, path_idx, edges_info):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, e in succ_raw[idx]:
                if j in path_idx:
                    continue
                edges_info.append(e)
                path_idx.append(j)
                dfs(j, path_idx, edges_info)
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [])
    else:
        leaves = []
        for s in range(n):
            beam = [(0.0, [s], [])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info in beam:
                    idx = path_idx[-1]
                    extended = False
                    for j, e in succ_raw[idx]:
                        if j in path_idx:
                            continue
                        extended = True
                        nxt.append((sc + e["score"], path_idx + [j], edges_info + [e]))
                    if not extended:
                        leaves.append((path_idx, edges_info))
                if not nxt:
                    break
                nxt.sort(key=lambda x: -x[0])
                beam = nxt[:beam_width]
                if len(leaves) >= beam_max_leaves:
                    break
            if len(leaves) >= beam_max_leaves:
                break
        seen = set()
        for path_idx, edges_info in leaves:
            key = tuple(path_idx)
            if key in seen:
                continue
            seen.add(key)
            all_paths.append((path_idx, edges_info))
        super_report["enumeration"]["n_beam_leaves"] = len(all_paths)

    return (
        all_paths,
        rejected_edges,
        n_legal_edges,
        nodes,
        super_report,
        stats,
        succ_raw,
    )


def _score_paths_on_nodes_m5(
    nodes: list, tracks: list, frame_interval: float
) -> tuple[list, int, dict]:
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}
    succ_raw, _, n_legal, m5_meta = _build_succ_m5(nodes, frame_interval)
    n = len(nodes)
    use_beam = n_legal > tp.FULL_ENUM_EDGE_CAP
    all_paths = []
    if not use_beam:
        def dfs(idx, path_idx, edges_info):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, e in succ_raw[idx]:
                if j in path_idx:
                    continue
                edges_info.append(e)
                path_idx.append(j)
                dfs(j, path_idx, edges_info)
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [])
    else:
        leaves = []
        for s in range(n):
            beam = [(0.0, [s], [])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info in beam:
                    idx = path_idx[-1]
                    extended = False
                    for j, e in succ_raw[idx]:
                        if j in path_idx:
                            continue
                        extended = True
                        nxt.append((sc + e["score"], path_idx + [j], edges_info + [e]))
                    if not extended:
                        leaves.append((path_idx, edges_info))
                if not nxt:
                    break
                nxt.sort(key=lambda x: -x[0])
                beam = nxt[: tp.DEFAULT_BEAM_WIDTH]
                if len(leaves) >= tp.DEFAULT_BEAM_MAX_LEAVES:
                    break
            if len(leaves) >= tp.DEFAULT_BEAM_MAX_LEAVES:
                break
        seen = set()
        for path_idx, edges_info in leaves:
            key = tuple(path_idx)
            if key in seen:
                continue
            seen.add(key)
            all_paths.append((path_idx, edges_info))

    scored = []
    for path_idx, edges_info in all_paths:
        score = float(sum(e["score"] for e in edges_info))
        scored.append(
            {
                "tids": tp.expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": [],
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    maximal = tp.attach_softmax(tp.maximal_paths(scored))
    return maximal, n_legal, {
        "n_legal_edges": n_legal,
        "mode": "beam" if use_beam else "full",
        "m5": m5_meta,
    }


# ============================================================
# M6：A + C + S（時間順序 only；hop 不計分）
# ============================================================

def _best_member_edge_m6(sa: tp.SuperNode, sb: tp.SuperNode):
    """時間順序合法成員對中取 A 最大者；hop 僅記錄不篩選。"""
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            key = tuple(sorted((u.cam, v.cam)))
            tol = tp.OVERLAP_PAIRS.get(key, tp.TOL)
            h_ok, h_dist = tp.same_object_h(u, v)
            if dt_raw < -tol:
                if not (h_ok or tp.corridor_prefers(u, v)):
                    rejects.append((u.tid, v.tid, "時間順序"))
                    continue
            a = m4_A(u, v)
            emb = tp.emb_sim(u, v)
            hop = tp.hop_count(u.cam, v.cam)
            if hop is None:
                hop = _topo_shortest_hops(u.cam, v.cam)
            cand = (u, v, dt, hop, emb, h_dist, a)
            if best is None:
                best = cand
            else:
                ba = best[6]
                if a == float("inf") and ba != float("inf"):
                    best = cand
                elif math.isfinite(a) and math.isfinite(ba) and (
                    a > ba or (a == ba and emb > best[4])
                ):
                    best = cand
                elif math.isfinite(a) and ba == float("-inf"):
                    best = cand
    return best, rejects


def _build_succ_m6(nodes: list) -> tuple[list, list, int, dict]:
    """
    時間序建邊 → A → 雙向 C → S=ln(1−ΣP_earlier)；score=A+C+S。
    Σ>=1 → degenerate，該邊不採用。
    """
    n = len(nodes)
    raw = {}
    rejected = []
    for i, j in itertools.permutations(range(n), 2):
        best, rejects = _best_member_edge_m6(nodes[i], nodes[j])
        if best is None:
            for r in rejects[:2]:
                rejected.append(r)
            continue
        raw[(i, j)] = best

    succ_idx = [[] for _ in range(n)]
    pred_idx = [[] for _ in range(n)]
    for (i, j) in raw:
        succ_idx[i].append(j)
        pred_idx[j].append(i)

    P_fwd = {}
    for i in range(n):
        js = succ_idx[i]
        if not js:
            continue
        As = [raw[(i, j)][6] for j in js]
        ps = _softmax_probs(As)
        for j, p in zip(js, ps):
            P_fwd[(i, j)] = p

    P_bwd = {}
    for j in range(n):
        is_ = pred_idx[j]
        if not is_:
            continue
        As = [raw[(i, j)][6] for i in is_]
        ps = _softmax_probs(As)
        for i, p in zip(is_, ps):
            P_bwd[(i, j)] = p

    def _ln_NP(N, p):
        if N <= 0 or p <= 0.0:
            return float("-inf")
        return float(math.log(N * p))

    succ = [[] for _ in range(n)]
    n_legal = 0
    n_degenerate = 0
    degenerate_edges = []

    for (i, j), (u, v, dt, hop, emb, h_dist, A) in raw.items():
        N_u = len(succ_idx[i])
        N_v_pred = len(pred_idx[j])
        p_fwd = P_fwd.get((i, j), 0.0)
        p_bwd = P_bwd.get((i, j), 0.0)
        C_fwd = _ln_NP(N_u, p_fwd)
        C_bwd = _ln_NP(N_v_pred, p_bwd)
        if C_fwd == float("-inf") or C_bwd == float("-inf"):
            C = float("-inf")
        else:
            C = C_fwd + C_bwd

        # S：跳過 t_start 更早的下家
        t_v = float(nodes[j].t_start)
        sum_p = 0.0
        skipped = []
        for k in succ_idx[i]:
            if k == j:
                continue
            if float(nodes[k].t_start) < t_v:
                pk = float(P_fwd.get((i, k), 0.0))
                sum_p += pk
                skipped.append(
                    {
                        "to_super": nodes[k].label,
                        "to_sid": nodes[k].sid,
                        "t_start": float(nodes[k].t_start),
                        "P": pk,
                        "A": float(raw[(i, k)][6])
                        if math.isfinite(raw[(i, k)][6])
                        else (
                            1e300
                            if raw[(i, k)][6] == float("inf")
                            else -1e300
                        ),
                    }
                )
        skipped.sort(key=lambda x: -x["P"])

        if sum_p >= 1.0:
            n_degenerate += 1
            degenerate_edges.append(
                {
                    "from_super": nodes[i].label,
                    "to_super": nodes[j].label,
                    "sum_P_skipped": float(sum_p),
                    "n_skipped": len(skipped),
                    "reason": "ΣP_skipped>=1",
                }
            )
            continue  # 路徑不採用

        S = float(math.log(1.0 - sum_p))

        if A == float("-inf") or C == float("-inf"):
            score = float("-inf")
        elif A == float("inf"):
            score = float("inf") if C != float("-inf") else float("-inf")
        else:
            score = float(A + C + S)

        e = {
            "from": u.tid,
            "to": v.tid,
            "from_super": nodes[i].label,
            "to_super": nodes[j].label,
            "from_members": nodes[i].tids,
            "to_members": nodes[j].tids,
            "dt": float(dt),
            "dt_z": float(dt),
            "hop": hop,  # 僅記錄；不計分
            "emb": float(emb),
            "h_dist": None if h_dist is None else float(h_dist),
            "handoff": bool(tp.is_handoff_edge(u, v, dt, h_dist)),
            "score": score,
            "A": float(A),
            "C": float(C),
            "C_fwd": float(C_fwd),
            "C_bwd": float(C_bwd),
            "S": float(S),
            "sum_P_skipped": float(sum_p),
            "n_skipped": int(len(skipped)),
            "skipped": skipped,
            "N_u": int(N_u),
            "N_v_pred": int(N_v_pred),
            "P_fwd": float(p_fwd),
            "P_bwd": float(p_bwd),
            "w_u": float(u.meta.get("w_intra", 0.0)),
            "w_v": float(v.meta.get("w_intra", 0.0)),
            "d_uv": _cos_dist(np.asarray(u.emb), np.asarray(v.emb)),
            "z_emb": 0.0,
            "z_time": 0.0,
            "hop_pen": 0.0,
            "scoring": "m6_A_C_plus_S",
            "dt_source": "super_union",
            "hist_emb": None,
            "degenerate": False,
        }
        for key in ("score", "A", "C", "C_fwd", "C_bwd", "S"):
            val = e[key]
            if val == float("inf"):
                e[key] = 1e300
                e[f"{key}_inf"] = True
            elif val == float("-inf"):
                e[key] = -1e300
                e[f"{key}_inf"] = True

        succ[i].append((j, e))
        n_legal += 1

    meta = {
        "scoring": "m6",
        "constants": [],
        "n_legal_edges": n_legal,
        "n_degenerate": n_degenerate,
        "degenerate_sample": degenerate_edges[:20],
        "note": "建邊僅時間順序；A+C 同 M4；S=ln(1−ΣP_earlier)；hop 不計分",
    }
    return succ, rejected, n_legal, meta


def _enumerate_from_succ_m6(nodes, succ_raw, n_legal_edges, *, beam_width, beam_max_leaves, force_full):
    n = len(nodes)
    use_beam = (not force_full) and (n_legal_edges > tp.FULL_ENUM_EDGE_CAP)
    all_paths = []
    if not use_beam:
        def dfs(idx, path_idx, edges_info):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, e in succ_raw[idx]:
                if j in path_idx:
                    continue
                edges_info.append(e)
                path_idx.append(j)
                dfs(j, path_idx, edges_info)
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [])
    else:
        leaves = []
        for s in range(n):
            beam = [(0.0, [s], [])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info in beam:
                    idx = path_idx[-1]
                    extended = False
                    for j, e in succ_raw[idx]:
                        if j in path_idx:
                            continue
                        extended = True
                        nxt.append((sc + e["score"], path_idx + [j], edges_info + [e]))
                    if not extended:
                        leaves.append((path_idx, edges_info))
                if not nxt:
                    break
                nxt.sort(key=lambda x: -x[0])
                beam = nxt[:beam_width]
                if len(leaves) >= beam_max_leaves:
                    break
            if len(leaves) >= beam_max_leaves:
                break
        seen = set()
        for path_idx, edges_info in leaves:
            key = tuple(path_idx)
            if key in seen:
                continue
            seen.add(key)
            all_paths.append((path_idx, edges_info))
    return all_paths, use_beam


def enumerate_paths_m6(
    tracks: list,
    merge_dir: Path,
    *,
    use_supernode: bool = True,
    beam_width: int = tp.DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = tp.DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
):
    attach_crop_embs(tracks, merge_dir)
    coexist_median = median_edge_emb(tracks)
    if use_supernode:
        nodes, super_report = tp.build_supernodes(
            tracks, overlap_emb_min=coexist_median
        )
    else:
        nodes, super_report = tp._build_nodes(tracks, False)
    super_report["minimal_coexistence_emb_median"] = coexist_median
    super_report["coexistence_overlap_emb_min"] = coexist_median

    succ_raw, rejected_edges, n_legal_edges, m6_meta = _build_succ_m6(nodes)
    stats = {
        "emb": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "dt": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "sim": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "hop_pen": 0.0,
        "scoring": "m6",
        "node_score": False,
        "dt_max": None,
        "m6": m6_meta,
        "constants": [],
    }
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal_edges,
        beam_width=beam_width,
        beam_max_leaves=beam_max_leaves,
        force_full=force_full,
    )
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_max": None,
        "n_degenerate": m6_meta.get("n_degenerate"),
        "note": (
            f"M6 合法邊={n_legal_edges} > {tp.FULL_ENUM_EDGE_CAP}：beam"
            if use_beam
            else "M6 全量 DFS（含前綴）"
        ),
    }
    if use_beam:
        super_report["enumeration"]["n_beam_leaves"] = len(all_paths)
    super_report["score_stats"] = stats
    super_report["z_stats"] = stats["emb"]
    return (
        all_paths,
        rejected_edges,
        n_legal_edges,
        nodes,
        super_report,
        stats,
        succ_raw,
    )


def _score_paths_on_nodes_m6(nodes: list, tracks: list) -> tuple[list, int, dict]:
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}
    succ_raw, _, n_legal, m6_meta = _build_succ_m6(nodes)
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal,
        beam_width=tp.DEFAULT_BEAM_WIDTH,
        beam_max_leaves=tp.DEFAULT_BEAM_MAX_LEAVES,
        force_full=False,
    )
    scored = []
    for path_idx, edges_info in all_paths:
        score = float(sum(e["score"] for e in edges_info))
        scored.append(
            {
                "tids": tp.expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": [],
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
                "min_A": _edges_min_A(edges_info),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    maximal = tp.attach_softmax(tp.maximal_paths(scored))
    for p in maximal:
        if "min_A" not in p or p["min_A"] is None:
            p["min_A"] = _edges_min_A(p.get("edges") or [])
    return maximal, n_legal, {
        "n_legal_edges": n_legal,
        "mode": "beam" if use_beam else "full",
        "m6": m6_meta,
    }


def _edges_min_A(edges: list) -> float | None:
    vals = []
    for e in edges or []:
        a = e.get("A")
        if a is None:
            continue
        try:
            af = float(a)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(af) or abs(af) >= 1e299:
            continue
        vals.append(af)
    if not vals:
        return None
    return float(min(vals))


def _hyp_min_A(hyp: dict) -> float | None:
    vals = []
    for seg in hyp.get("segments") or []:
        for e in seg.get("edges") or []:
            a = e.get("A")
            if a is None:
                continue
            try:
                af = float(a)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(af) or abs(af) >= 1e299:
                continue
            vals.append(af)
    if not vals:
        for e in hyp.get("edges") or []:
            a = e.get("A")
            if a is None:
                continue
            try:
                af = float(a)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(af) or abs(af) >= 1e299:
                continue
            vals.append(af)
    if not vals:
        return None
    return float(min(vals))


# ============================================================
# M7：logit(emb) + C + S（其餘同 M6；C/S 公式不動）
# ============================================================

def emb_logit(emb: float) -> float:
    """ln(emb/(1−emb))；emb≤0 → −∞；emb≥1 → +∞。"""
    e = float(emb)
    if e >= 1.0:
        return float("inf")
    if e <= 0.0:
        return float("-inf")
    return float(math.log(e / (1.0 - e)))


def _best_member_edge_m7(sa: tp.SuperNode, sb: tp.SuperNode):
    """時間順序合法成員對中取 logit 最大者；hop 僅記錄不篩選。"""
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            key = tuple(sorted((u.cam, v.cam)))
            tol = tp.OVERLAP_PAIRS.get(key, tp.TOL)
            h_ok, h_dist = tp.same_object_h(u, v)
            if dt_raw < -tol:
                if not (h_ok or tp.corridor_prefers(u, v)):
                    rejects.append((u.tid, v.tid, "時間順序"))
                    continue
            emb = tp.emb_sim(u, v)
            logit = emb_logit(emb)
            hop = tp.hop_count(u.cam, v.cam)
            if hop is None:
                hop = _topo_shortest_hops(u.cam, v.cam)
            # 附帶 A 僅供診斷，不參與選邊
            a_diag = m4_A(u, v)
            cand = (u, v, dt, hop, emb, h_dist, logit, a_diag)
            if best is None:
                best = cand
            else:
                bl = best[6]
                if logit == float("inf") and bl != float("inf"):
                    best = cand
                elif math.isfinite(logit) and math.isfinite(bl) and (
                    logit > bl or (logit == bl and emb > best[4])
                ):
                    best = cand
                elif math.isfinite(logit) and bl == float("-inf"):
                    best = cand
    return best, rejects


def _build_succ_m7(nodes: list) -> tuple[list, list, int, dict]:
    """
    時間序建邊 → logit → 雙向 C → S=ln(1−ΣP_earlier)；score=logit+C+S。
    C、S 公式同 M6（softmax 對 logit）；Σ>=1 → degenerate。
    """
    n = len(nodes)
    raw = {}
    rejected = []
    for i, j in itertools.permutations(range(n), 2):
        best, rejects = _best_member_edge_m7(nodes[i], nodes[j])
        if best is None:
            for r in rejects[:2]:
                rejected.append(r)
            continue
        raw[(i, j)] = best

    succ_idx = [[] for _ in range(n)]
    pred_idx = [[] for _ in range(n)]
    for (i, j) in raw:
        succ_idx[i].append(j)
        pred_idx[j].append(i)

    P_fwd = {}
    for i in range(n):
        js = succ_idx[i]
        if not js:
            continue
        Ls = [raw[(i, j)][6] for j in js]
        ps = _softmax_probs(Ls)
        for j, p in zip(js, ps):
            P_fwd[(i, j)] = p

    P_bwd = {}
    for j in range(n):
        is_ = pred_idx[j]
        if not is_:
            continue
        Ls = [raw[(i, j)][6] for i in is_]
        ps = _softmax_probs(Ls)
        for i, p in zip(is_, ps):
            P_bwd[(i, j)] = p

    def _ln_NP(N, p):
        if N <= 0 or p <= 0.0:
            return float("-inf")
        return float(math.log(N * p))

    succ = [[] for _ in range(n)]
    n_legal = 0
    n_degenerate = 0
    degenerate_edges = []

    for (i, j), (u, v, dt, hop, emb, h_dist, logit, a_diag) in raw.items():
        N_u = len(succ_idx[i])
        N_v_pred = len(pred_idx[j])
        p_fwd = P_fwd.get((i, j), 0.0)
        p_bwd = P_bwd.get((i, j), 0.0)
        C_fwd = _ln_NP(N_u, p_fwd)
        C_bwd = _ln_NP(N_v_pred, p_bwd)
        if C_fwd == float("-inf") or C_bwd == float("-inf"):
            C = float("-inf")
        else:
            C = C_fwd + C_bwd

        t_v = float(nodes[j].t_start)
        sum_p = 0.0
        skipped = []
        for k in succ_idx[i]:
            if k == j:
                continue
            if float(nodes[k].t_start) < t_v:
                pk = float(P_fwd.get((i, k), 0.0))
                sum_p += pk
                skipped.append(
                    {
                        "to_super": nodes[k].label,
                        "to_sid": nodes[k].sid,
                        "t_start": float(nodes[k].t_start),
                        "P": pk,
                        "logit": float(raw[(i, k)][6])
                        if math.isfinite(raw[(i, k)][6])
                        else (
                            1e300
                            if raw[(i, k)][6] == float("inf")
                            else -1e300
                        ),
                    }
                )
        skipped.sort(key=lambda x: -x["P"])

        if sum_p >= 1.0:
            n_degenerate += 1
            degenerate_edges.append(
                {
                    "from_super": nodes[i].label,
                    "to_super": nodes[j].label,
                    "sum_P_skipped": float(sum_p),
                    "n_skipped": len(skipped),
                    "reason": "ΣP_skipped>=1",
                }
            )
            continue

        S = float(math.log(1.0 - sum_p))

        if logit == float("-inf") or C == float("-inf"):
            score = float("-inf")
        elif logit == float("inf"):
            score = float("inf") if C != float("-inf") else float("-inf")
        else:
            score = float(logit + C + S)

        e = {
            "from": u.tid,
            "to": v.tid,
            "from_super": nodes[i].label,
            "to_super": nodes[j].label,
            "from_members": nodes[i].tids,
            "to_members": nodes[j].tids,
            "dt": float(dt),
            "dt_z": float(dt),
            "hop": hop,
            "emb": float(emb),
            "h_dist": None if h_dist is None else float(h_dist),
            "handoff": bool(tp.is_handoff_edge(u, v, dt, h_dist)),
            "score": score,
            "logit": float(logit),
            "A": float(a_diag),  # 診斷對照；不計入 score
            "C": float(C),
            "C_fwd": float(C_fwd),
            "C_bwd": float(C_bwd),
            "S": float(S),
            "sum_P_skipped": float(sum_p),
            "n_skipped": int(len(skipped)),
            "skipped": skipped,
            "N_u": int(N_u),
            "N_v_pred": int(N_v_pred),
            "P_fwd": float(p_fwd),
            "P_bwd": float(p_bwd),
            "w_u": float(u.meta.get("w_intra", 0.0)),
            "w_v": float(v.meta.get("w_intra", 0.0)),
            "d_uv": _cos_dist(np.asarray(u.emb), np.asarray(v.emb)),
            "z_emb": 0.0,
            "z_time": 0.0,
            "hop_pen": 0.0,
            "scoring": "m7_logit_C_plus_S",
            "dt_source": "super_union",
            "hist_emb": None,
            "degenerate": False,
        }
        for key in ("score", "logit", "A", "C", "C_fwd", "C_bwd", "S"):
            val = e[key]
            if val == float("inf"):
                e[key] = 1e300
                e[f"{key}_inf"] = True
            elif val == float("-inf"):
                e[key] = -1e300
                e[f"{key}_inf"] = True

        succ[i].append((j, e))
        n_legal += 1

    meta = {
        "scoring": "m7",
        "constants": [],
        "n_legal_edges": n_legal,
        "n_degenerate": n_degenerate,
        "degenerate_sample": degenerate_edges[:20],
        "note": (
            "建邊僅時間順序；logit=ln(emb/(1−emb))+C+S；"
            "C/S 公式同 M6；hop 不計分"
        ),
    }
    return succ, rejected, n_legal, meta


def enumerate_paths_m7(
    tracks: list,
    merge_dir: Path,
    *,
    use_supernode: bool = True,
    beam_width: int = tp.DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = tp.DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
):
    attach_crop_embs(tracks, merge_dir)
    coexist_median = median_edge_emb(tracks)
    if use_supernode:
        nodes, super_report = tp.build_supernodes(
            tracks, overlap_emb_min=coexist_median
        )
    else:
        nodes, super_report = tp._build_nodes(tracks, False)
    super_report["minimal_coexistence_emb_median"] = coexist_median
    super_report["coexistence_overlap_emb_min"] = coexist_median

    succ_raw, rejected_edges, n_legal_edges, m7_meta = _build_succ_m7(nodes)
    stats = {
        "emb": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "dt": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "sim": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "hop_pen": 0.0,
        "scoring": "m7",
        "node_score": False,
        "dt_max": None,
        "m7": m7_meta,
        "constants": [],
    }
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal_edges,
        beam_width=beam_width,
        beam_max_leaves=beam_max_leaves,
        force_full=force_full,
    )
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_max": None,
        "n_degenerate": m7_meta.get("n_degenerate"),
        "note": (
            f"M7 合法邊={n_legal_edges} > {tp.FULL_ENUM_EDGE_CAP}：beam"
            if use_beam
            else "M7 全量 DFS（含前綴）"
        ),
    }
    if use_beam:
        super_report["enumeration"]["n_beam_leaves"] = len(all_paths)
    super_report["score_stats"] = stats
    super_report["z_stats"] = stats["emb"]
    return (
        all_paths,
        rejected_edges,
        n_legal_edges,
        nodes,
        super_report,
        stats,
        succ_raw,
    )


def _edges_min_logit(edges: list) -> float | None:
    vals = []
    for e in edges or []:
        a = e.get("logit")
        if a is None:
            continue
        try:
            af = float(a)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(af) or abs(af) >= 1e299:
            continue
        vals.append(af)
    if not vals:
        return None
    return float(min(vals))


def _hyp_min_logit(hyp: dict) -> float | None:
    vals = []
    for seg in hyp.get("segments") or []:
        for e in seg.get("edges") or []:
            a = e.get("logit")
            if a is None:
                continue
            try:
                af = float(a)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(af) or abs(af) >= 1e299:
                continue
            vals.append(af)
    if not vals:
        for e in hyp.get("edges") or []:
            a = e.get("logit")
            if a is None:
                continue
            try:
                af = float(a)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(af) or abs(af) >= 1e299:
                continue
            vals.append(af)
    if not vals:
        return None
    return float(min(vals))


def _score_paths_on_nodes_m7(nodes: list, tracks: list) -> tuple[list, int, dict]:
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}
    succ_raw, _, n_legal, m7_meta = _build_succ_m7(nodes)
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal,
        beam_width=tp.DEFAULT_BEAM_WIDTH,
        beam_max_leaves=tp.DEFAULT_BEAM_MAX_LEAVES,
        force_full=False,
    )
    scored = []
    for path_idx, edges_info in all_paths:
        score = float(sum(e["score"] for e in edges_info))
        scored.append(
            {
                "tids": tp.expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": [],
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
                "min_logit": _edges_min_logit(edges_info),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    maximal = tp.attach_softmax(tp.maximal_paths(scored))
    for p in maximal:
        if "min_logit" not in p or p["min_logit"] is None:
            p["min_logit"] = _edges_min_logit(p.get("edges") or [])
    return maximal, n_legal, {
        "n_legal_edges": n_legal,
        "mode": "beam" if use_beam else "full",
        "m7": m7_meta,
    }


# ============================================================
# M8：C + S（softmax 對裸 emb；其餘同 M6）
# ============================================================

def _best_member_edge_m8(sa: tp.SuperNode, sb: tp.SuperNode):
    """時間順序合法成員對中取 emb 最大者；hop 僅記錄不篩選。"""
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            key = tuple(sorted((u.cam, v.cam)))
            tol = tp.OVERLAP_PAIRS.get(key, tp.TOL)
            h_ok, h_dist = tp.same_object_h(u, v)
            if dt_raw < -tol:
                if not (h_ok or tp.corridor_prefers(u, v)):
                    rejects.append((u.tid, v.tid, "時間順序"))
                    continue
            emb = float(tp.emb_sim(u, v))
            hop = tp.hop_count(u.cam, v.cam)
            if hop is None:
                hop = _topo_shortest_hops(u.cam, v.cam)
            cand = (u, v, dt, hop, emb, h_dist)
            if best is None or emb > best[4]:
                best = cand
    return best, rejects


def _build_succ_m8(nodes: list) -> tuple[list, list, int, dict]:
    """
    時間序建邊 → 雙向 C（softmax 對裸 emb）→ S；score = C + S。
    C、S 公式同 M6；Σ>=1 → degenerate。
    """
    n = len(nodes)
    raw = {}
    rejected = []
    for i, j in itertools.permutations(range(n), 2):
        best, rejects = _best_member_edge_m8(nodes[i], nodes[j])
        if best is None:
            for r in rejects[:2]:
                rejected.append(r)
            continue
        raw[(i, j)] = best

    succ_idx = [[] for _ in range(n)]
    pred_idx = [[] for _ in range(n)]
    for (i, j) in raw:
        succ_idx[i].append(j)
        pred_idx[j].append(i)

    # softmax 對裸 emb
    P_fwd = {}
    for i in range(n):
        js = succ_idx[i]
        if not js:
            continue
        embs = [raw[(i, j)][4] for j in js]
        ps = _softmax_probs(embs)
        for j, p in zip(js, ps):
            P_fwd[(i, j)] = p

    P_bwd = {}
    for j in range(n):
        is_ = pred_idx[j]
        if not is_:
            continue
        embs = [raw[(i, j)][4] for i in is_]
        ps = _softmax_probs(embs)
        for i, p in zip(is_, ps):
            P_bwd[(i, j)] = p

    def _ln_NP(N, p):
        if N <= 0 or p <= 0.0:
            return float("-inf")
        return float(math.log(N * p))

    succ = [[] for _ in range(n)]
    n_legal = 0
    n_degenerate = 0
    degenerate_edges = []

    for (i, j), (u, v, dt, hop, emb, h_dist) in raw.items():
        N_u = len(succ_idx[i])
        N_v_pred = len(pred_idx[j])
        p_fwd = P_fwd.get((i, j), 0.0)
        p_bwd = P_bwd.get((i, j), 0.0)
        C_fwd = _ln_NP(N_u, p_fwd)
        C_bwd = _ln_NP(N_v_pred, p_bwd)
        if C_fwd == float("-inf") or C_bwd == float("-inf"):
            C = float("-inf")
        else:
            C = C_fwd + C_bwd

        t_v = float(nodes[j].t_start)
        sum_p = 0.0
        skipped = []
        for k in succ_idx[i]:
            if k == j:
                continue
            if float(nodes[k].t_start) < t_v:
                pk = float(P_fwd.get((i, k), 0.0))
                sum_p += pk
                skipped.append(
                    {
                        "to_super": nodes[k].label,
                        "to_sid": nodes[k].sid,
                        "t_start": float(nodes[k].t_start),
                        "P": pk,
                        "emb": float(raw[(i, k)][4]),
                    }
                )
        skipped.sort(key=lambda x: -x["P"])

        if sum_p >= 1.0:
            n_degenerate += 1
            degenerate_edges.append(
                {
                    "from_super": nodes[i].label,
                    "to_super": nodes[j].label,
                    "sum_P_skipped": float(sum_p),
                    "n_skipped": len(skipped),
                    "reason": "ΣP_skipped>=1",
                }
            )
            continue

        S = float(math.log(1.0 - sum_p))

        if C == float("-inf"):
            score = float("-inf")
        else:
            score = float(C + S)

        e = {
            "from": u.tid,
            "to": v.tid,
            "from_super": nodes[i].label,
            "to_super": nodes[j].label,
            "from_members": nodes[i].tids,
            "to_members": nodes[j].tids,
            "dt": float(dt),
            "dt_z": float(dt),
            "hop": hop,
            "emb": float(emb),
            "h_dist": None if h_dist is None else float(h_dist),
            "handoff": bool(tp.is_handoff_edge(u, v, dt, h_dist)),
            "score": score,
            "C": float(C),
            "C_fwd": float(C_fwd),
            "C_bwd": float(C_bwd),
            "S": float(S),
            "sum_P_skipped": float(sum_p),
            "n_skipped": int(len(skipped)),
            "skipped": skipped,
            "N_u": int(N_u),
            "N_v_pred": int(N_v_pred),
            "P_fwd": float(p_fwd),
            "P_bwd": float(p_bwd),
            "z_emb": 0.0,
            "z_time": 0.0,
            "hop_pen": 0.0,
            "scoring": "m8_C_plus_S",
            "dt_source": "super_union",
            "hist_emb": None,
            "degenerate": False,
        }
        for key in ("score", "C", "C_fwd", "C_bwd", "S"):
            val = e[key]
            if val == float("inf"):
                e[key] = 1e300
                e[f"{key}_inf"] = True
            elif val == float("-inf"):
                e[key] = -1e300
                e[f"{key}_inf"] = True

        succ[i].append((j, e))
        n_legal += 1

    meta = {
        "scoring": "m8",
        "constants": [],
        "n_legal_edges": n_legal,
        "n_degenerate": n_degenerate,
        "degenerate_sample": degenerate_edges[:20],
        "note": "建邊僅時間順序；score=C+S；softmax 對裸 emb；hop 不計分",
    }
    return succ, rejected, n_legal, meta


def enumerate_paths_m8(
    tracks: list,
    merge_dir: Path,
    *,
    use_supernode: bool = True,
    beam_width: int = tp.DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = tp.DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
):
    attach_crop_embs(tracks, merge_dir)
    coexist_median = median_edge_emb(tracks)
    if use_supernode:
        nodes, super_report = tp.build_supernodes(
            tracks, overlap_emb_min=coexist_median
        )
    else:
        nodes, super_report = tp._build_nodes(tracks, False)
    super_report["minimal_coexistence_emb_median"] = coexist_median
    super_report["coexistence_overlap_emb_min"] = coexist_median

    succ_raw, rejected_edges, n_legal_edges, m8_meta = _build_succ_m8(nodes)
    stats = {
        "emb": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "dt": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "sim": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "hop_pen": 0.0,
        "scoring": "m8",
        "node_score": False,
        "dt_max": None,
        "m8": m8_meta,
        "constants": [],
    }
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal_edges,
        beam_width=beam_width,
        beam_max_leaves=beam_max_leaves,
        force_full=force_full,
    )
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_max": None,
        "n_degenerate": m8_meta.get("n_degenerate"),
        "note": (
            f"M8 合法邊={n_legal_edges} > {tp.FULL_ENUM_EDGE_CAP}：beam"
            if use_beam
            else "M8 全量 DFS（含前綴）"
        ),
    }
    if use_beam:
        super_report["enumeration"]["n_beam_leaves"] = len(all_paths)
    super_report["score_stats"] = stats
    super_report["z_stats"] = stats["emb"]
    return (
        all_paths,
        rejected_edges,
        n_legal_edges,
        nodes,
        super_report,
        stats,
        succ_raw,
    )


def _score_paths_on_nodes_m8(nodes: list, tracks: list) -> tuple[list, int, dict]:
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}
    succ_raw, _, n_legal, m8_meta = _build_succ_m8(nodes)
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal,
        beam_width=tp.DEFAULT_BEAM_WIDTH,
        beam_max_leaves=tp.DEFAULT_BEAM_MAX_LEAVES,
        force_full=False,
    )
    scored = []
    for path_idx, edges_info in all_paths:
        score = float(sum(e["score"] for e in edges_info))
        scored.append(
            {
                "tids": tp.expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": [],
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    maximal = tp.attach_softmax(tp.maximal_paths(scored))
    return maximal, n_legal, {
        "n_legal_edges": n_legal,
        "mode": "beam" if use_beam else "full",
        "m8": m8_meta,
    }


# ============================================================
# M9：LLR + C + S（softmax 對 LLR；不乘 shrink_w）
# ============================================================

def _default_m9_calib_path() -> Path:
    return tp.OUTPUT_ROOT / "v1.0" / "calibration_gt0507.pkl"


def _load_m9_calib(path: Path | str | None = None) -> tuple[dict, Path]:
    p = Path(path) if path else _default_m9_calib_path()
    p = p.resolve()
    if not p.is_file():
        raise RuntimeError(f"M9 異常：找不到校準檔 {p}")
    calib = tp.load_calibration(p)
    if "emb_same" not in calib or "emb_diff" not in calib:
        raise RuntimeError(f"M9 異常：校準檔缺 emb_same/emb_diff：{p}")
    return calib, p


def emb_llr_raw(calib: dict, emb: float) -> float:
    """LLR = ln(f_same/f_diff)；不乘 shrink_w。"""
    return float(tp.llr(calib["emb_same"], calib["emb_diff"], float(emb)))


def _best_member_edge_m9(sa: tp.SuperNode, sb: tp.SuperNode, calib: dict):
    """時間順序合法成員對中取 LLR 最大者；hop 僅記錄。"""
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            key = tuple(sorted((u.cam, v.cam)))
            tol = tp.OVERLAP_PAIRS.get(key, tp.TOL)
            h_ok, h_dist = tp.same_object_h(u, v)
            if dt_raw < -tol:
                if not (h_ok or tp.corridor_prefers(u, v)):
                    rejects.append((u.tid, v.tid, "時間順序"))
                    continue
            emb = float(tp.emb_sim(u, v))
            llr = emb_llr_raw(calib, emb)
            hop = tp.hop_count(u.cam, v.cam)
            if hop is None:
                hop = _topo_shortest_hops(u.cam, v.cam)
            cand = (u, v, dt, hop, emb, h_dist, llr)
            if best is None:
                best = cand
            else:
                bl = best[6]
                if math.isfinite(llr) and math.isfinite(bl) and (
                    llr > bl or (llr == bl and emb > best[4])
                ):
                    best = cand
                elif math.isfinite(llr) and not math.isfinite(bl) and bl == float("-inf"):
                    best = cand
                elif llr == float("inf") and bl != float("inf"):
                    best = cand
    return best, rejects


def _build_succ_m9(nodes: list, calib: dict) -> tuple[list, list, int, dict]:
    """
    時間序建邊 → LLR → 雙向 C（softmax 對 LLR）→ S；score=LLR+C+S。
    不乘 shrink_w。Σ>=1 → degenerate。
    """
    n = len(nodes)
    raw = {}
    rejected = []
    for i, j in itertools.permutations(range(n), 2):
        best, rejects = _best_member_edge_m9(nodes[i], nodes[j], calib)
        if best is None:
            for r in rejects[:2]:
                rejected.append(r)
            continue
        raw[(i, j)] = best

    succ_idx = [[] for _ in range(n)]
    pred_idx = [[] for _ in range(n)]
    for (i, j) in raw:
        succ_idx[i].append(j)
        pred_idx[j].append(i)

    P_fwd = {}
    for i in range(n):
        js = succ_idx[i]
        if not js:
            continue
        Ls = [raw[(i, j)][6] for j in js]
        ps = _softmax_probs(Ls)
        for j, p in zip(js, ps):
            P_fwd[(i, j)] = p

    P_bwd = {}
    for j in range(n):
        is_ = pred_idx[j]
        if not is_:
            continue
        Ls = [raw[(i, j)][6] for i in is_]
        ps = _softmax_probs(Ls)
        for i, p in zip(is_, ps):
            P_bwd[(i, j)] = p

    def _ln_NP(N, p):
        if N <= 0 or p <= 0.0:
            return float("-inf")
        return float(math.log(N * p))

    succ = [[] for _ in range(n)]
    n_legal = 0
    n_degenerate = 0
    degenerate_edges = []

    for (i, j), (u, v, dt, hop, emb, h_dist, llr) in raw.items():
        N_u = len(succ_idx[i])
        N_v_pred = len(pred_idx[j])
        p_fwd = P_fwd.get((i, j), 0.0)
        p_bwd = P_bwd.get((i, j), 0.0)
        C_fwd = _ln_NP(N_u, p_fwd)
        C_bwd = _ln_NP(N_v_pred, p_bwd)
        if C_fwd == float("-inf") or C_bwd == float("-inf"):
            C = float("-inf")
        else:
            C = C_fwd + C_bwd

        t_v = float(nodes[j].t_start)
        sum_p = 0.0
        skipped = []
        for k in succ_idx[i]:
            if k == j:
                continue
            if float(nodes[k].t_start) < t_v:
                pk = float(P_fwd.get((i, k), 0.0))
                sum_p += pk
                skipped.append(
                    {
                        "to_super": nodes[k].label,
                        "to_sid": nodes[k].sid,
                        "t_start": float(nodes[k].t_start),
                        "P": pk,
                        "LLR": float(raw[(i, k)][6])
                        if math.isfinite(raw[(i, k)][6])
                        else (
                            1e300
                            if raw[(i, k)][6] == float("inf")
                            else -1e300
                        ),
                        "emb": float(raw[(i, k)][4]),
                    }
                )
        skipped.sort(key=lambda x: -x["P"])

        if sum_p >= 1.0:
            n_degenerate += 1
            degenerate_edges.append(
                {
                    "from_super": nodes[i].label,
                    "to_super": nodes[j].label,
                    "sum_P_skipped": float(sum_p),
                    "n_skipped": len(skipped),
                    "reason": "ΣP_skipped>=1",
                }
            )
            continue

        S = float(math.log(1.0 - sum_p))

        if llr == float("-inf") or C == float("-inf"):
            score = float("-inf")
        elif llr == float("inf"):
            score = float("inf") if C != float("-inf") else float("-inf")
        else:
            score = float(llr + C + S)

        e = {
            "from": u.tid,
            "to": v.tid,
            "from_super": nodes[i].label,
            "to_super": nodes[j].label,
            "from_members": nodes[i].tids,
            "to_members": nodes[j].tids,
            "dt": float(dt),
            "dt_z": float(dt),
            "hop": hop,
            "emb": float(emb),
            "h_dist": None if h_dist is None else float(h_dist),
            "handoff": bool(tp.is_handoff_edge(u, v, dt, h_dist)),
            "score": score,
            "LLR": float(llr),
            "C": float(C),
            "C_fwd": float(C_fwd),
            "C_bwd": float(C_bwd),
            "S": float(S),
            "sum_P_skipped": float(sum_p),
            "n_skipped": int(len(skipped)),
            "skipped": skipped,
            "N_u": int(N_u),
            "N_v_pred": int(N_v_pred),
            "P_fwd": float(p_fwd),
            "P_bwd": float(p_bwd),
            "z_emb": 0.0,
            "z_time": 0.0,
            "hop_pen": 0.0,
            "scoring": "m9_LLR_C_plus_S",
            "dt_source": "super_union",
            "hist_emb": None,
            "degenerate": False,
            "shrink_applied": False,
        }
        for key in ("score", "LLR", "C", "C_fwd", "C_bwd", "S"):
            val = e[key]
            if val == float("inf"):
                e[key] = 1e300
                e[f"{key}_inf"] = True
            elif val == float("-inf"):
                e[key] = -1e300
                e[f"{key}_inf"] = True

        succ[i].append((j, e))
        n_legal += 1

    emb_same = calib["emb_same"]
    emb_diff = calib["emb_diff"]
    meta = {
        "scoring": "m9",
        "constants": [],
        "n_legal_edges": n_legal,
        "n_degenerate": n_degenerate,
        "degenerate_sample": degenerate_edges[:20],
        "note": (
            "建邊僅時間順序；LLR+C+S；softmax 對 LLR；"
            "不乘 shrink_w；hop 不計分"
        ),
        "calib_emb_same": {
            "mu": float(emb_same["mu"]),
            "sigma": float(emb_same["sigma"]),
            "n": emb_same.get("n"),
            "shrink_w": emb_same.get("shrink_w"),
        },
        "calib_emb_diff": {
            "mu": float(emb_diff["mu"]),
            "sigma": float(emb_diff["sigma"]),
            "n": emb_diff.get("n"),
            "shrink_w": emb_diff.get("shrink_w"),
        },
        "note_no_shrink": (
            "LLR 不乘 shrink_w：單尺度／單證據下 w 為共同倍率，不影響排序。"
        ),
    }
    return succ, rejected, n_legal, meta


def enumerate_paths_m9(
    tracks: list,
    merge_dir: Path,
    *,
    calib: dict | None = None,
    calib_path: Path | str | None = None,
    use_supernode: bool = True,
    beam_width: int = tp.DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = tp.DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
):
    if calib is None:
        calib, calib_path = _load_m9_calib(calib_path)
    else:
        calib_path = Path(calib_path) if calib_path else _default_m9_calib_path()

    attach_crop_embs(tracks, merge_dir)
    coexist_median = median_edge_emb(tracks)
    if use_supernode:
        nodes, super_report = tp.build_supernodes(
            tracks, overlap_emb_min=coexist_median
        )
    else:
        nodes, super_report = tp._build_nodes(tracks, False)
    super_report["minimal_coexistence_emb_median"] = coexist_median
    super_report["coexistence_overlap_emb_min"] = coexist_median

    succ_raw, rejected_edges, n_legal_edges, m9_meta = _build_succ_m9(nodes, calib)
    m9_meta["calibration_path"] = str(Path(calib_path).resolve())
    stats = {
        "emb": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "dt": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "sim": {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0},
        "hop_pen": 0.0,
        "scoring": "m9",
        "node_score": False,
        "dt_max": None,
        "m9": m9_meta,
        "m9_calib": calib,
        "constants": [],
    }
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal_edges,
        beam_width=beam_width,
        beam_max_leaves=beam_max_leaves,
        force_full=force_full,
    )
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_max": None,
        "n_degenerate": m9_meta.get("n_degenerate"),
        "note": (
            f"M9 合法邊={n_legal_edges} > {tp.FULL_ENUM_EDGE_CAP}：beam"
            if use_beam
            else "M9 全量 DFS（含前綴）"
        ),
    }
    if use_beam:
        super_report["enumeration"]["n_beam_leaves"] = len(all_paths)
    super_report["score_stats"] = stats
    super_report["z_stats"] = stats["emb"]
    return (
        all_paths,
        rejected_edges,
        n_legal_edges,
        nodes,
        super_report,
        stats,
        succ_raw,
    )


def _score_paths_on_nodes_m9(
    nodes: list, tracks: list, calib: dict
) -> tuple[list, int, dict]:
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}
    succ_raw, _, n_legal, m9_meta = _build_succ_m9(nodes, calib)
    all_paths, use_beam = _enumerate_from_succ_m6(
        nodes,
        succ_raw,
        n_legal,
        beam_width=tp.DEFAULT_BEAM_WIDTH,
        beam_max_leaves=tp.DEFAULT_BEAM_MAX_LEAVES,
        force_full=False,
    )
    scored = []
    for path_idx, edges_info in all_paths:
        score = float(sum(e["score"] for e in edges_info))
        scored.append(
            {
                "tids": tp.expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": [],
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    maximal = tp.attach_softmax(tp.maximal_paths(scored))
    return maximal, n_legal, {
        "n_legal_edges": n_legal,
        "mode": "beam" if use_beam else "full",
        "m9": m9_meta,
    }


def _z_stats_1d(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 1.0, "n": 0, "median": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if std < 1e-12:
        std = 1.0
    return {
        "mean": mean,
        "std": std,
        "n": int(arr.size),
        "median": float(np.median(arr)),
    }


def _emb_z_stats(edge_embs: list[float]) -> dict:
    return _z_stats_1d(edge_embs)


def build_score_stats(
    edge_metas: list[dict], tracks: list, nodes: list, cfg: RunConfig
) -> dict:
    emb_s = _z_stats_1d([m["emb"] for m in edge_metas])
    dt_s = _z_stats_1d([m["dt_z"] for m in edge_metas])
    # 候選 = load_tracks 後全部 track 的 query sim（相對化，不查校準）
    sim_s = _z_stats_1d([float(t.sim) for t in tracks])
    return {
        "emb": emb_s,
        "dt": dt_s,
        "sim": sim_s,
        "hop_pen": float(cfg.hop_pen),
        "hop_pen_note": (
            f"hop2 罰 {cfg.hop_pen}（標稱為 1 個標準差單位；敏感度見 compare）"
        ),
        "scoring": cfg.scoring,
        "node_score": bool(cfg.node_score),
        "dt_max": cfg.dt_max,
    }


def _node_z(sn: tp.SuperNode, stats: dict, cfg: RunConfig) -> dict:
    if cfg.scoring not in ("m2",) or not cfg.node_score:
        return {
            "super": sn.label,
            "members": sn.tids,
            "sim": float(sn.sim),
            "score": 0.0,
            "z_sim": 0.0,
            "enabled": False,
        }
    mu, sd = stats["sim"]["mean"], stats["sim"]["std"]
    z = (float(sn.sim) - mu) / sd
    return {
        "super": sn.label,
        "members": sn.tids,
        "sim": float(sn.sim),
        "score": float(z),
        "z_sim": float(z),
        "enabled": True,
    }


def _edge_rec(
    nodes,
    idx,
    j,
    u,
    v,
    dt,
    hop,
    emb,
    h_dist,
    handoff,
    dt_z,
    stats: dict,
    cfg: RunConfig,
) -> dict:
    if cfg.scoring == "m1":
        mu, sd = stats["emb"]["mean"], stats["emb"]["std"]
        z_emb = (float(emb) - mu) / sd
        score = float(z_emb)
        return {
            "from": u.tid,
            "to": v.tid,
            "from_super": nodes[idx].label,
            "to_super": nodes[j].label,
            "from_members": nodes[idx].tids,
            "to_members": nodes[j].tids,
            "dt": float(dt),
            "dt_z": float(dt_z),
            "hop": hop,
            "emb": float(emb),
            "h_dist": None if h_dist is None else float(h_dist),
            "handoff": bool(handoff),
            "score": score,
            "z_emb": float(z_emb),
            "z_time": 0.0,
            "hop_pen": 0.0,
            "scoring": "m1_emb_zscore",
            "dt_source": "super_union",
            "hist_emb": None,
        }

    # M2 / M3：完全相同公式；僅 scoring 標籤不同
    z_emb = (float(emb) - stats["emb"]["mean"]) / stats["emb"]["std"]
    z_time = -(float(dt_z) - stats["dt"]["mean"]) / stats["dt"]["std"]
    hop_pen = 0.0 if int(hop) <= 1 else float(cfg.hop_pen)
    score = float(z_emb + z_time + hop_pen)
    scoring_tag = "m3_professor_case" if cfg.scoring == "m3" else "m2_relative_multifactor"
    return {
        "from": u.tid,
        "to": v.tid,
        "from_super": nodes[idx].label,
        "to_super": nodes[j].label,
        "from_members": nodes[idx].tids,
        "to_members": nodes[j].tids,
        "dt": float(dt),
        "dt_z": float(dt_z),
        "hop": hop,
        "emb": float(emb),
        "h_dist": None if h_dist is None else float(h_dist),
        "handoff": bool(handoff),
        "score": score,
        "z_emb": float(z_emb),
        "z_time": float(z_time),
        "hop_pen": float(hop_pen),
        "scoring": scoring_tag,
        "dt_source": "super_union",
        "hist_emb": None,
    }


def path_score_from_parts(
    nodes_on_path: list, edges_info: list, stats: dict, cfg: RunConfig
) -> tuple[float, list]:
    node_evs = [_node_z(sn, stats, cfg) for sn in nodes_on_path]
    total = float(sum(n["score"] for n in node_evs) + sum(e["score"] for e in edges_info))
    return total, node_evs


def path_score_minimal(edges_info: list) -> tuple[float, list]:
    """M1 相容：僅邊分。"""
    return float(sum(e["score"] for e in edges_info)), []


def enumerate_paths_cfg(
    tracks: list,
    cfg: RunConfig,
    *,
    use_supernode: bool = True,
    beam_width: int = tp.DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = tp.DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
):
    coexist_median = median_edge_emb(tracks)
    if use_supernode:
        nodes, super_report = tp.build_supernodes(
            tracks, overlap_emb_min=coexist_median
        )
    else:
        nodes, super_report = tp._build_nodes(tracks, False)

    super_report["minimal_coexistence_emb_median"] = coexist_median
    super_report["coexistence_overlap_emb_min"] = coexist_median
    super_report["coexistence_note"] = (
        f"名單制相對標準：emb > median(track-level legal edge emb)={coexist_median:.6f}"
    )

    succ, rejected_edges, n_legal_edges, edge_metas = _build_succ_minimal(
        nodes, dt_max=cfg.dt_max
    )
    stats = build_score_stats(edge_metas, tracks, nodes, cfg)
    # M1 相容欄位
    z_stats = stats["emb"]

    n = len(nodes)
    use_beam = (not force_full) and (n_legal_edges > tp.FULL_ENUM_EDGE_CAP)
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_max": cfg.dt_max,
        "edge_emb_mean": stats["emb"]["mean"],
        "edge_emb_std": stats["emb"]["std"],
        "edge_dt_mean": stats["dt"]["mean"],
        "edge_dt_std": stats["dt"]["std"],
        "note": (
            f"合法邊={n_legal_edges} > {tp.FULL_ENUM_EDGE_CAP}：beam 近似 Softmax／Top-k"
            if use_beam
            else "全量 DFS（含前綴）"
        ),
    }
    super_report["z_stats"] = z_stats
    super_report["score_stats"] = stats

    def make_e(idx, j, u, v, dt, hop, emb, h_dist, handoff, dt_z):
        return _edge_rec(
            nodes, idx, j, u, v, dt, hop, emb, h_dist, handoff, dt_z, stats, cfg
        )

    def node0_score(i):
        return float(_node_z(nodes[i], stats, cfg)["score"])

    all_paths = []

    if not use_beam:
        def dfs(idx, path_idx, edges_info):
            all_paths.append((list(path_idx), list(edges_info)))
            for item in succ[idx]:
                j, u, v, dt, hop, emb, h_dist, handoff, dt_z = item
                if j in path_idx:
                    continue
                e = make_e(idx, j, u, v, dt, hop, emb, h_dist, handoff, dt_z)
                edges_info.append(e)
                path_idx.append(j)
                dfs(j, path_idx, edges_info)
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [])
    else:
        leaves = []
        for s in range(n):
            beam = [(node0_score(s), [s], [])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info in beam:
                    idx = path_idx[-1]
                    extended = False
                    for item in succ[idx]:
                        j, u, v, dt, hop, emb, h_dist, handoff, dt_z = item
                        if j in path_idx:
                            continue
                        e = make_e(
                            idx, j, u, v, dt, hop, emb, h_dist, handoff, dt_z
                        )
                        extended = True
                        nxt.append(
                            (
                                sc + e["score"] + node0_score(j),
                                path_idx + [j],
                                edges_info + [e],
                            )
                        )
                    if not extended:
                        leaves.append((path_idx, edges_info))
                if not nxt:
                    break
                nxt.sort(key=lambda x: -x[0])
                beam = nxt[:beam_width]
                if len(leaves) >= beam_max_leaves:
                    break
            if len(leaves) >= beam_max_leaves:
                break

        seen = set()
        for path_idx, edges_info in leaves:
            key = tuple(path_idx)
            if key in seen:
                continue
            seen.add(key)
            all_paths.append((path_idx, edges_info))
        super_report["enumeration"]["n_beam_leaves"] = len(all_paths)

    return (
        all_paths,
        rejected_edges,
        n_legal_edges,
        nodes,
        super_report,
        stats,
        succ,
    )


# M1 舊介面相容
def enumerate_paths_minimal(tracks: list, **kwargs):
    cfg = RunConfig(scoring="m1", node_score=False, dt_max=None)
    (
        all_paths,
        rejected,
        n_legal,
        nodes,
        super_report,
        stats,
        succ,
    ) = enumerate_paths_cfg(tracks, cfg, **kwargs)
    return all_paths, rejected, n_legal, nodes, super_report, stats["emb"], succ


def _score_paths_on_nodes_cfg(
    nodes: list,
    tracks: list,
    stats: dict,
    cfg: RunConfig,
    *,
    beam_width: int = tp.DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = tp.DEFAULT_BEAM_MAX_LEAVES,
) -> tuple[list, int, dict]:
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}

    n = len(nodes)
    succ = [[] for _ in range(n)]
    n_legal = 0
    for i, j in itertools.permutations(range(n), 2):
        if cfg.dt_max is not None and nodes[j].t_end < nodes[i].t_start - float(
            cfg.dt_max
        ):
            continue
        best, _ = _best_member_edge_minimal(nodes[i], nodes[j], dt_max=cfg.dt_max)
        if best is not None:
            u, v, dt, hop, emb, h_dist = best
            handoff = tp.is_handoff_edge(u, v, dt, h_dist)
            dt_z = 0.0 if handoff else float(dt)
            succ[i].append((j, u, v, dt, hop, emb, h_dist, handoff, dt_z))
            n_legal += 1

    use_beam = n_legal > tp.FULL_ENUM_EDGE_CAP
    enum_meta = {
        "n_legal_edges": n_legal,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "n_nodes": n,
    }
    all_paths = []

    def make_e(idx, j, u, v, dt, hop, emb, h_dist, handoff, dt_z):
        return _edge_rec(
            nodes, idx, j, u, v, dt, hop, emb, h_dist, handoff, dt_z, stats, cfg
        )

    def node0_score(i):
        return float(_node_z(nodes[i], stats, cfg)["score"])

    if not use_beam:
        def dfs(idx, path_idx, edges_info):
            all_paths.append((list(path_idx), list(edges_info)))
            for item in succ[idx]:
                j, u, v, dt, hop, emb, h_dist, handoff, dt_z = item
                if j in path_idx:
                    continue
                e = make_e(idx, j, u, v, dt, hop, emb, h_dist, handoff, dt_z)
                edges_info.append(e)
                path_idx.append(j)
                dfs(j, path_idx, edges_info)
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [])
    else:
        leaves = []
        for s in range(n):
            beam = [(node0_score(s), [s], [])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info in beam:
                    idx = path_idx[-1]
                    extended = False
                    for item in succ[idx]:
                        j, u, v, dt, hop, emb, h_dist, handoff, dt_z = item
                        if j in path_idx:
                            continue
                        e = make_e(
                            idx, j, u, v, dt, hop, emb, h_dist, handoff, dt_z
                        )
                        extended = True
                        nxt.append(
                            (
                                sc + e["score"] + node0_score(j),
                                path_idx + [j],
                                edges_info + [e],
                            )
                        )
                    if not extended:
                        leaves.append((path_idx, edges_info))
                if not nxt:
                    break
                nxt.sort(key=lambda x: -x[0])
                beam = nxt[:beam_width]
                if len(leaves) >= beam_max_leaves:
                    break
            if len(leaves) >= beam_max_leaves:
                break
        seen = set()
        for path_idx, edges_info in leaves:
            key = tuple(path_idx)
            if key in seen:
                continue
            seen.add(key)
            all_paths.append((path_idx, edges_info))
        enum_meta["n_beam_leaves"] = len(all_paths)

    scored = []
    for path_idx, edges_info in all_paths:
        sn_path = [nodes[i] for i in path_idx]
        score, node_evs = path_score_from_parts(sn_path, edges_info, stats, cfg)
        scored.append(
            {
                "tids": tp.expand_path_tids(nodes, path_idx),
                "super_labels": [nodes[i].label for i in path_idx],
                "super_ids": [nodes[i].sid for i in path_idx],
                "score": score,
                "edges": edges_info,
                "node_evidence": node_evs,
                "t_start": float(min(nodes[i].t_start for i in path_idx)),
                "t_end": float(max(nodes[i].t_end for i in path_idx)),
            }
        )
    scored.sort(key=lambda p: -p["score"])
    maximal = tp.attach_softmax(tp.maximal_paths(scored))
    return maximal, n_legal, enum_meta


def grow_segmented_hypothesis_cfg(
    seed_path: dict,
    all_nodes: list,
    tracks: list,
    stats: dict,
    cfg: RunConfig,
    *,
    max_segments: int = tp.DEFAULT_MAX_HYP_SEGMENTS,
    pool_cache: dict | None = None,
) -> list[dict]:
    cache = pool_cache if pool_cache is not None else {}
    seg1 = tp._path_as_segment(seed_path, 1, None)
    segments = [seg1]
    used_sids = set(seed_path.get("super_ids") or [])
    if not used_sids:
        used_tids = set(seed_path["tids"])
        used_sids = {n.sid for n in all_nodes if used_tids & set(n.tids)}
    remaining = [n for n in all_nodes if n.sid not in used_sids]
    prev_end = float(seed_path["t_end"])

    for seg_i in range(2, max_segments + 1):
        pool = [n for n in remaining if n.t_start > prev_end]
        if len(pool) < 1:
            break
        key = frozenset(n.sid for n in pool)
        if key in cache:
            maximal = cache[key]
        else:
            if cfg.scoring in ("m4", "m4b"):
                maximal, _, _ = _score_paths_on_nodes_m4(pool, tracks)
            elif cfg.scoring == "m5":
                fi = float((stats.get("m5") or {}).get("frame_interval") or (1.0 / 3.0))
                maximal, _, _ = _score_paths_on_nodes_m5(pool, tracks, fi)
            elif cfg.scoring == "m6":
                maximal, _, _ = _score_paths_on_nodes_m6(pool, tracks)
            elif cfg.scoring == "m7":
                maximal, _, _ = _score_paths_on_nodes_m7(pool, tracks)
            elif cfg.scoring == "m8":
                maximal, _, _ = _score_paths_on_nodes_m8(pool, tracks)
            elif cfg.scoring == "m9":
                calib = (stats.get("m9_calib") or None)
                if calib is None:
                    calib, _ = _load_m9_calib(cfg.calibration_path)
                maximal, _, _ = _score_paths_on_nodes_m9(pool, tracks, calib)
            else:
                maximal, _, _ = _score_paths_on_nodes_cfg(pool, tracks, stats, cfg)
            cache[key] = maximal
        if not maximal:
            break
        top = maximal[0]
        gap = float(top["t_start"] - prev_end)
        segments.append(tp._path_as_segment(top, seg_i, gap))
        used_now = set(top.get("super_ids") or [])
        remaining = [n for n in remaining if n.sid not in used_now]
        prev_end = float(top["t_end"])
    return segments


def _m4_edge_is_negative(e: dict) -> bool:
    """Σ < 0 → 可斷點。±inf 標記（|x|≥1e299）按符號判定。"""
    sc = e.get("score")
    if sc is None:
        return False
    sc = float(sc)
    return sc < 0.0


def _labels_to_tids(lab: str) -> list[str]:
    if lab.startswith("{") and lab.endswith("}"):
        return [x.strip() for x in lab[1:-1].split(",") if x.strip()]
    return [lab]


def _split_path_at_neg_breaks(
    path: dict, break_edge_indices: list[int], nodes_by_sid: dict
) -> list[dict]:
    """
    在指定邊序號處斷開：不收那些邊的分數；各段獨立、時間不回頭。
    break_edge_indices：0-based，對應 path['edges'][i]（連接 node i → i+1）。
    """
    labels = list(path.get("super_labels") or [])
    sids = list(path.get("super_ids") or [])
    edges = list(path.get("edges") or [])
    n_nodes = len(labels)
    if n_nodes < 2 or not edges:
        return []

    breaks = sorted({int(i) for i in break_edge_indices if 0 <= int(i) < len(edges)})
    if not breaks:
        return []

    # 節點區間 [lo, hi] inclusive
    ranges: list[tuple[int, int]] = []
    start = 0
    for bi in breaks:
        # 段結束於 node bi；下一段從 bi+1 開始
        if bi < start:
            continue
        ranges.append((start, bi))
        start = bi + 1
    if start <= n_nodes - 1:
        ranges.append((start, n_nodes - 1))

    # 過濾空段
    ranges = [(lo, hi) for lo, hi in ranges if lo <= hi]
    if len(ranges) < 2:
        return []

    segments = []
    prev_t_end = None
    for seg_i, (lo, hi) in enumerate(ranges, 1):
        seg_labels = labels[lo : hi + 1]
        seg_sids = sids[lo : hi + 1] if sids else []
        seg_edges = edges[lo:hi]  # 段內邊；不含斷點邊
        tids: list[str] = []
        for lab in seg_labels:
            for t in _labels_to_tids(lab):
                if t not in tids:
                    tids.append(t)
        if seg_sids and all(s in nodes_by_sid for s in seg_sids):
            t0 = float(min(nodes_by_sid[s].t_start for s in seg_sids))
            t1 = float(max(nodes_by_sid[s].t_end for s in seg_sids))
        else:
            t0 = float(path["t_start"])
            t1 = float(path["t_end"])
        gap = None
        if prev_t_end is not None:
            gap = float(t0 - prev_t_end)
        segments.append(
            {
                "segment": seg_i,
                "path": " -> ".join(seg_labels),
                "super_labels": seg_labels,
                "tids": tids,
                "super_ids": seg_sids,
                "score": float(sum(float(e.get("score") or 0.0) for e in seg_edges)),
                "t_start": t0,
                "t_end": t1,
                "gap_after_prev_sec": gap,
                "edges": seg_edges,
                "node_evidence": [],
            }
        )
        prev_t_end = t1
    return segments


def generate_m4b_break_hypotheses(
    path: dict, seed_rank: int, all_nodes: list
) -> list[dict]:
    """
    對一條路徑：找出 Σ<0 可斷點，產生各單斷法 +（若≥2）全斷法。
    全縫版由呼叫端另加，此處只產分段版。
    """
    edges = path.get("edges") or []
    neg_idxs = [i for i, e in enumerate(edges) if _m4_edge_is_negative(e)]
    if not neg_idxs:
        return []

    nodes_by_sid = {n.sid: n for n in all_nodes}
    # 若 path 缺 super_ids，用 label 對照
    if not path.get("super_ids"):
        lab_to_sid = {n.label: n.sid for n in all_nodes}
        path = dict(path)
        path["super_ids"] = [
            lab_to_sid.get(lab, f"UNK_{i}")
            for i, lab in enumerate(path.get("super_labels") or [])
        ]
        for sid in path["super_ids"]:
            if sid not in nodes_by_sid:
                # 無法對時戳則仍可分段，用整段時間
                pass

    hyps = []

    def _mk(breaks: list[int], source: str) -> dict | None:
        segs = _split_path_at_neg_breaks(path, breaks, nodes_by_sid)
        if len(segs) < 2:
            return None
        # 時間不回頭：段間 t_start 應遞增
        for a, b in zip(segs, segs[1:]):
            if float(b["t_start"]) < float(a["t_end"]) - 1e-9:
                # 允許重疊容許內的交接；若嚴重回頭則丟棄
                if float(b["t_start"]) + 1e-9 < float(a["t_start"]):
                    return None
        hyp = tp._hypothesis_from_segments(
            segs, source=source, seed_rank=seed_rank
        )
        hyp["break_edge_indices"] = list(breaks)
        hyp["break_edges"] = [
            {
                "edge_index": i,
                "from": edges[i].get("from"),
                "to": edges[i].get("to"),
                "from_super": edges[i].get("from_super"),
                "to_super": edges[i].get("to_super"),
                "score": edges[i].get("score"),
                "A": edges[i].get("A"),
                "C": edges[i].get("C"),
                "M": edges[i].get("M"),
            }
            for i in breaks
            if i < len(edges)
        ]
        hyp["continuous_score"] = float(path.get("score") or 0.0)
        hyp["score_gain_vs_continuous"] = float(hyp["score"]) - float(
            path.get("score") or 0.0
        )
        return hyp

    for bi in neg_idxs:
        h = _mk([bi], "m4b_single_break")
        if h:
            hyps.append(h)
    if len(neg_idxs) >= 2:
        h = _mk(neg_idxs, "m4b_all_breaks")
        if h:
            hyps.append(h)
    return hyps


def build_ranked_hypotheses_cfg(
    single_maximal: list,
    all_nodes: list,
    tracks: list,
    stats: dict,
    cfg: RunConfig,
    *,
    seed_top_k: int = tp.DEFAULT_SEGMENT_SEED_TOP_K,
    max_segments: int = tp.DEFAULT_MAX_HYP_SEGMENTS,
) -> tuple[list, dict]:
    by_tid = {t.tid: t for t in tracks}
    rejected = []
    pool = []
    seen_keys = set()
    pool_cache: dict = {}

    def _try_add(hyp: dict) -> None:
        key = tuple(tuple(seg["tids"]) for seg in hyp["segments"])
        if key in seen_keys:
            return
        bad = tp.hypothesis_internal_contradictions(hyp["tids"], by_tid)
        if bad:
            rejected.append(
                {
                    "path": hyp["path"],
                    "n_segments": hyp["n_segments"],
                    "score": hyp["score"],
                    "contradictions": bad,
                }
            )
            return
        seen_keys.add(key)
        hyp["internal_contradictions"] = []
        pool.append(hyp)

    for rank, p in enumerate(single_maximal, 1):
        hyp = tp._hypothesis_from_segments(
            [tp._path_as_segment(p, 1, None)],
            source="single_maximal",
            seed_rank=rank,
        )
        _try_add(hyp)

    seeds = single_maximal[: max(int(seed_top_k), 0)]
    n_grown = 0
    for rank, seed in enumerate(seeds, 1):
        used_sids = set(seed.get("super_ids") or [])
        if not used_sids:
            used_tids = set(seed["tids"])
            used_sids = {n.sid for n in all_nodes if used_tids & set(n.tids)}
        has_later = any(
            n.sid not in used_sids and n.t_start > float(seed["t_end"])
            for n in all_nodes
        )
        if not has_later:
            continue
        segs = grow_segmented_hypothesis_cfg(
            seed,
            all_nodes,
            tracks,
            stats,
            cfg,
            max_segments=max_segments,
            pool_cache=pool_cache,
        )
        if len(segs) < 2:
            continue
        hyp = tp._hypothesis_from_segments(
            segs, source="seed_grown", seed_rank=rank
        )
        before = len(pool)
        _try_add(hyp)
        if len(pool) > before:
            n_grown += 1

    # M4b：負分邊可斷點 → 分段假設進同池
    n_break_added = 0
    if cfg.scoring == "m4b":
        for rank, seed in enumerate(seeds, 1):
            for hyp in generate_m4b_break_hypotheses(seed, rank, all_nodes):
                before = len(pool)
                _try_add(hyp)
                if len(pool) > before:
                    n_break_added += 1

    pool.sort(key=lambda h: -h["score"])
    pool = tp.attach_softmax(pool)
    # M6：附 min-A；M7：附 min-logit（均不參與排名）
    if cfg.scoring == "m6":
        for h in pool:
            h["min_A"] = _hyp_min_A(h)
    if cfg.scoring == "m7":
        for h in pool:
            h["min_logit"] = _hyp_min_logit(h)
    meta = {
        "n_single_maximal": len(single_maximal),
        "n_hypotheses_ranked": len(pool),
        "n_segmented_added": n_grown,
        "n_m4b_break_added": n_break_added if cfg.scoring == "m4b" else 0,
        "n_rejected_contradiction": len(rejected),
        "seed_top_k": seed_top_k,
        "rejected_sample": rejected[:20],
        "note": (
            f"{cfg.scoring}：極大路徑+分段同池；"
            f"dt_max={cfg.dt_max}；node_score={cfg.node_score}；"
            + (
                "M4b 負分邊可斷點競價；"
                if cfg.scoring == "m4b"
                else ""
            )
            + (
                "附 min-A（不參與排名）；"
                if cfg.scoring == "m6"
                else ""
            )
            + (
                "附 min-logit（不參與排名）；"
                if cfg.scoring == "m7"
                else ""
            )
            + "矛盾作廢；softmax"
        ),
    }
    return pool, meta


# M1 舊介面
def grow_segmented_hypothesis_minimal(seed_path, all_nodes, z_stats, **kw):
    cfg = RunConfig(scoring="m1")
    stats = {"emb": z_stats, "dt": _z_stats_1d([]), "sim": _z_stats_1d([])}
    return grow_segmented_hypothesis_cfg(
        seed_path, all_nodes, [], stats, cfg, **kw
    )


def build_ranked_hypotheses_minimal(single_maximal, all_nodes, tracks, z_stats, **kw):
    cfg = RunConfig(scoring="m1")
    stats = {"emb": z_stats, "dt": _z_stats_1d([]), "sim": _z_stats_1d([])}
    return build_ranked_hypotheses_cfg(
        single_maximal, all_nodes, tracks, stats, cfg, **kw
    )


def run_with_config(merge_dir: Path, cfg: RunConfig):
    t0 = time.perf_counter()
    tp.SIM_MIN = float(cfg.sim_min)
    mode = tp.configure_for_input(str(merge_dir))
    tracks = tp.load_tracks(str(merge_dir))
    t_load = time.perf_counter()

    if cfg.scoring in ("m4", "m4b"):
        (
            all_paths,
            _rejected,
            n_legal_edges,
            nodes,
            super_report,
            stats,
            _succ,
        ) = enumerate_paths_m4(tracks, Path(merge_dir))
    elif cfg.scoring == "m5":
        (
            all_paths,
            _rejected,
            n_legal_edges,
            nodes,
            super_report,
            stats,
            _succ,
        ) = enumerate_paths_m5(tracks, Path(merge_dir))
    elif cfg.scoring == "m6":
        (
            all_paths,
            _rejected,
            n_legal_edges,
            nodes,
            super_report,
            stats,
            _succ,
        ) = enumerate_paths_m6(tracks, Path(merge_dir))
    elif cfg.scoring == "m7":
        (
            all_paths,
            _rejected,
            n_legal_edges,
            nodes,
            super_report,
            stats,
            _succ,
        ) = enumerate_paths_m7(tracks, Path(merge_dir))
    elif cfg.scoring == "m8":
        (
            all_paths,
            _rejected,
            n_legal_edges,
            nodes,
            super_report,
            stats,
            _succ,
        ) = enumerate_paths_m8(tracks, Path(merge_dir))
    elif cfg.scoring == "m9":
        (
            all_paths,
            _rejected,
            n_legal_edges,
            nodes,
            super_report,
            stats,
            _succ,
        ) = enumerate_paths_m9(
            tracks,
            Path(merge_dir),
            calib_path=cfg.calibration_path,
        )
    else:
        (
            all_paths,
            _rejected,
            n_legal_edges,
            nodes,
            super_report,
            stats,
            _succ,
        ) = enumerate_paths_cfg(tracks, cfg)
    t_enum = time.perf_counter()

    scored = []
    for path_idx, edges_info in all_paths:
        sn_path = [nodes[i] for i in path_idx]
        if cfg.scoring in ("m4", "m4b", "m5", "m6", "m7", "m8", "m9"):
            score = float(sum(e["score"] for e in edges_info))
            node_evs = []
            extra = {}
            if cfg.scoring == "m6":
                extra["min_A"] = _edges_min_A(edges_info)
            elif cfg.scoring == "m7":
                extra["min_logit"] = _edges_min_logit(edges_info)
        else:
            score, node_evs = path_score_from_parts(sn_path, edges_info, stats, cfg)
            extra = {}
        row = {
            "tids": tp.expand_path_tids(nodes, path_idx),
            "super_labels": [nodes[i].label for i in path_idx],
            "super_ids": [nodes[i].sid for i in path_idx],
            "score": score,
            "edges": edges_info,
            "node_evidence": node_evs,
            "t_start": float(min(nodes[i].t_start for i in path_idx)),
            "t_end": float(max(nodes[i].t_end for i in path_idx)),
        }
        row.update(extra)
        scored.append(row)
    scored.sort(key=lambda p: -p["score"])
    single_maximal = tp.maximal_paths(scored)
    single_maximal_diag = tp.attach_softmax([dict(p) for p in single_maximal])

    ranked, rank_meta = build_ranked_hypotheses_cfg(
        single_maximal, nodes, tracks, stats, cfg
    )
    t_end = time.perf_counter()

    timing = {
        "load_sec": t_load - t0,
        "enumerate_sec": t_enum - t_load,
        "rank_sec": t_end - t_enum,
        "total_sec": t_end - t0,
    }
    tag = cfg.variant_tag or (
        f"{cfg.scoring}"
        f"_node{'on' if cfg.node_score else 'off'}"
        f"_dtmax{'on' if cfg.dt_max is not None else 'off'}"
    )
    if cfg.scoring == "m3":
        rules = M3_RULES
    elif cfg.scoring == "m4b":
        rules = M4B_RULES
    elif cfg.scoring == "m4":
        rules = M4_RULES
    elif cfg.scoring == "m5":
        rules = M5_RULES
    elif cfg.scoring == "m6":
        rules = M6_RULES
    elif cfg.scoring == "m7":
        rules = M7_RULES
    elif cfg.scoring == "m8":
        rules = M8_RULES
    elif cfg.scoring == "m9":
        rules = M9_RULES
    elif cfg.scoring == "m2":
        rules = M2_RULES
    else:
        rules = MINIMAL_RULES_M1
    options = {
        "variant": tag,
        "scoring": cfg.scoring,
        "node_score": bool(cfg.node_score),
        "dt_max": cfg.dt_max,
        "hop_pen": float(cfg.hop_pen),
        "no_calibration": cfg.scoring != "m9",
        "no_EMB_EDGE_MIN": True,
        "no_hist_emb_gate": True,
        "coexistence_emb_median": super_report.get("minimal_coexistence_emb_median"),
        "score_stats": {k: v for k, v in stats.items() if k != "m9_calib"},
        "z_stats": stats["emb"],
        "enumeration": super_report.get("enumeration"),
        "ranking_meta": rank_meta,
        "rules": rules,
        "timing": timing,
        "constants": [] if cfg.scoring in ("m4", "m4b", "m5", "m6", "m7", "m8", "m9") else None,
        "m4b_note": (
            "Σ<0 可斷點；斷開版與全縫版同池競價；計分= M4"
            if cfg.scoring == "m4b"
            else None
        ),
        "segments": list((ranked[0].get("segments") if ranked else None) or []),
        "single_maximal_top1": (
            {
                "score": single_maximal_diag[0]["score"],
                "path": " -> ".join(
                    single_maximal_diag[0].get("super_labels")
                    or single_maximal_diag[0]["tids"]
                ),
                "path_probability": single_maximal_diag[0].get("path_probability"),
                "tids": single_maximal_diag[0]["tids"],
            }
            if single_maximal_diag
            else None
        ),
    }
    return {
        "mode": mode,
        "tracks": tracks,
        "scored": scored,
        "ranked": ranked,
        "n_legal_edges": n_legal_edges,
        "nodes": nodes,
        "super_report": super_report,
        "options": options,
        "timing": timing,
        "stats": stats,
        "cfg": cfg,
        "tag": tag,
    }


def run_minimal(merge_dir: Path, *, sim_min: float = 0.85):
    return run_with_config(
        merge_dir,
        RunConfig(
            scoring="m1",
            node_score=False,
            dt_max=None,
            sim_min=sim_min,
            variant_tag="minimal_M1",
        ),
    )


def _hyp_brief(p: dict, rank: int) -> dict:
    segs_brief = []
    for seg in p.get("segments") or []:
        segs_brief.append(
            {
                "segment": seg.get("segment"),
                "path": seg.get("path"),
                "tids": seg.get("tids"),
                "super_labels": seg.get("super_labels"),
                "score": seg.get("score"),
                "t_start": seg.get("t_start"),
                "t_end": seg.get("t_end"),
                "gap_after_prev_sec": seg.get("gap_after_prev_sec"),
            }
        )
    return {
        "rank": rank,
        "score": p["score"],
        "path_probability": p.get("path_probability"),
        "hypothesis_type": p.get("hypothesis_type"),
        "n_segments": p.get("n_segments"),
        "tids": p["tids"],
        "super_labels": p.get("super_labels"),
        "path": p.get("path") or " -> ".join(p.get("super_labels") or p["tids"]),
        "segments": segs_brief,
        "source": p.get("source"),
        "seed_rank": p.get("seed_rank"),
        "min_A": p.get("min_A"),
        "min_logit": p.get("min_logit"),
    }


def precision_recall_vs_gt(path_tids: list[str], gt_set: set[str]) -> dict:
    n = len(path_tids)
    hit = sum(1 for t in path_tids if t in gt_set)
    n_gt = len(gt_set)
    return {
        "n_path": n,
        "n_hit": hit,
        "n_gt": n_gt,
        "precision": (hit / n) if n else 0.0,
        "recall": (hit / n_gt) if n_gt else 0.0,
        "hit_tids": [t for t in path_tids if t in gt_set],
        "false_positive": [t for t in path_tids if t not in gt_set],
        "false_negative": sorted(gt_set - set(path_tids)),
    }


def _dataset_short(name: str) -> str:
    if "20260507" in name:
        return "0507"
    if "20260528" in name:
        return "0528"
    return name


def _save_summary(result: dict, merge_dir: Path, out_dir: Path, stem: str) -> dict:
    ranked = result["ranked"]
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "variant": result["tag"],
        "mode": result["mode"],
        "scoring": result["cfg"].scoring,
        "input_dir": str(merge_dir),
        "sim_min": float(result["cfg"].sim_min),
        "n_tracks": len(result["tracks"]),
        "n_legal_edges": result["n_legal_edges"],
        "n_paths_all": len(result["scored"]),
        "n_hypotheses_ranked": len(ranked),
        "options": {
            k: v for k, v in result["options"].items() if k != "segments"
        },
        "supernodes": {
            k: result["super_report"].get(k)
            for k in (
                "n_tracks",
                "n_supernodes",
                "n_merged_pairs",
                "multi_only",
                "minimal_coexistence_emb_median",
                "enumeration",
                "score_stats",
            )
        },
        "timing": result["timing"],
        "top1": _hyp_brief(ranked[0], 1) if ranked else None,
        "top3_hypotheses": [
            _hyp_brief(h, i) for i, h in enumerate(ranked[:3], 1)
        ],
    }
    out_json = out_dir / f"{stem}.json"
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"JSON：{out_json}")
    return summary


def cmd_run(argv=None):
    p = argparse.ArgumentParser(
        description="路徑枚舉增量入口（交付預設 M9；M0 請用 track_path.py）"
    )
    p.add_argument(
        "input_dir",
        nargs="?",
        default=str(tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507"),
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--scoring",
        choices=["m1", "m2", "m3", "m4", "m4b", "m5", "m6", "m7", "m8", "m9"],
        default="m9",
        help="計分版本（交付預設 m9）",
    )
    p.add_argument("--node-score", choices=["on", "off"], default="off")
    p.add_argument("--dtmax", choices=["on", "off"], default="off")
    p.add_argument(
        "--hop-pen",
        type=float,
        default=-1.0,
        help="M2 hop2 罰分（預設 -1.0＝1σ單位）",
    )
    p.add_argument("--tag", type=str, default="")
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="M9 校準 pkl（預設 calibration_gt0507.pkl）",
    )
    args = p.parse_args(argv)

    merge_dir = Path(args.input_dir).resolve()
    if not merge_dir.is_dir():
        raise SystemExit(f"找不到資料夾：{merge_dir}")

    out_dir = (
        args.out_dir
        or (
            tp.OUTPUT_ROOT
            / "v1.0"
            / (
                "m9_comparison" if args.scoring == "m9"
                else (
                    "m8_comparison" if args.scoring == "m8"
                    else (
                        "m7_comparison" if args.scoring == "m7"
                        else (
                            "m6_comparison" if args.scoring == "m6"
                            else (
                                "m5_comparison" if args.scoring == "m5"
                                else (
                                    "m4b_comparison" if args.scoring == "m4b"
                                    else (
                                        "m4_comparison" if args.scoring == "m4"
                                        else (
                                            "m3_comparison" if args.scoring == "m3"
                                            else (
                                                "m2_comparison"
                                                if args.scoring == "m2"
                                                else "minimal_comparison"
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = RunConfig(
        scoring=args.scoring,
        node_score=(args.node_score == "on") if args.scoring == "m2" else False,
        dt_max=(
            None if args.scoring in ("m1", "m3", "m4", "m4b", "m5", "m6", "m7", "m8", "m9")
            else (DT_MAX_M2_ON if args.dtmax == "on" else None)
        ),
        hop_pen=float(args.hop_pen),
        sim_min=float(args.sim_min),
        variant_tag=args.tag
        or (
            f"{args.scoring}"
            if args.scoring in ("m3", "m4", "m4b", "m5", "m6", "m7", "m8", "m9")
            else (
                f"{args.scoring}"
                f"_node{args.node_score}"
                f"_dtmax{args.dtmax}"
                f"_hop{args.hop_pen:g}"
            )
        ),
        calibration_path=(
            str(args.calibration.resolve()) if args.calibration else None
        ),
    )
    print(
        f"[{cfg.variant_tag}] input={merge_dir}  "
        f"scoring={cfg.scoring} node={cfg.node_score} dt_max={cfg.dt_max} "
        f"hop_pen={cfg.hop_pen}"
    )
    result = run_with_config(merge_dir, cfg)
    ranked = result["ranked"]
    short = _dataset_short(merge_dir.name)
    if ranked:
        top = ranked[0]
        print(
            f"Top-1 P={top['path_probability']:.6f}  score={top['score']:.4f}  "
            f"n_seg={top['n_segments']}  {top.get('path', '')}"
        )
    enum = result["super_report"].get("enumeration") or {}
    print(
        f"合法邊={result['n_legal_edges']}  mode={enum.get('mode')}  "
        f"耗時={result['timing']['total_sec']:.2f}s"
    )
    stem = f"{short}_{cfg.variant_tag}_top1"
    return _save_summary(result, merge_dir, out_dir, stem)


def _align_nodes(labels_a: list, labels_b: list, name_a="A", name_b="B") -> list[dict]:
    n = max(len(labels_a), len(labels_b))
    rows = []
    for i in range(n):
        a = labels_a[i] if i < len(labels_a) else "—"
        b = labels_b[i] if i < len(labels_b) else "—"
        rows.append({"i": i + 1, name_a: a, name_b: b, "diff": a != b})
    return rows


def _flatten_super_labels(top: dict | None) -> list[str]:
    if not top:
        return []
    if top.get("n_segments", 1) > 1 and top.get("segments"):
        out = []
        for seg in top["segments"]:
            labs = seg.get("super_labels") or []
            if labs:
                out.extend(labs)
            elif seg.get("path"):
                part = seg["path"].replace(" || ", " -> ")
                out.extend([x.strip() for x in part.split("->") if x.strip()])
        return out
    if top.get("super_labels"):
        return list(top["super_labels"])
    if top.get("segments"):
        out = []
        for seg in top["segments"]:
            if seg.get("super_labels"):
                out.extend(seg["super_labels"])
            elif seg.get("path"):
                part = seg["path"].replace(" || ", " -> ")
                out.extend([x.strip() for x in part.split("->") if x.strip()])
        return out
    path = top.get("path") or ""
    parts = []
    for chunk in path.split("||"):
        parts.extend([x.strip() for x in chunk.split("->") if x.strip()])
    return parts


def _top_pack(summary_or_top, gt_set: set[str], extra: dict | None = None) -> dict:
    if summary_or_top is None:
        return {}
    if "top1" in summary_or_top:
        top = summary_or_top["top1"]
        base = {
            "path": top.get("path"),
            "super_labels": _flatten_super_labels(top),
            "tids": top.get("tids") or [],
            "P": top.get("path_probability"),
            "n_segments": top.get("n_segments"),
            "score": top.get("score"),
            **precision_recall_vs_gt(top.get("tids") or [], gt_set),
            "n_legal_edges": summary_or_top.get("n_legal_edges"),
            "enum_mode": (summary_or_top.get("options") or {})
            .get("enumeration", {})
            .get("mode"),
            "timing_sec": (summary_or_top.get("timing") or {}).get("total_sec"),
        }
    else:
        top = summary_or_top
        base = {
            "path": top.get("path"),
            "super_labels": _flatten_super_labels(top),
            "tids": top.get("tids") or [],
            "P": top.get("path_probability"),
            "n_segments": top.get("n_segments"),
            "score": top.get("score"),
            **precision_recall_vs_gt(top.get("tids") or [], gt_set),
        }
    if extra:
        base.update(extra)
    return base


def _fmt_pct(x: float) -> str:
    return f"{x:.3f}"


# ---------- 舊 M1 compare（保留）----------
def cmd_compare(argv=None):
    p = argparse.ArgumentParser(description="M0 vs M1 對照實驗")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "minimal_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 委派：重跑 M1 並引用既有比較報告邏輯（簡化：呼叫 compare_m2 的 M1 欄）
    print("提示：完整多版對照請用 `compare_m2`。此處僅重跑 M0/M1。")
    rows = []
    for short, merge_name, gt_name, m0_name in [
        (
            "0507",
            "人員追蹤_20260507",
            "ground_truth_20260507.json",
            "人員追蹤_20260507_llr_top1.json",
        ),
        (
            "0528",
            "人員追蹤_20260528",
            "ground_truth_20260528.json",
            "人員追蹤_20260528_llr_top1.json",
        ),
    ]:
        merge = tp.QUERY_FILTER_OUTPUT_ROOT / merge_name
        gt_set = set(
            json.loads(
                (tp.OUTPUT_ROOT / "v1.0" / gt_name).read_text(encoding="utf-8")
            )["person_tids"]
        )
        m0 = json.loads(
            (
                tp.OUTPUT_ROOT / "path_enum_llr" / "freeze_v1.1" / m0_name
            ).read_text(encoding="utf-8")
        )
        m1_sum = cmd_run(
            [
                str(merge),
                "--scoring",
                "m1",
                "--sim-min",
                str(args.sim_min),
                "--out-dir",
                str(out_dir),
                "--tag",
                "minimal_M1",
            ]
        )
        rows.append(
            {
                "short": short,
                "m0": _top_pack(m0, gt_set, {"source": "freeze_v1.1"}),
                "m1": _top_pack(m1_sum, gt_set),
            }
        )
    report = args.report or (out_dir / "comparison_minimal.md")
    lines = [
        "# M0 vs M1（簡表）",
        "",
        f"生成：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "> GT 僅用於評估。",
        "",
        "| 資料集 | 版 | prec | rec | P |",
        "|--------|----|-----:|----:|---:|",
    ]
    for row in rows:
        for k, tag in (("m0", "M0"), ("m1", "M1")):
            d = row[k]
            lines.append(
                f"| {row['short']} | {tag} | {_fmt_pct(d['precision'])} | "
                f"{_fmt_pct(d['recall'])} | {d['P']:.6f} |"
            )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"報告：{report}")
    return rows


# ---------- M2 對照實驗 ----------
def cmd_compare_m2(argv=None):
    p = argparse.ArgumentParser(description="M0/M1/M2a/M2b/M2c 對照 + hop 敏感度")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m2_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
            "m0": (
                tp.OUTPUT_ROOT
                / "path_enum_llr"
                / "freeze_v1.1"
                / "人員追蹤_20260507_llr_top1.json"
            ),
            "m1_cached": (
                tp.OUTPUT_ROOT
                / "v1.0"
                / "minimal_comparison"
                / "0507_minimal_top1.json"
            ),
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
            "m0": (
                tp.OUTPUT_ROOT
                / "path_enum_llr"
                / "freeze_v1.1"
                / "人員追蹤_20260528_llr_top1.json"
            ),
            "m1_cached": (
                tp.OUTPUT_ROOT
                / "v1.0"
                / "minimal_comparison"
                / "0528_minimal_top1.json"
            ),
        },
    ]

    # 變體定義
    variants = [
        ("M2a", dict(scoring="m2", node_score=True, dt_max=DT_MAX_M2_ON, hop_pen=-1.0)),
        ("M2b", dict(scoring="m2", node_score=False, dt_max=DT_MAX_M2_ON, hop_pen=-1.0)),
        ("M2c", dict(scoring="m2", node_score=True, dt_max=None, hop_pen=-1.0)),
        ("M1", dict(scoring="m1", node_score=False, dt_max=None, hop_pen=-1.0)),
    ]
    hop_sens = [-0.5, -1.0, -2.0]

    account = {"generated_at": datetime.now().isoformat(timespec="seconds"), "datasets": {}}

    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])
        m0 = json.loads(ds["m0"].read_text(encoding="utf-8"))
        pack = {"gt_set": sorted(gt_set), "M0": _top_pack(m0, gt_set, {"source": str(ds["m0"])})}

        # M1：優先引用既有；否則重跑
        if ds["m1_cached"].is_file():
            m1j = json.loads(ds["m1_cached"].read_text(encoding="utf-8"))
            pack["M1"] = _top_pack(m1j, gt_set, {"source": str(ds["m1_cached"])})
            print(f"[{short}] M1 引用既有 {ds['m1_cached'].name}")
        else:
            print(f"[{short}] M1 重跑")
            m1_sum = cmd_run(
                [
                    str(ds["merge"]),
                    "--scoring",
                    "m1",
                    "--sim-min",
                    str(args.sim_min),
                    "--out-dir",
                    str(out_dir),
                    "--tag",
                    "M1",
                ]
            )
            pack["M1"] = _top_pack(m1_sum, gt_set)

        for tag, kw in variants:
            if tag == "M1":
                continue
            print(f"\n===== {short} {tag} =====")
            cfg = RunConfig(
                scoring=kw["scoring"],
                node_score=kw["node_score"],
                dt_max=kw["dt_max"],
                hop_pen=kw["hop_pen"],
                sim_min=float(args.sim_min),
                variant_tag=tag,
            )
            result = run_with_config(ds["merge"], cfg)
            summary = _save_summary(
                result, ds["merge"], out_dir, f"{short}_{tag}_top1"
            )
            pack[tag] = _top_pack(
                summary,
                gt_set,
                {
                    "node_score": cfg.node_score,
                    "dt_max": cfg.dt_max,
                    "hop_pen": cfg.hop_pen,
                    "score_stats": result["stats"],
                    "timing": result["timing"],
                },
            )

        # hop_pen 敏感度（固定 M2a 設定：node on, dtmax on）
        pack["hop_sensitivity"] = []
        for hp in hop_sens:
            print(f"\n===== {short} hop_pen={hp} (M2a 設定) =====")
            cfg = RunConfig(
                scoring="m2",
                node_score=True,
                dt_max=DT_MAX_M2_ON,
                hop_pen=float(hp),
                sim_min=float(args.sim_min),
                variant_tag=f"M2a_hop{hp:g}",
            )
            result = run_with_config(ds["merge"], cfg)
            summary = _save_summary(
                result, ds["merge"], out_dir, f"{short}_M2a_hop{hp:g}_top1"
            )
            top = summary["top1"]
            pack["hop_sensitivity"].append(
                {
                    "hop_pen": hp,
                    "path": top.get("path") if top else None,
                    "tids": top.get("tids") if top else [],
                    "P": top.get("path_probability") if top else None,
                    "score": top.get("score") if top else None,
                    **precision_recall_vs_gt(
                        (top.get("tids") if top else []) or [], gt_set
                    ),
                }
            )

        pack["align_M2a_vs_M0"] = _align_nodes(
            pack["M0"].get("super_labels") or _flatten_super_labels(m0.get("top1")),
            pack["M2a"].get("super_labels") or [],
            "M0",
            "M2a",
        )
        account["datasets"][short] = pack

    report_path = (
        args.report.resolve()
        if args.report
        else (out_dir / "comparison_m2.md")
    )
    root_report = tp.REPO_ROOT.parent / "comparison_m2.md"
    text = _render_m2_report(account)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    account_path = out_dir / "comparison_m2_account.json"
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n報告：{report_path}")
    print(f"副本：{root_report}")
    print(f"帳本：{account_path}")
    return account


def _render_m2_report(account: dict) -> str:
    lines = []
    lines.append("# M2 全相對化多因子 vs M0 / M1 對照實驗")
    lines.append("")
    lines.append(f"生成時間：{account.get('generated_at')}")
    lines.append("")
    lines.append(
        "> **GT 僅用於評估**（precision / recall / P）；"
        "M2 推論零校準分布、零手調交換率（hop_pen 以 1σ 單位註記並做敏感度）。"
    )
    lines.append("")
    lines.append(
        "變體：`M2a` node on + dtmax on；`M2b` node off + dtmax on（query-free）；"
        "`M2c` node on + dtmax off；`M1` 僅邊 emb z-score；`M0` = freeze_v1.1。"
    )
    lines.append("")

    # --- 1 總表 ---
    lines.append("## 1. Top-1 prec / rec / P 總表")
    lines.append("")
    lines.append(
        "| 資料集 | 版 | precision | recall | P(softmax) | n_seg | n_path | n_hit |"
    )
    lines.append(
        "|--------|----|-----------:|-------:|-----------:|------:|-------:|------:|"
    )
    order = ["M0", "M1", "M2a", "M2b", "M2c"]
    for short, pack in account["datasets"].items():
        for key in order:
            d = pack[key]
            lines.append(
                f"| {short} | {key} | {_fmt_pct(d['precision'])} | "
                f"{_fmt_pct(d['recall'])} | {float(d['P']):.6f} | "
                f"{d.get('n_segments')} | {d['n_path']} | {d['n_hit']} |"
            )
    lines.append("")

    # --- 2 M2a vs M0 ---
    lines.append("## 2. M2a vs M0 逐節點對照（相對化能否追上校準？）")
    lines.append("")
    for short, pack in account["datasets"].items():
        m0, m2a = pack["M0"], pack["M2a"]
        same = m0["tids"] == m2a["tids"]
        lines.append(f"### {short}")
        lines.append("")
        lines.append(f"- M0：`{m0.get('path')}`")
        lines.append(f"- M2a：`{m2a.get('path')}`")
        lines.append(
            f"- prec/rec：M0 {_fmt_pct(m0['precision'])}/{_fmt_pct(m0['recall'])}；"
            f"M2a {_fmt_pct(m2a['precision'])}/{_fmt_pct(m2a['recall'])}"
        )
        if same:
            lines.append(
                "- **路徑 tid 集合與順序一致** → 相對化多因子在此案追上凍結校準計分。"
            )
        else:
            lines.append(
                "- **主問題答覆：相對化計分尚未追上校準計分**（路徑有差）。差異節點："
            )
            lines.append("")
            lines.append("| # | M0 | M2a | 差異 |")
            lines.append("|---|----|-----|------|")
            for a in pack["align_M2a_vs_M0"]:
                mark = "★" if a["diff"] else ""
                lines.append(
                    f"| {a['i']} | `{a['M0']}` | `{a['M2a']}` | {mark} |"
                )
            only_m0 = sorted(set(m0["tids"]) - set(m2a["tids"]))
            only_m2 = sorted(set(m2a["tids"]) - set(m0["tids"]))
            lines.append("")
            lines.append(f"- 僅 M0：{only_m0}")
            lines.append(f"- 僅 M2a：{only_m2}")
        lines.append("")

    # --- 3 M2b vs M1 錯收 ---
    lines.append("## 3. M2b（query-free）若重演 M1 錯收——逐案")
    lines.append("")
    for short, pack in account["datasets"].items():
        m1, m2b = pack["M1"], pack["M2b"]
        gt = set(pack["gt_set"])
        lines.append(f"### {short}")
        lines.append("")
        lines.append(f"- M1 FP：{m1.get('false_positive')}")
        lines.append(f"- M2b FP：{m2b.get('false_positive')}")
        lines.append(f"- M1 FN：{m1.get('false_negative')}")
        lines.append(f"- M2b FN：{m2b.get('false_negative')}")
        shared_fp = sorted(set(m1.get("false_positive") or []) & set(m2b.get("false_positive") or []))
        m1_only_fp = sorted(set(m1.get("false_positive") or []) - set(m2b.get("false_positive") or []))
        m2b_new_fp = sorted(set(m2b.get("false_positive") or []) - set(m1.get("false_positive") or []))
        lines.append(f"- 兩版共有錯收：{shared_fp}")
        lines.append(f"- 僅 M1 錯收：{m1_only_fp}")
        lines.append(f"- 僅 M2b 新錯收：{m2b_new_fp}")
        if shared_fp or (m2b.get("false_positive") and not m2b.get("node_score", True)):
            lines.append(
                "- **解讀**：M2b 關閉節點 z_sim；若仍出現與 M1 同類／相同錯收，"
                "是「節點身分分必要性」的第二次證據（相對邊多因子仍不足以取代節點分）。"
            )
        for tid in shared_fp:
            lines.append(
                f"  - 逐案 `{tid}`：M1 與 M2b 皆錯收 → 無節點分時相對邊分無法排除。"
            )
        for tid in m2b_new_fp:
            lines.append(
                f"  - 逐案 `{tid}`：M2b 新錯收（M1 無）→ 查 z_time/hop_pen/DT_MAX 交互。"
            )
        for tid in (m2b.get("false_negative") or []):
            if tid in (m1.get("false_negative") or []):
                lines.append(f"  - 漏收 `{tid}`：M1/M2b 皆漏（相對 GT）。")
            else:
                lines.append(f"  - 漏收 `{tid}`：M2b 新漏、M1 有 → 邊多因子排序改變。")
        # vs M2a：node on 是否救回
        m2a = pack["M2a"]
        rescued = sorted(
            set(m2b.get("false_positive") or []) - set(m2a.get("false_positive") or [])
        )
        if rescued:
            lines.append(
                f"- M2a（node on）相對 M2b 排除的錯收：{rescued} "
                "→ 節點 z_sim 直接貢獻。"
            )
        lines.append("")

    # --- 4 hop 敏感度 ---
    lines.append("## 4. hop_pen 敏感度（−0.5 / −1.0 / −2.0，其餘同 M2a）")
    lines.append("")
    lines.append("| 資料集 | hop_pen | prec | rec | P | Top-1 路徑是否相對 −1.0 改變 |")
    lines.append("|--------|--------:|-----:|----:|---:|------------------------------|")
    for short, pack in account["datasets"].items():
        base = None
        for row in pack["hop_sensitivity"]:
            if abs(row["hop_pen"] + 1.0) < 1e-12:
                base = tuple(row.get("tids") or [])
                break
        for row in pack["hop_sensitivity"]:
            changed = (
                "—"
                if base is None
                else ("是 ★" if tuple(row.get("tids") or []) != base else "否")
            )
            lines.append(
                f"| {short} | {row['hop_pen']} | {_fmt_pct(row['precision'])} | "
                f"{_fmt_pct(row['recall'])} | {float(row['P']):.6f} | {changed} |"
            )
    lines.append("")
    lines.append("各檔路徑：")
    lines.append("")
    for short, pack in account["datasets"].items():
        lines.append(f"### {short}")
        for row in pack["hop_sensitivity"]:
            lines.append(f"- hop_pen={row['hop_pen']}: `{row.get('path')}`")
        lines.append("")
    lines.append(
        "註：敏感度僅報告 Top-1 是否變動；**不因此改採 −0.5/−2.0**（禁止調參湊分）。"
    )
    lines.append("")

    # --- 5 規則一頁 ---
    lines.append("## 5. 極簡規則清單更新版（給教授的一頁）")
    lines.append("")
    lines.append("### M2（本輪）")
    lines.append("")
    for i, r in enumerate(M2_RULES, 1):
        lines.append(f"{i}. {r}")
    lines.append("")
    lines.append(
        "明確沒有：校準 pkl、EMB_EDGE_MIN、EMB_HIST_MIN、MIN_TRANSIT、"
        "LLR same/diff 分布、幾何進路徑分、手調交換率權重（除 hop_pen=−1σ 之標稱單位）。"
    )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "實驗約束：未調參湊分數；未改 M2 設計公式；"
        "未修改 track_path.py / query_filter / config / calibration。"
    )
    lines.append("")
    return "\n".join(lines)


def cmd_compare_m3(argv=None):
    """M3（教授原案）對照實驗：0507 / 0528 Top-1 prec/rec/P，空窗橋接帳目，亂接檢驗。"""
    p = argparse.ArgumentParser(description="M3 教授原案對照實驗")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m3_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--hop-pen", type=float, default=-1.0)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0507_top1.json",
            "m2a_cached": tp.OUTPUT_ROOT / "v1.0" / "m2_comparison" / "0507_M2a_top1.json",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0528_top1.json",
            "m2a_cached": tp.OUTPUT_ROOT / "v1.0" / "m2_comparison" / "0528_M2a_top1.json",
        },
    ]

    account = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "hop_pen": float(args.hop_pen),
        "datasets": {},
    }
    results_by_short: dict[str, dict] = {}

    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])

        # M0（v1.1）
        m0 = json.loads(ds["m0_json"].read_text(encoding="utf-8"))
        m0_top = m0.get("top1") or {}
        m0_pack = _top_pack(m0, gt_set, {"source": str(ds["m0_json"])})

        # M2a（引用既有或跳過）
        m2a_pack: dict = {}
        if ds["m2a_cached"].is_file():
            m2a_j = json.loads(ds["m2a_cached"].read_text(encoding="utf-8"))
            m2a_pack = _top_pack(m2a_j, gt_set, {"source": str(ds["m2a_cached"])})
            print(f"[{short}] M2a 引用既有 {ds['m2a_cached'].name}")
        else:
            print(f"[{short}] M2a 未找到，跳過（comparison_m3 只顯示 M2a 欄為空）")

        # M3
        print(f"\n===== {short} M3 =====")
        cfg = RunConfig(
            scoring="m3",
            node_score=False,   # M3 規格：無節點分
            dt_max=None,        # M3 規格：無 DT_MAX
            hop_pen=float(args.hop_pen),
            sim_min=float(args.sim_min),
            variant_tag="M3",
        )
        result = run_with_config(ds["merge"], cfg)
        summary = _save_summary(result, ds["merge"], out_dir, f"{short}_M3_top1")
        m3_pack = _top_pack(
            summary,
            gt_set,
            {"node_score": False, "dt_max": None, "hop_pen": float(args.hop_pen)},
        )
        results_by_short[short] = {
            "result": result,
            "summary": summary,
            "gt_set": gt_set,
            "cfg": cfg,
        }

        account["datasets"][short] = {
            "gt_set": sorted(gt_set),
            "M0": m0_pack,
            "M2a": m2a_pack,
            "M3": m3_pack,
            "m3_stats": result["stats"],
            "m3_super_report": {
                k: result["super_report"].get(k)
                for k in ("n_tracks", "n_supernodes", "n_merged_pairs",
                          "minimal_coexistence_emb_median", "enumeration", "score_stats")
            },
        }

    # ---- 拼圖：0528 M3 Top-1 ----
    _render_m3_collage(results_by_short, out_dir)

    # ---- 報告 ----
    report_path = (
        args.report.resolve() if args.report
        else (out_dir / "comparison_m3.md")
    )
    root_report = tp.REPO_ROOT.parent / "comparison_m3.md"
    text = _render_m3_report(account, results_by_short)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    account_path = out_dir / "comparison_m3_account.json"
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n報告：{report_path}")
    print(f"副本：{root_report}")
    print(f"帳本：{account_path}")
    return account


def _render_m3_collage(results_by_short: dict, out_dir: Path) -> None:
    """畫 0528 M3 Top-1 拼圖（同規格：GT 綠/非GT 紅；邊上標 z_emb/z_time/hop_pen/小計）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[m3_collage] Pillow 不可用，跳過拼圖")
        return

    short = "0528"
    if short not in results_by_short:
        return
    data = results_by_short[short]
    result = data["result"]
    gt_set = data["gt_set"]
    ranked = result["ranked"]
    if not ranked:
        print("[m3_collage] 無 Top-1，跳過")
        return

    top = ranked[0]
    merge = tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528"
    tracks = result["tracks"]
    by_tid = {t.tid: t for t in tracks}

    def _font(size: int):
        for name in (
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ):
            if Path(name).is_file():
                try:
                    return ImageFont.truetype(name, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    FONT_T = _font(17)
    FONT_XS = _font(10)

    def members_of_label(lab):
        if lab.startswith("{") and lab.endswith("}"):
            return [x.strip() for x in lab[1:-1].split(",") if x.strip()]
        return [lab]

    def thumb(path, size):
        if path is None or not Path(path).is_file():
            return Image.new("RGB", size, (230, 230, 230))
        im = Image.open(path).convert("RGB")
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (236, 236, 236))
        canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
        return canvas

    # 取出分段資訊
    segs = sorted(top.get("segments") or [], key=lambda s: int(s.get("segment") or 1))
    if not segs:
        segs = [{"segment": 1, "super_labels": top.get("super_labels") or [],
                 "tids": top.get("tids") or [], "edges": top.get("edges") or [],
                 "gap_after_prev_sec": None}]

    tw, th = 100, 130
    margin, title_h, row_gap, cell_gap = 12, 68, 20, 68
    max_cols = 8

    rows = []
    for si, seg in enumerate(segs):
        if si > 0:
            g = seg.get("gap_after_prev_sec") or 178.3
            rows.append(("gap", f"觀測空窗 {g:.1f} 秒（此期間不在任何鏡頭視野內）"))
        labels = seg.get("super_labels") or []
        edges = seg.get("edges") or []
        nodes_list = []
        for i, lab in enumerate(labels):
            mems = members_of_label(lab)
            if i == 0 and edges and edges[0].get("from_members"):
                mems = list(edges[0]["from_members"])
            elif i > 0 and i - 1 < len(edges) and edges[i - 1].get("to_members"):
                mems = list(edges[i - 1]["to_members"])
            e = edges[i] if i < len(edges) else None
            nodes_list.append((mems, e))
        for st in range(0, len(nodes_list), max_cols):
            rows.append(("nodes", nodes_list[st: st + max_cols]))

    rh, rw = [], []
    for row in rows:
        if row[0] == "gap":
            rh.append(30)
            rw.append(900)
            continue
        hmax = th + 72
        wsum = 0
        for mems, e in row[1]:
            cw = max(140, len(mems) * (tw + 8) + 12)
            wsum += cw
            if e is not None:
                wsum += cell_gap
        rh.append(hmax)
        rw.append(wsum)

    W = margin * 2 + max(rw + [800])
    H = title_h + sum(rh) + row_gap * max(0, len(rh) - 1) + margin
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    n_seg = top.get("n_segments", 1)
    pr_top = precision_recall_vs_gt(top.get("tids") or [], gt_set)
    title_lines = [
        f"M3（教授原案）0528 Top-1  n_seg={n_seg}  score={top.get('score', 0):.4f}",
        f"prec={pr_top['precision']:.4f} ({pr_top['n_hit']}/{pr_top['n_path']})  "
        f"rec={pr_top['recall']:.4f} ({pr_top['n_hit']}/{pr_top['n_gt']})  "
        f"P={top.get('path_probability', 0):.6f}",
        "邊分 = z_emb + z_time + hop_pen（z=0 為全圖均值，負分=低於平均）  GT 僅供評估與著色",
    ]
    ty = 6
    for line in title_lines:
        draw.text((margin, ty), line, fill=(10, 10, 10), font=FONT_XS)
        ty += 14
    ty += 2
    draw.rectangle([margin, ty, margin + 14, ty + 12], outline=(0, 160, 0), width=2)
    draw.text((margin + 18, ty), "GT", fill=(0, 140, 0), font=FONT_XS)
    draw.rectangle([margin + 55, ty, margin + 69, ty + 12], outline=(200, 40, 40), width=2)
    draw.text((margin + 73, ty), "非GT", fill=(200, 40, 40), font=FONT_XS)

    y = title_h
    for row in rows:
        if row[0] == "gap":
            draw.rectangle([margin, y, W - margin, y + 26], fill=(255, 245, 230), outline=(200, 120, 40))
            draw.text((margin + 6, y + 6), row[1], fill=(160, 80, 0), font=FONT_XS)
            y += 30 + row_gap
            continue
        x = margin
        hmax = th + 72
        for mems, e in row[1]:
            cw = max(140, len(mems) * (tw + 8) + 12)
            bc = (0, 160, 0) if all(m in gt_set for m in mems) else (200, 40, 40)
            draw.rectangle([x, y, x + cw - 1, y + hmax - 1], outline=bc, width=3)
            if len(mems) > 1:
                draw.text((x + 4, y + 2), "共存合併", fill=bc, font=FONT_XS)
            for i, tid in enumerate(mems):
                t = by_tid.get(tid)
                cam, tid_s = tid.rsplit("_", 1)
                try:
                    _, crops = tp._crop_paths_for_track(merge, cam, int(tid_s))
                    rep = tp._pick_rep_crop(crops)
                except Exception:
                    rep = None
                sx = x + 4 + i * (tw + 8)
                sy = y + 16
                img.paste(thumb(rep, (tw, th)), (sx, sy))
                mbc = (0, 160, 0) if tid in gt_set else (200, 40, 40)
                draw.rectangle([sx, sy, sx + tw - 1, sy + th - 1], outline=mbc, width=2)
                if t:
                    txt = f"{tid}\n{cam}\n[{t.t_start:.1f}-{t.t_end:.1f}]\nsim={t.sim:.3f}"
                else:
                    txt = tid
                ly = sy + th + 2
                for line in txt.split("\n"):
                    draw.text((sx, ly), line, fill=(20, 20, 20), font=FONT_XS)
                    ly += 11
            x += cw
            if e is not None:
                mid = y + hmax // 2
                draw.line([(x + 4, mid), (x + cell_gap - 8, mid)], fill=(40, 40, 40), width=2)
                z_emb = e.get("z_emb", 0.0)
                z_time = e.get("z_time", 0.0)
                hp = e.get("hop_pen", 0.0)
                sc = e.get("score", 0.0)
                sc_col = (0, 140, 0) if sc >= 0 else (200, 0, 0)
                draw.multiline_text(
                    (x + 2, mid - 30),
                    f"z_e={z_emb:+.2f}\nz_t={z_time:+.2f}\nhp={hp:+.1f}",
                    fill=(40, 40, 140),
                    font=FONT_XS,
                )
                draw.text((x + 2, mid + 8), f"Σ={sc:+.2f}", fill=sc_col, font=FONT_XS)
                x += cell_gap
        y += hmax + row_gap

    out_png = out_dir / "人員追蹤_20260528_m3_top1_collage.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    print(f"拼圖：{out_png}")


def _render_m3_report(account: dict, results_by_short: dict) -> str:
    lines = [
        "# M3（教授原案）對照實驗",
        "",
        f"生成時間：{account.get('generated_at')}",
        "",
        "> **GT 僅用於評估與著色**；M3 推論零校準、零手調交換率。",
        "",
        "M3 規格：共存合併超節點（物理規則）+ 時間順序 + hop≤2（無 DT_MAX、無 emb 底線）",
        "+ 邊分 = z_emb + z_time + hop_pen（hop_pen=0/0/−1；全圖自標準化）+ 無節點分。",
        "",
    ]

    # --- 1 總表 ---
    lines += [
        "## 1. Top-1 prec / rec / P 總表（M0 / M2a / M3）",
        "",
        "| 資料集 | 版 | precision | recall | P(softmax) | n_seg | n_path | n_hit |",
        "|--------|----|-----------:|-------:|-----------:|------:|-------:|------:|",
    ]
    for short, pack in account["datasets"].items():
        for key in ("M0", "M2a", "M3"):
            d = pack.get(key) or {}
            if not d:
                lines.append(f"| {short} | {key} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {short} | {key} | {_fmt_pct(d.get('precision', 0))} | "
                f"{_fmt_pct(d.get('recall', 0))} | {float(d.get('P') or 0):.6f} | "
                f"{d.get('n_segments')} | {d.get('n_path')} | {d.get('n_hit')} |"
            )
    lines.append("")

    # --- 2 主戲：0528 空窗橋接 ---
    lines += [
        "## 2. 主戲——0528 空窗檢驗（M3 Top-1 是單段還是分段？）",
        "",
    ]
    short = "0528"
    pack = account["datasets"].get(short, {})
    m3 = pack.get("M3") or {}
    n_seg = m3.get("n_segments", "?")
    lines.append(f"**M3 Top-1：n_segments = {n_seg}**")
    lines.append("")

    data = results_by_short.get(short)
    if data:
        result = data["result"]
        gt_set = data["gt_set"]
        ranked = result["ranked"]
        top = ranked[0] if ranked else None
        if top:
            segs = sorted(top.get("segments") or [], key=lambda s: int(s.get("segment") or 1))
            if int(n_seg) == 1:
                lines.append(
                    "Top-1 **單段**：路徑從前段真跡延伸，**跨越 178s 空窗區間**，"
                    "以橋接節點硬接後段。橋接邊帳目如下："
                )
                lines.append("")
                # 找跨越空窗的邊：前段結束時間 ≈ 394s，後段開始 ≈ 572s
                # 在 edges 中找 dt > 100s 的邊
                all_edges = top.get("edges") or []
                bridge_edges = [e for e in all_edges if (e.get("dt") or 0) > 100]
                if bridge_edges:
                    lines.append(
                        "| from | to | hop | dt(s) | emb | z_emb | z_time | hop_pen | Σscore | from-GT? | to-GT? |"
                    )
                    lines.append(
                        "|------|-----|----:|------:|----:|------:|-------:|--------:|-------:|:--------:|:------:|"
                    )
                    for e in bridge_edges:
                        f_gt = "✓" if e["from"] in gt_set else "✗"
                        t_gt = "✓" if e["to"] in gt_set else "✗"
                        lines.append(
                            f"| `{e['from']}` | `{e['to']}` | {e.get('hop')} | "
                            f"{e.get('dt', 0):.1f} | {e.get('emb', 0):.3f} | "
                            f"{e.get('z_emb', 0):+.3f} | {e.get('z_time', 0):+.3f} | "
                            f"{e.get('hop_pen', 0):+.1f} | {e.get('score', 0):+.3f} | "
                            f"{f_gt} | {t_gt} |"
                        )
                    lines.append("")
                    lines.append(
                        "上表各橋接邊的 z_time 極負（dt 大，在全圖均值基礎下遠高於均值），"
                        "z_emb 若也不夠正，則橋接邊為**負分**，說明正負分抑制有效但仍被其他"
                        "路徑節點正分沖銷。"
                    )
                else:
                    # 列出所有邊帳目
                    lines.append("（全部邊帳目，dt 從小到大）：")
                    lines.append("")
                    lines.append(
                        "| from | to | hop | dt(s) | z_emb | z_time | hop_pen | Σ |"
                    )
                    lines.append("|------|-----|----:|------:|------:|-------:|--------:|---:|")
                    for e in sorted(all_edges, key=lambda e: e.get("dt", 0)):
                        lines.append(
                            f"| `{e['from']}` | `{e['to']}` | {e.get('hop')} | "
                            f"{e.get('dt', 0):.1f} | {e.get('z_emb', 0):+.3f} | "
                            f"{e.get('z_time', 0):+.3f} | {e.get('hop_pen', 0):+.1f} | "
                            f"{e.get('score', 0):+.3f} |"
                        )
                    lines.append("")
            else:
                lines.append(
                    f"Top-1 **分段（{n_seg} 段）**：M3 在無 DT_MAX 條件下，路徑組裝時"
                    "找不到跨越 178s 空窗的高分邊，分段假設勝出。"
                )
                lines.append("")
                for seg in segs:
                    g = seg.get("gap_after_prev_sec")
                    seg_pr = precision_recall_vs_gt(seg.get("tids") or [], gt_set)
                    lines.append(
                        f"- **seg{seg['segment']}**：[{seg.get('t_start'):.1f}s – {seg.get('t_end'):.1f}s]  "
                        f"score={seg.get('score'):.4f}  "
                        f"prec={seg_pr['precision']:.3f} rec={seg_pr['recall']:.3f}"
                    )
                    if g is not None:
                        lines.append(f"  - 前一段結束後空窗 {g:.1f}s")
                    seg_edges = seg.get("edges") or []
                    bridge = [e for e in seg_edges if (e.get("dt") or 0) > 50]
                    if bridge:
                        for e in bridge:
                            f_gt = "GT" if e["from"] in gt_set else "nonGT"
                            t_gt = "GT" if e["to"] in gt_set else "nonGT"
                            lines.append(
                                f"  - 段內長跨邊 `{e['from']}`({f_gt})→`{e['to']}`({t_gt})  "
                                f"hop={e.get('hop')} dt={e.get('dt'):.1f}s  "
                                f"z_emb={e.get('z_emb'):+.3f} z_time={e.get('z_time'):+.3f} "
                                f"Σ={e.get('score'):+.3f}"
                            )
                lines.append("")

    # --- 3 亂接檢驗 ---
    lines += [
        "## 3. 亂接檢驗（Top-1 路徑長度、平均邊分、強制去負分邊後斷在哪）",
        "",
    ]
    if data and data["result"]["ranked"]:
        top = data["result"]["ranked"][0]
        all_edges = top.get("edges") or []
        n_path = len(top.get("tids") or [])
        n_edge = len(all_edges)
        avg_sc = (sum(e.get("score", 0) for e in all_edges) / n_edge) if n_edge else 0
        neg_edges = [(i, e) for i, e in enumerate(all_edges) if (e.get("score") or 0) < 0]
        lines += [
            f"- Top-1（0528）路徑節點數：{n_path}（含所有 tid）",
            f"- 路徑邊數：{n_edge}",
            f"- 平均邊分：{avg_sc:+.4f}",
            f"- 負分邊數：{len(neg_edges)} / {n_edge}",
            "",
        ]
        if neg_edges:
            lines.append("「強制去掉負分邊」——路徑會在以下位置斷開：")
            lines.append("")
            lines.append("| 邊序號 | from_super → to_super | hop | dt(s) | Σscore |")
            lines.append("|-------:|----------------------|----:|------:|-------:|")
            for i, e in neg_edges:
                lines.append(
                    f"| {i+1} | `{e.get('from_super')}` → `{e.get('to_super')}` | "
                    f"{e.get('hop')} | {e.get('dt', 0):.1f} | {e.get('score', 0):+.3f} |"
                )
            lines.append("")
            lines.append(
                "若強制去除上表全部負分邊，路徑分裂為以下連通段（位置以邊序號分隔）：即"
                "每個負分邊前後就是一個自然斷點。正負分能區別「有意義的延伸」vs「湊長」，"
                "但不會自動拒絕——需搭配排名機制讓去掉負分邊的路徑有機會得分更高才能勝出。"
            )
        else:
            lines.append("Top-1 路徑中無負分邊：正負分機制有效，每條邊均高於全圖均值。")
        lines.append("")

    # --- 4 拼圖說明 ---
    lines += [
        "## 4. 0528 M3 Top-1 拼圖",
        "",
        "輸出：`output/v1.0/m3_comparison/人員追蹤_20260528_m3_top1_collage.png`",
        "",
        "每格邊標：z_emb / z_time / hop_pen 及小計 Σ；正分綠、負分紅；GT 綠框 / 非GT 紅框。",
        "若有橋接（硬接）邊，紅框節點會自己說話。",
        "",
    ]

    # --- 5 規則一頁 ---
    lines += [
        "## 5. M3 規則清單（教授原案）",
        "",
    ]
    for i, r in enumerate(M3_RULES, 1):
        lines.append(f"{i}. {r}")
    lines.append("")
    lines.append(
        "明確沒有：校準 pkl、EMB_EDGE_MIN、DT_MAX、LLR same/diff 分布、"
        "幾何進路徑分、手調交換率、節點證據分。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "實驗約束：GT 僅用於評估與著色，不進入推論；"
        "未修改 track_path.py / query_filter / config / calibration；"
        "未調參湊分數。"
    )
    lines.append("")
    return "\n".join(lines)


def _m4_edge_disp(e: dict) -> dict:
    """報告用：把 ±1e300 還原成 ±inf 字串。"""
    out = dict(e)

    def _fix(key):
        v = out.get(key)
        if v is None:
            return
        if out.get(f"{key}_inf") or (isinstance(v, (int, float)) and abs(v) >= 1e299):
            out[key] = "+inf" if v > 0 else "-inf"

    for k in ("score", "A", "C", "C_fwd", "C_bwd"):
        _fix(k)
    return out


def cmd_compare_m4(argv=None):
    """M4（A+C−M）對照：0507/0528；空窗橋接 A/C/M；亂接與常數清單。"""
    p = argparse.ArgumentParser(description="M4 A+C−M 對照實驗")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m4_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0507_top1.json",
            "m2a": tp.OUTPUT_ROOT / "v1.0" / "m2_comparison" / "0507_M2a_top1.json",
            "m3": tp.OUTPUT_ROOT / "v1.0" / "m3_comparison" / "0507_M3_top1.json",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0528_top1.json",
            "m2a": tp.OUTPUT_ROOT / "v1.0" / "m2_comparison" / "0528_M2a_top1.json",
            "m3": tp.OUTPUT_ROOT / "v1.0" / "m3_comparison" / "0528_M3_top1.json",
        },
    ]

    account = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "constants": [],
        "datasets": {},
    }
    results_by_short: dict = {}

    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])
        pack = {"gt_set": sorted(gt_set)}

        m0 = json.loads(ds["m0_json"].read_text(encoding="utf-8"))
        pack["M0"] = _top_pack(m0, gt_set, {"source": str(ds["m0_json"])})

        for key, path in (("M2a", ds["m2a"]), ("M3", ds["m3"])):
            if path.is_file():
                pack[key] = _top_pack(
                    json.loads(path.read_text(encoding="utf-8")),
                    gt_set,
                    {"source": str(path)},
                )
                print(f"[{short}] {key} 引用 {path.name}")
            else:
                pack[key] = {}
                print(f"[{short}] {key} 未找到：{path}")

        print(f"\n===== {short} M4 =====")
        cfg = RunConfig(
            scoring="m4",
            node_score=False,
            dt_max=None,
            sim_min=float(args.sim_min),
            variant_tag="M4",
        )
        result = run_with_config(ds["merge"], cfg)
        summary = _save_summary(result, ds["merge"], out_dir, f"{short}_M4_top1")
        pack["M4"] = _top_pack(summary, gt_set, {"constants": []})
        results_by_short[short] = {
            "result": result,
            "summary": summary,
            "gt_set": gt_set,
            "merge": ds["merge"],
        }
        account["datasets"][short] = pack

    # 拼圖
    _render_m4_collage(results_by_short, out_dir)

    # 報告
    report_path = args.report.resolve() if args.report else (out_dir / "comparison_m4.md")
    root_report = tp.REPO_ROOT.parent / "comparison_m4.md"
    text = _render_m4_report(account, results_by_short)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    account_path = out_dir / "comparison_m4_account.json"
    # 精簡帳本：0528 Top-1 邊
    if "0528" in results_by_short and results_by_short["0528"]["result"]["ranked"]:
        top = results_by_short["0528"]["result"]["ranked"][0]
        account["datasets"]["0528"]["m4_top1_edges"] = [
            _m4_edge_disp(e) for e in (top.get("edges") or [])
        ]
        account["datasets"]["0528"]["m4_top1_path"] = top.get("path")
        account["datasets"]["0528"]["m4_n_segments"] = top.get("n_segments")
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n報告：{report_path}")
    print(f"副本：{root_report}")
    print(f"帳本：{account_path}")
    return account


def _render_m4_collage(results_by_short: dict, out_dir: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[m4_collage] Pillow 不可用，跳過")
        return
    if "0528" not in results_by_short:
        return
    data = results_by_short["0528"]
    result = data["result"]
    gt_set = data["gt_set"]
    merge = data["merge"]
    ranked = result["ranked"]
    if not ranked:
        return
    top = ranked[0]
    by_tid = {t.tid: t for t in result["tracks"]}

    def _font(size: int):
        for name in (
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ):
            if Path(name).is_file():
                try:
                    return ImageFont.truetype(name, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    FONT_XS = _font(10)

    def members_of_label(lab):
        if lab.startswith("{") and lab.endswith("}"):
            return [x.strip() for x in lab[1:-1].split(",") if x.strip()]
        return [lab]

    def thumb(path, size):
        if path is None or not Path(path).is_file():
            return Image.new("RGB", size, (230, 230, 230))
        im = Image.open(path).convert("RGB")
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (236, 236, 236))
        canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
        return canvas

    segs = sorted(top.get("segments") or [], key=lambda s: int(s.get("segment") or 1))
    if not segs:
        segs = [{
            "segment": 1,
            "super_labels": top.get("super_labels") or [],
            "edges": top.get("edges") or [],
            "gap_after_prev_sec": None,
        }]

    tw, th = 100, 130
    margin, title_h, row_gap, cell_gap = 12, 72, 20, 72
    max_cols = 8
    rows = []
    for si, seg in enumerate(segs):
        if si > 0:
            g = seg.get("gap_after_prev_sec") or 178.3
            rows.append(("gap", f"觀測空窗 {g:.1f} 秒（此期間不在任何鏡頭視野內）"))
        labels = seg.get("super_labels") or []
        edges = seg.get("edges") or []
        # 若 segment 無 edges，用 top.edges 對齊單段
        if not edges and int(seg.get("segment") or 1) == 1:
            edges = top.get("edges") or []
        nodes_list = []
        for i, lab in enumerate(labels):
            mems = members_of_label(lab)
            if i == 0 and edges and edges[0].get("from_members"):
                mems = list(edges[0]["from_members"])
            elif i > 0 and i - 1 < len(edges) and edges[i - 1].get("to_members"):
                mems = list(edges[i - 1]["to_members"])
            e = edges[i] if i < len(edges) else None
            nodes_list.append((mems, e))
        for st in range(0, len(nodes_list), max_cols):
            rows.append(("nodes", nodes_list[st: st + max_cols]))

    rh, rw = [], []
    for row in rows:
        if row[0] == "gap":
            rh.append(30)
            rw.append(900)
            continue
        hmax = th + 72
        wsum = 0
        for mems, e in row[1]:
            cw = max(140, len(mems) * (tw + 8) + 12)
            wsum += cw + (cell_gap if e is not None else 0)
        rh.append(hmax)
        rw.append(wsum)

    W = margin * 2 + max(rw + [800])
    H = title_h + sum(rh) + row_gap * max(0, len(rh) - 1) + margin
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pr = precision_recall_vs_gt(top.get("tids") or [], gt_set)
    for i, line in enumerate([
        f"M4（A+C−M）0528 Top-1  n_seg={top.get('n_segments')}  score={top.get('score', 0):.4f}",
        f"prec={pr['precision']:.4f} ({pr['n_hit']}/{pr['n_path']})  "
        f"rec={pr['recall']:.4f} ({pr['n_hit']}/{pr['n_gt']})  "
        f"P={top.get('path_probability', 0):.6f}",
        "邊分 = A + C − M（無常數／無全圖統計）  GT 僅供評估與著色",
    ]):
        draw.text((margin, 6 + i * 14), line, fill=(10, 10, 10), font=FONT_XS)
    y = title_h
    for row in rows:
        if row[0] == "gap":
            draw.rectangle([margin, y, W - margin, y + 26], fill=(255, 245, 230), outline=(200, 120, 40))
            draw.text((margin + 6, y + 6), row[1], fill=(160, 80, 0), font=FONT_XS)
            y += 30 + row_gap
            continue
        x = margin
        hmax = th + 72
        for mems, e in row[1]:
            cw = max(140, len(mems) * (tw + 8) + 12)
            bc = (0, 160, 0) if all(m in gt_set for m in mems) else (200, 40, 40)
            draw.rectangle([x, y, x + cw - 1, y + hmax - 1], outline=bc, width=3)
            if len(mems) > 1:
                draw.text((x + 4, y + 2), "共存合併", fill=bc, font=FONT_XS)
            for i, tid in enumerate(mems):
                t = by_tid.get(tid)
                cam, tid_s = tid.rsplit("_", 1)
                try:
                    _, crops = tp._crop_paths_for_track(merge, cam, int(tid_s))
                    rep = tp._pick_rep_crop(crops)
                except Exception:
                    rep = None
                sx = x + 4 + i * (tw + 8)
                sy = y + 16
                img.paste(thumb(rep, (tw, th)), (sx, sy))
                mbc = (0, 160, 0) if tid in gt_set else (200, 40, 40)
                draw.rectangle([sx, sy, sx + tw - 1, sy + th - 1], outline=mbc, width=2)
                if t:
                    txt = f"{tid}\n{cam}\n[{t.t_start:.1f}-{t.t_end:.1f}]\nsim={t.sim:.3f}"
                else:
                    txt = tid
                ly = sy + th + 2
                for line in txt.split("\n"):
                    draw.text((sx, ly), line, fill=(20, 20, 20), font=FONT_XS)
                    ly += 11
            x += cw
            if e is not None:
                mid = y + hmax // 2
                draw.line([(x + 4, mid), (x + cell_gap - 8, mid)], fill=(40, 40, 40), width=2)
                A, C, M = e.get("A", 0), e.get("C", 0), e.get("M", 0)
                sc = e.get("score", 0)
                def _fmt(v):
                    if isinstance(v, (int, float)) and abs(v) >= 1e299:
                        return "+∞" if v > 0 else "−∞"
                    return f"{v:+.2f}"
                sc_col = (0, 140, 0) if (isinstance(sc, (int, float)) and sc >= 0) else (200, 0, 0)
                draw.multiline_text(
                    (x + 2, mid - 32),
                    f"A={_fmt(A)}\nC={_fmt(C)}\nM={M:.2f}",
                    fill=(40, 40, 140),
                    font=FONT_XS,
                )
                draw.text((x + 2, mid + 12), f"Σ={_fmt(sc)}", fill=sc_col, font=FONT_XS)
                x += cell_gap
        y += hmax + row_gap

    out_png = out_dir / "人員追蹤_20260528_m4_top1_collage.png"
    img.save(out_png)
    print(f"拼圖：{out_png}")


def _render_m4_report(account: dict, results_by_short: dict) -> str:
    def _fmtv(v):
        if isinstance(v, (int, float)) and abs(v) >= 1e299:
            return "+inf" if v > 0 else "-inf"
        if isinstance(v, float):
            return f"{v:+.3f}"
        return str(v)

    lines = [
        "# M4（A + C − M）對照實驗",
        "",
        f"生成時間：{account.get('generated_at')}",
        "",
        "> **GT 僅用於評估與著色**；M4 推論無校準、無全圖統計、無手調常數。",
        "",
        "M4 邊分 = A + C − M：",
        "- A = ln(((w_u+w_v)/2)/d_uv)；w=kept 兩兩 cosine distance 平均；"
        "**d_uv=代表向量（kept 平均 emb）cosine distance**",
        "- C = ln(N_u·P(v|u)) + ln(N_v_pred·P(u|v))；P 對 A 做 softmax",
        "- M = ln(1+miss)；miss=空檔內其他合法下家數+(hop2?1:0)",
        "",
        "## 1. Top-1 prec / rec / P 總表（M0 / M2a / M3 / M4）",
        "",
        "| 資料集 | 版 | precision | recall | P(softmax) | n_seg | n_path | n_hit |",
        "|--------|----|-----------:|-------:|-----------:|------:|-------:|------:|",
    ]
    for short, pack in account["datasets"].items():
        for key in ("M0", "M2a", "M3", "M4"):
            d = pack.get(key) or {}
            if not d:
                lines.append(f"| {short} | {key} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {short} | {key} | {_fmt_pct(d.get('precision', 0))} | "
                f"{_fmt_pct(d.get('recall', 0))} | {float(d.get('P') or 0):.6f} | "
                f"{d.get('n_segments')} | {d.get('n_path')} | {d.get('n_hit')} |"
            )
    lines.append("")

    # --- 2 空窗 ---
    lines += [
        "## 2. 0528 空窗檢驗：M4 是否還走 09_96 / 07_139 假橋？",
        "",
    ]
    data = results_by_short.get("0528")
    if data and data["result"]["ranked"]:
        top = data["result"]["ranked"][0]
        gt_set = data["gt_set"]
        n_seg = top.get("n_segments")
        path = top.get("path") or ""
        lines.append(f"**M4 Top-1：n_segments = {n_seg}**")
        lines.append(f"- 路徑：`{path}`")
        lines.append("")
        has_fake = ("K8-09_96" in (top.get("tids") or [])) or ("K8-07_139" in (top.get("tids") or []))
        lines.append(
            f"- 是否含假橋節點 `K8-09_96` / `K8-07_139`：**"
            f"{'是 ★' if has_fake else '否（已避開）'}**"
        )
        lines.append("")

        # 找橋接邊：dt>100 或涉及 09_96/07_139
        edges = top.get("edges") or []
        bridge = [
            e for e in edges
            if (e.get("dt") or 0) > 100
            or e.get("from") in ("K8-09_96", "K8-07_139")
            or e.get("to") in ("K8-09_96", "K8-07_139")
        ]
        # 也掃全圖合法邊裡 09_96→07_139（即使不在 Top-1）
        lines.append("### 橋接相關邊 A / C / M 分解")
        lines.append("")
        lines.append(
            "| from | to | hop | dt | A | C_fwd | C_bwd | C | M | miss | Σ | from-GT | to-GT |"
        )
        lines.append(
            "|------|-----|----:|---:|--:|------:|------:|--:|--:|-----:|--:|:-------:|:-----:|"
        )

        shown = bridge if bridge else [e for e in edges if (e.get("dt") or 0) > 50]
        if not shown:
            shown = edges[:3]
        for e in shown:
            lines.append(
                f"| `{e.get('from')}` | `{e.get('to')}` | {e.get('hop')} | "
                f"{e.get('dt', 0):.1f} | {_fmtv(e.get('A'))} | {_fmtv(e.get('C_fwd'))} | "
                f"{_fmtv(e.get('C_bwd'))} | {_fmtv(e.get('C'))} | {e.get('M', 0):.3f} | "
                f"{e.get('miss')} | {_fmtv(e.get('score'))} | "
                f"{'✓' if e.get('from') in gt_set else '✗'} | "
                f"{'✓' if e.get('to') in gt_set else '✗'} |"
            )
        lines.append("")
        if has_fake:
            lines.append(
                "解讀：假橋仍在 Top-1 → 反向競爭項（C_bwd）未能壓過其他正分累積；"
                "見下節 miss 計數。"
            )
        else:
            lines.append(
                "解讀：假橋未進 Top-1。重點看若系統中該邊仍合法，其 C_bwd / M 是否偏負。"
            )
        lines.append("")

        # 即使不在 top1，從 succ 圖找 09_96→07_139
        nodes = data["result"]["nodes"]
        by_label = {}
        for sn in nodes:
            by_label[sn.label] = sn
            for tid in sn.tids:
                by_label[tid] = sn
        # 重建 m4 邊找特定對
        tracks = data["result"]["tracks"]
        attach_crop_embs(tracks, data["merge"])
        succ_raw, _, _, _ = _build_succ_m4(nodes)
        target_edges = []
        seen_via = set()
        for i, items in enumerate(succ_raw):
            for j, e in items:
                via = (e.get("from"), e.get("to"))
                is_bridge = via == ("K8-09_96", "K8-07_139") or (
                    "K8-09_96" in (e.get("from_members") or [])
                    and "K8-07_139" in (e.get("to_members") or [])
                )
                if is_bridge and via not in seen_via:
                    seen_via.add(via)
                    target_edges.append(e)
        if target_edges:
            lines.append("### 系統內 `K8-09_96 → K8-07_139` 合法邊帳目（不論是否 Top-1）")
            lines.append("")
            for e in target_edges:
                lines.append(
                    f"- `{e['from']}`→`{e['to']}` hop={e.get('hop')} dt={e.get('dt'):.1f}s  "
                    f"A={_fmtv(e.get('A'))}  C={_fmtv(e.get('C'))} "
                    f"(fwd={_fmtv(e.get('C_fwd'))}, bwd={_fmtv(e.get('C_bwd'))})  "
                    f"M={e.get('M'):.3f} (miss={e.get('miss')}, "
                    f"others={e.get('miss_others')}, hop2={e.get('miss_hop2')})  "
                    f"Σ={_fmtv(e.get('score'))}  "
                    f"P_fwd={e.get('P_fwd'):.4f} P_bwd={e.get('P_bwd'):.4f}  "
                    f"N_u={e.get('N_u')} N_v_pred={e.get('N_v_pred')}"
                )
                if e.get("missed_labels"):
                    lines.append(f"  - 空檔內錯過下家：{e['missed_labels']}")
            lines.append("")

    # --- 3 178s 候選數 ---
    lines += [
        "## 3. 178s 空窗期間的合法候選數（錯過機會判定）",
        "",
    ]
    if data:
        # 前段真跡結束～後段真跡開始：用 M0 分段或固定 394→572
        m0_segs = []
        try:
            m0j = json.loads(
                (tp.OUTPUT_ROOT / "v1.0" / "0528_top1.json").read_text(encoding="utf-8")
            )
            m0_segs = (m0j.get("top1") or {}).get("segments") or []
        except Exception:
            pass
        if len(m0_segs) >= 2:
            t_gap0 = float(m0_segs[0].get("t_end") or 394.09)
            t_gap1 = float(m0_segs[1].get("t_start") or 572.38)
        else:
            t_gap0, t_gap1 = 394.09, 572.38
        lines.append(
            f"空窗區間取 M0 分段邊界：**(t_end_seg1, t_start_seg2) = "
            f"({t_gap0:.2f}, {t_gap1:.2f})，寬度 {t_gap1 - t_gap0:.2f}s**"
        )
        lines.append("")
        nodes = data["result"]["nodes"]
        in_gap = [
            sn for sn in nodes
            if t_gap0 < float(sn.t_start) < t_gap1
        ]
        lines.append(f"- 空窗內 `t_start` 落點的超節點數：**{len(in_gap)}**")
        for sn in in_gap:
            gt_set = data["gt_set"]
            flags = ",".join(
                ("GT" if tid in gt_set else "nonGT") for tid in sn.tids
            )
            lines.append(
                f"  - `{sn.label}` [{sn.t_start:.1f}–{sn.t_end:.1f}] ({flags})"
            )
        lines.append("")
        lines.append(
            "驗證：若某邊跨越此空窗（from.t_end ≤ 空窗起、to.t_start ≥ 空窗迄），"
            "miss_others 應計入上列於空檔內出現的其他合法下家。"
        )
        lines.append("")
        enum = (data["result"]["super_report"].get("enumeration") or {})
        if enum.get("mode") == "beam":
            lines.append(
                f"> **註記（非新增機制）**：M4 無 DT_MAX → 合法邊={enum.get('n_legal_edges')} "
                f"> FULL_ENUM_EDGE_CAP，沿用既有 beam 枚舉"
                f"（width={enum.get('beam_width')}, leaves={enum.get('n_beam_leaves')}）。"
                "Top-1 為 beam 近似，非全量 DFS。"
            )
            lines.append("")

    # --- 4 逐邊帳目 ---
    lines += [
        "## 4. Top-1 逐邊帳目（A / C / M / 小計）與拼圖",
        "",
        "拼圖：`output/v1.0/m4_comparison/人員追蹤_20260528_m4_top1_collage.png`",
        "",
    ]
    if data and data["result"]["ranked"]:
        top = data["result"]["ranked"][0]
        gt_set = data["gt_set"]
        edges = top.get("edges") or []
        lines.append(
            "| # | from_super → to_super | hop | dt | A | C | M | Σ | GT? |"
        )
        lines.append(
            "|--:|----------------------|----:|---:|--:|--:|--:|--|:---:|"
        )
        for i, e in enumerate(edges, 1):
            mems = list(e.get("from_members") or []) + list(e.get("to_members") or [])
            all_gt = all(t in gt_set for t in mems) if mems else False
            lines.append(
                f"| {i} | `{e.get('from_super')}` → `{e.get('to_super')}` | "
                f"{e.get('hop')} | {e.get('dt', 0):.1f} | {_fmtv(e.get('A'))} | "
                f"{_fmtv(e.get('C'))} | {e.get('M', 0):.3f} | {_fmtv(e.get('score'))} | "
                f"{'綠' if all_gt else '紅'} |"
            )
        lines.append("")

        # 0507 也列簡表
        if "0507" in results_by_short and results_by_short["0507"]["result"]["ranked"]:
            t07 = results_by_short["0507"]["result"]["ranked"][0]
            lines.append(f"### 0507 Top-1：`{t07.get('path')}`")
            lines.append("")

    # --- 5 常數清單 ---
    lines += [
        "## 5. 常數清單（給教授）",
        "",
        "**應為空：`[]`**",
        "",
        "本模式未引入：DT_MAX、EMB_EDGE_MIN、σ、μ、hop_pen 手調值、"
        "全圖 z-score、節點分權重、任何交換率常數。",
        "",
        "距離定義（非常數，是度量選擇，已於規格註明）：",
        "- cosine distance = 1 − cos（L2-normalized emb）",
        "- d_uv 採**代表向量距離**（非 crops 兩兩交叉平均）",
        "",
        "---",
        "",
        "實驗約束：GT 僅用於評估與著色；未修改 track_path.py / query_filter / "
        "config / calibration；未調參湊分數；未加未列機制。",
        "",
    ]
    return "\n".join(lines)


def cmd_compare_m4b(argv=None):
    """M4b = M4 計分 + Σ<0 可斷點分段同池。"""
    p = argparse.ArgumentParser(description="M4b 負分邊可斷點對照實驗")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m4b_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0507_top1.json",
            "m4": tp.OUTPUT_ROOT / "v1.0" / "m4_comparison" / "0507_M4_top1.json",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0528_top1.json",
            "m4": tp.OUTPUT_ROOT / "v1.0" / "m4_comparison" / "0528_M4_top1.json",
        },
    ]

    account = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "constants": [],
        "m4b_semantics": (
            "一條邊比隨機亂接還差（Σ<0）時，「承認斷開」獲得與「硬縫」同場競價的資格，"
            "由總分裁決；斷不斷不是規則決定，是競爭決定；全程零常數。"
        ),
        "datasets": {},
    }
    results_by_short: dict = {}

    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])
        pack: dict = {"gt_set": sorted(gt_set)}

        m0 = json.loads(ds["m0_json"].read_text(encoding="utf-8"))
        pack["M0"] = _top_pack(m0, gt_set, {"source": str(ds["m0_json"])})

        if ds["m4"].is_file():
            pack["M4"] = _top_pack(
                json.loads(ds["m4"].read_text(encoding="utf-8")),
                gt_set,
                {"source": str(ds["m4"])},
            )
            print(f"[{short}] M4 引用 {ds['m4'].name}")
        else:
            print(f"[{short}] M4 未找到，現場重跑 M4 供對照")
            r4 = run_with_config(
                ds["merge"],
                RunConfig(scoring="m4", node_score=False, dt_max=None,
                          sim_min=float(args.sim_min), variant_tag="M4"),
            )
            s4 = _save_summary(r4, ds["merge"], out_dir, f"{short}_M4_top1")
            pack["M4"] = _top_pack(s4, gt_set)

        print(f"\n===== {short} M4b =====")
        cfg = RunConfig(
            scoring="m4b",
            node_score=False,
            dt_max=None,
            sim_min=float(args.sim_min),
            variant_tag="M4b",
        )
        result = run_with_config(ds["merge"], cfg)
        summary = _save_summary(result, ds["merge"], out_dir, f"{short}_M4b_top1")
        pack["M4b"] = _top_pack(summary, gt_set, {"constants": []})

        top = result["ranked"][0] if result["ranked"] else None
        # 找同節點序的全縫版（single_maximal）與斷開版對照
        cont = None
        if top and result["options"].get("single_maximal_top1"):
            # 在 ranked 裡找 source=single_maximal 且 tids 與某一斷開版相關
            for h in result["ranked"]:
                if h.get("source") == "single_maximal" and cont is None:
                    # 取與 Top-1 相關的全縫：若 Top-1 是 break，continuous_score 已記
                    pass
        diag = {
            "top1_source": top.get("source") if top else None,
            "top1_n_segments": top.get("n_segments") if top else None,
            "top1_path": top.get("path") if top else None,
            "top1_score": top.get("score") if top else None,
            "break_edge_indices": top.get("break_edge_indices") if top else None,
            "break_edges": top.get("break_edges") if top else None,
            "continuous_score": top.get("continuous_score") if top else None,
            "score_gain_vs_continuous": top.get("score_gain_vs_continuous") if top else None,
            "n_m4b_break_added": (result["options"].get("ranking_meta") or {}).get(
                "n_m4b_break_added"
            ),
            "segments": top.get("segments") if top else None,
        }
        # 若 Top-1 是分段，找對應全縫（同 seed 的 continuous）
        if top and int(top.get("n_segments") or 1) > 1:
            # 搜尋 ranked 中 single_maximal，其 super_labels 串接 == 分段 labels 串接
            flat_labs = []
            for seg in top.get("segments") or []:
                flat_labs.extend(seg.get("super_labels") or [])
            for h in result["ranked"]:
                if h.get("source") != "single_maximal":
                    continue
                if list(h.get("super_labels") or []) == flat_labs:
                    diag["matched_continuous"] = {
                        "score": h.get("score"),
                        "P": h.get("path_probability"),
                        "path": h.get("path"),
                        "n_segments": h.get("n_segments"),
                    }
                    if diag.get("continuous_score") is None:
                        diag["continuous_score"] = h.get("score")
                        diag["score_gain_vs_continuous"] = float(top["score"]) - float(
                            h["score"]
                        )
                    break
            # 也找「只斷 09_96→07_139」的假設
            for h in result["ranked"]:
                be = h.get("break_edges") or []
                if len(be) == 1 and be[0].get("from") == "K8-09_96" and be[0].get("to") == "K8-07_139":
                    diag["break_at_09_96_07_139"] = {
                        "rank_score": h.get("score"),
                        "P": h.get("path_probability"),
                        "path": h.get("path"),
                        "n_segments": h.get("n_segments"),
                        "gain": h.get("score_gain_vs_continuous"),
                    }
                    break

        # 0507：負分邊數
        if top and int(top.get("n_segments") or 1) == 1:
            edges = top.get("edges") or []
            neg = [e for e in edges if _m4_edge_is_negative(e)]
            diag["n_negative_edges_in_top1"] = len(neg)
            diag["negative_edges"] = [
                {"from": e.get("from"), "to": e.get("to"), "score": e.get("score")}
                for e in neg
            ]

        pack["m4b_diag"] = diag
        # 路人 / 回歸檢查（0528）
        tids = set(top.get("tids") or []) if top else set()
        pack["m4b_checklist"] = {
            "has_K8-12_14": "K8-12_14" in tids,
            "has_K8-30_5": "K8-30_5" in tids,
            "has_K8-09_94": "K8-09_94" in tids,
            "has_K8-07_93": "K8-07_93" in tids,
            "has_K8-09_96": "K8-09_96" in tids,
            "has_K8-07_139": "K8-07_139" in tids,
            "has_K8-07_1": "K8-07_1" in tids,
            "has_K8-09_167": "K8-09_167" in tids,
            "false_positive": pack["M4b"].get("false_positive"),
            "false_negative": pack["M4b"].get("false_negative"),
        }

        results_by_short[short] = {
            "result": result,
            "summary": summary,
            "gt_set": gt_set,
            "merge": ds["merge"],
            "diag": diag,
        }
        account["datasets"][short] = pack

    _render_m4b_collages(results_by_short, out_dir)

    report_path = args.report.resolve() if args.report else (out_dir / "comparison_m4b.md")
    root_report = tp.REPO_ROOT.parent / "comparison_m4b.md"
    text = _render_m4b_report(account, results_by_short)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    account_path = out_dir / "comparison_m4b_account.json"
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n報告：{report_path}")
    print(f"副本：{root_report}")
    print(f"帳本：{account_path}")
    return account


def _render_m4b_collages(results_by_short: dict, out_dir: Path) -> None:
    """0507 / 0528 M4b Top-1 拼圖（分段則兩段+空窗）。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[m4b_collage] Pillow 不可用")
        return

    def _font(size: int):
        for name in (
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ):
            if Path(name).is_file():
                try:
                    return ImageFont.truetype(name, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    FONT_XS = _font(10)

    def members_of_label(lab):
        if lab.startswith("{") and lab.endswith("}"):
            return [x.strip() for x in lab[1:-1].split(",") if x.strip()]
        return [lab]

    def thumb(path, size):
        if path is None or not Path(path).is_file():
            return Image.new("RGB", size, (230, 230, 230))
        im = Image.open(path).convert("RGB")
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (236, 236, 236))
        canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
        return canvas

    def _fmt(v):
        if isinstance(v, (int, float)) and abs(v) >= 1e299:
            return "+∞" if v > 0 else "−∞"
        return f"{v:+.2f}"

    for short, tag_name in (("0507", "人員追蹤_20260507"), ("0528", "人員追蹤_20260528")):
        if short not in results_by_short:
            continue
        data = results_by_short[short]
        result = data["result"]
        gt_set = data["gt_set"]
        merge = data["merge"]
        if not result["ranked"]:
            continue
        top = result["ranked"][0]
        by_tid = {t.tid: t for t in result["tracks"]}

        segs = sorted(top.get("segments") or [], key=lambda s: int(s.get("segment") or 1))
        if not segs:
            segs = [{
                "segment": 1,
                "super_labels": top.get("super_labels") or [],
                "edges": top.get("edges") or [],
                "gap_after_prev_sec": None,
            }]

        tw, th = 100, 130
        margin, title_h, row_gap, cell_gap = 12, 80, 20, 72
        max_cols = 8
        rows = []
        for si, seg in enumerate(segs):
            if si > 0:
                g = seg.get("gap_after_prev_sec")
                g = 0.0 if g is None else float(g)
                rows.append(("gap", f"觀測空窗 {g:.1f} 秒（段間實際空窗；斷點競價，非 DT_MAX）"))
            labels = seg.get("super_labels") or []
            edges = seg.get("edges") or []
            if not edges and int(seg.get("segment") or 1) == 1 and int(top.get("n_segments") or 1) == 1:
                edges = top.get("edges") or []
            nodes_list = []
            for i, lab in enumerate(labels):
                mems = members_of_label(lab)
                if i == 0 and edges and edges[0].get("from_members"):
                    mems = list(edges[0]["from_members"])
                elif i > 0 and i - 1 < len(edges) and edges[i - 1].get("to_members"):
                    mems = list(edges[i - 1]["to_members"])
                e = edges[i] if i < len(edges) else None
                nodes_list.append((mems, e))
            for st in range(0, len(nodes_list), max_cols):
                rows.append(("nodes", nodes_list[st: st + max_cols]))

        rh, rw = [], []
        for row in rows:
            if row[0] == "gap":
                rh.append(30)
                rw.append(900)
                continue
            hmax = th + 72
            wsum = 0
            for mems, e in row[1]:
                cw = max(140, len(mems) * (tw + 8) + 12)
                wsum += cw + (cell_gap if e is not None else 0)
            rh.append(hmax)
            rw.append(wsum)

        W = margin * 2 + max(rw + [800])
        H = title_h + sum(rh) + row_gap * max(0, len(rh) - 1) + margin
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        pr = precision_recall_vs_gt(top.get("tids") or [], gt_set)
        src = top.get("source") or ""
        for i, line in enumerate([
            f"M4b（M4 + Σ<0 可斷點）{short} Top-1  n_seg={top.get('n_segments')}  "
            f"score={top.get('score', 0):.4f}  src={src}",
            f"prec={pr['precision']:.4f} ({pr['n_hit']}/{pr['n_path']})  "
            f"rec={pr['recall']:.4f} ({pr['n_hit']}/{pr['n_gt']})  "
            f"P={top.get('path_probability', 0):.6f}",
            "邊分 = A + C − M；斷開=不收負分邊  GT 僅供評估與著色  常數=[]",
        ]):
            draw.text((margin, 6 + i * 14), line, fill=(10, 10, 10), font=FONT_XS)

        y = title_h
        for row in rows:
            if row[0] == "gap":
                draw.rectangle(
                    [margin, y, W - margin, y + 26],
                    fill=(255, 245, 230),
                    outline=(200, 120, 40),
                )
                draw.text((margin + 6, y + 6), row[1], fill=(160, 80, 0), font=FONT_XS)
                y += 30 + row_gap
                continue
            x = margin
            hmax = th + 72
            for mems, e in row[1]:
                cw = max(140, len(mems) * (tw + 8) + 12)
                bc = (0, 160, 0) if all(m in gt_set for m in mems) else (200, 40, 40)
                draw.rectangle([x, y, x + cw - 1, y + hmax - 1], outline=bc, width=3)
                if len(mems) > 1:
                    draw.text((x + 4, y + 2), "共存合併", fill=bc, font=FONT_XS)
                for i, tid in enumerate(mems):
                    t = by_tid.get(tid)
                    cam, tid_s = tid.rsplit("_", 1)
                    try:
                        _, crops = tp._crop_paths_for_track(merge, cam, int(tid_s))
                        rep = tp._pick_rep_crop(crops)
                    except Exception:
                        rep = None
                    sx = x + 4 + i * (tw + 8)
                    sy = y + 16
                    img.paste(thumb(rep, (tw, th)), (sx, sy))
                    mbc = (0, 160, 0) if tid in gt_set else (200, 40, 40)
                    draw.rectangle(
                        [sx, sy, sx + tw - 1, sy + th - 1], outline=mbc, width=2
                    )
                    if t:
                        txt = f"{tid}\n{cam}\n[{t.t_start:.1f}-{t.t_end:.1f}]\nsim={t.sim:.3f}"
                    else:
                        txt = tid
                    ly = sy + th + 2
                    for line in txt.split("\n"):
                        draw.text((sx, ly), line, fill=(20, 20, 20), font=FONT_XS)
                        ly += 11
                x += cw
                if e is not None:
                    mid = y + hmax // 2
                    draw.line(
                        [(x + 4, mid), (x + cell_gap - 8, mid)],
                        fill=(40, 40, 40),
                        width=2,
                    )
                    A, C, M = e.get("A", 0), e.get("C", 0), e.get("M", 0)
                    sc = e.get("score", 0)
                    sc_col = (
                        (0, 140, 0)
                        if (isinstance(sc, (int, float)) and sc >= 0)
                        else (200, 0, 0)
                    )
                    draw.multiline_text(
                        (x + 2, mid - 32),
                        f"A={_fmt(A)}\nC={_fmt(C)}\nM={M:.2f}",
                        fill=(40, 40, 140),
                        font=FONT_XS,
                    )
                    draw.text(
                        (x + 2, mid + 12), f"Σ={_fmt(sc)}", fill=sc_col, font=FONT_XS
                    )
                    x += cell_gap
            y += hmax + row_gap

        out_png = out_dir / f"{tag_name}_m4b_top1_collage.png"
        img.save(out_png)
        print(f"拼圖：{out_png}")


def _render_m4b_report(account: dict, results_by_short: dict) -> str:
    lines = [
        "# M4b（M4 + Σ<0 可斷點分段）對照實驗",
        "",
        f"生成時間：{account.get('generated_at')}",
        "",
        f"> **語意**：{account.get('m4b_semantics')}",
        "",
        "> **GT 僅用於評估與著色**。M4b 計分 = M4（A+C−M）；只補假設層。",
        "",
        "## 1. 總表：M0 / M4 / M4b",
        "",
        "| 資料集 | 版 | precision | recall | P | n_seg | n_path | n_hit |",
        "|--------|----|-----------:|-------:|--:|------:|-------:|------:|",
    ]
    for short, pack in account["datasets"].items():
        for key in ("M0", "M4", "M4b"):
            d = pack.get(key) or {}
            if not d:
                lines.append(f"| {short} | {key} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {short} | {key} | {_fmt_pct(d.get('precision', 0))} | "
                f"{_fmt_pct(d.get('recall', 0))} | {float(d.get('P') or 0):.6f} | "
                f"{d.get('n_segments')} | {d.get('n_path')} | {d.get('n_hit')} |"
            )
    lines.append("")

    # --- 0528 主戲 ---
    lines += ["## 2. 主戲——0528：M4b Top-1 是否分段？", ""]
    pack = account["datasets"].get("0528") or {}
    diag = pack.get("m4b_diag") or {}
    chk = pack.get("m4b_checklist") or {}
    m4b = pack.get("M4b") or {}
    n_seg = diag.get("top1_n_segments")
    lines.append(f"**n_segments = {n_seg}**；source = `{diag.get('top1_source')}`")
    lines.append(f"- 路徑：`{diag.get('top1_path')}`")
    lines.append(f"- 總分：{diag.get('top1_score')}")
    lines.append("")

    if n_seg and int(n_seg) > 1:
        lines.append("### a. 斷點是否在 09_96→07_139？")
        lines.append("")
        be = diag.get("break_edges") or []
        hit = any(
            (b.get("from") == "K8-09_96" and b.get("to") == "K8-07_139")
            or (b.get("from_super") == "K8-09_96" and b.get("to_super") == "K8-07_139")
            for b in be
        )
        lines.append(f"- Top-1 斷點列表：`{be}`")
        lines.append(f"- 是否含 09_96→07_139：**{'是 ★' if hit else '否'}**")
        if diag.get("break_at_09_96_07_139"):
            lines.append(f"- 單斷該邊之假設：`{diag['break_at_09_96_07_139']}`")
        lines.append("")

        lines.append("### b. 分段版 vs 全縫版總分")
        lines.append("")
        cont = diag.get("matched_continuous") or {}
        lines.append(f"- 分段版 score = **{diag.get('top1_score')}**")
        lines.append(
            f"- 全縫版 score = **{diag.get('continuous_score')}** "
            f"（matched={cont.get('score')}）"
        )
        gain = diag.get("score_gain_vs_continuous")
        if gain is not None:
            lines.append(f"- 分段贏全縫：**{gain:+.4f}**")
        lines.append("")

        lines.append("### c. 12_14 / 30_5 回歸？五個路人出局？")
        lines.append("")
        lines.append(f"- `K8-12_14` 在路徑：**{chk.get('has_K8-12_14')}**")
        lines.append(f"- `K8-30_5` 在路徑：**{chk.get('has_K8-30_5')}**")
        lines.append(f"- `K8-09_94` 在路徑：**{chk.get('has_K8-09_94')}**")
        lines.append(
            f"- 路人殘留：07_1={chk.get('has_K8-07_1')}  "
            f"07_93={chk.get('has_K8-07_93')}  "
            f"09_96={chk.get('has_K8-09_96')}  "
            f"07_139={chk.get('has_K8-07_139')}  "
            f"09_167={chk.get('has_K8-09_167')}"
        )
        lines.append(f"- FP：{chk.get('false_positive')}")
        lines.append(f"- FN：{chk.get('false_negative')}")
        lines.append("")
        lines.append("各段：")
        for seg in diag.get("segments") or []:
            g = seg.get("gap_after_prev_sec")
            lines.append(
                f"- seg{seg.get('segment')}: [{seg.get('t_start'):.1f}–{seg.get('t_end'):.1f}] "
                f"score={seg.get('score'):.4f}  "
                f"{'gap='+format(g,'.1f')+'s  ' if g is not None else ''}"
                f"`{seg.get('path')}`"
            )
        lines.append("")
    else:
        lines.append("Top-1 **仍為單段**（未選分段假設）。")
        lines.append(f"- n_m4b_break_added（池內新增斷開假設數）= {diag.get('n_m4b_break_added')}")
        lines.append("")

    # --- 0507 ---
    lines += [
        "## 3. 0507 回歸：M4b 應退化為 M4（無負分邊 → 無斷點）",
        "",
    ]
    p07 = account["datasets"].get("0507") or {}
    d07 = p07.get("m4b_diag") or {}
    m4 = p07.get("M4") or {}
    m4b = p07.get("M4b") or {}
    same = (m4.get("tids") == m4b.get("tids")) and (
        int(m4b.get("n_segments") or 1) == 1
    )
    lines.append(f"- M4  path：`{m4.get('path')}`")
    lines.append(f"- M4b path：`{m4b.get('path')}`")
    lines.append(f"- M4b n_seg：{m4b.get('n_segments')}")
    lines.append(f"- Top-1 負分邊數：{d07.get('n_negative_edges_in_top1')}")
    lines.append(f"- **路徑與單段是否與 M4 一致：{'是 ✓' if same else '否 ★'}**")
    if not same:
        lines.append(
            f"- 差異：M4 tids={m4.get('tids')}；M4b tids={m4b.get('tids')}；"
            f"neg={d07.get('negative_edges')}"
        )
    lines.append("")

    # --- 拼圖 ---
    lines += [
        "## 4. 拼圖",
        "",
        "- `output/v1.0/m4b_comparison/人員追蹤_20260507_m4b_top1_collage.png`",
        "- `output/v1.0/m4b_comparison/人員追蹤_20260528_m4b_top1_collage.png`",
        "",
        "## 5. 常數清單（給教授）",
        "",
        "**應仍為空：`[]`**",
        "",
        "M4b 未引入秒數門檻、DT_MAX、權重或任何手調常數；"
        "可斷點條件僅 Σ<0（分數符號，非常數）。",
        "",
        "---",
        "",
        "實驗約束：GT 僅評估與著色；未改 M4 計分；未改 track_path.py；未調參。",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# validate_emb_edge：邊層級外觀計分驗證（emb LLR vs A）
# 純輸出；建邊僅時間順序（無 hop / DT_MAX / 門檻）；不改既有模式。
# ============================================================

def _sn_rep_from_crops(members: list) -> tuple[np.ndarray, float, int]:
    """
    節點代表向量 = kept crops 逐維平均後 L2；超節點=成員 crops 合併。
    回傳 (rep_emb, w_intra, n_crops)。
    """
    embs: list = []
    for t in members:
        crops = t.meta.get("crop_embs") or []
        if crops:
            embs.extend(crops)
        elif t.emb is not None:
            embs.append(np.asarray(t.emb, dtype=np.float64))
    if not embs:
        raise RuntimeError(
            f"validate_emb_edge 異常：超節點無 crop emb（members={[t.tid for t in members]}）"
        )
    mean = np.mean(np.stack(embs, axis=0), axis=0)
    rep = tp._l2_normalize(np.asarray(mean, dtype=np.float64))
    return rep, float(_pairwise_mean_dist(embs)), len(embs)


def _m4_A_from_reps(w_u: float, w_v: float, emb: float) -> float:
    """A = ln(((w_u+w_v)/2) / d_uv)；d_uv = 1 − emb（cosine）。"""
    num = 0.5 * (float(w_u) + float(w_v))
    d_uv = float(1.0 - float(emb))
    if d_uv == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    if num == 0.0:
        return float("-inf")
    return float(math.log(num / d_uv))


def _time_order_ok_member(u: tp.Track, v: tp.Track, dt_raw: float) -> bool:
    key = tuple(sorted((u.cam, v.cam)))
    tol = tp.OVERLAP_PAIRS.get(key, tp.TOL)
    if dt_raw < -tol:
        h_ok, _ = tp.same_object_h(u, v)
        if not (h_ok or tp.corridor_prefers(u, v)):
            return False
    return True


def _member_hop(u: tp.Track, v: tp.Track):
    hop = tp.hop_count(u.cam, v.cam)
    if hop is None:
        h_ok, _ = tp.same_object_h(u, v)
        if h_ok and tuple(sorted((u.cam, v.cam))) in tp.ADJACENT:
            hop = 1
    return hop


def _build_time_only_emb_edges(nodes: list) -> list[dict]:
    """
    教授新提案：僅時間順序建邊（重疊對沿用交接容許）。
    無 hop / DT_MAX / 任何門檻。emb / A 用節點代表向量（crops 合併）。
    """
    # 預計算每節點 rep / w
    reps = []
    ws = []
    n_crops = []
    for sn in nodes:
        rep, w, nc = _sn_rep_from_crops(sn.members)
        reps.append(rep)
        ws.append(w)
        n_crops.append(nc)

    n = len(nodes)
    edges: list[dict] = []
    for i, j in itertools.permutations(range(n), 2):
        sa, sb = nodes[i], nodes[j]
        dt_raw = float(sb.t_start - sa.t_end)
        # 至少一對成員通過時間順序
        time_ok = False
        hops = []
        for u in sa.members:
            for v in sb.members:
                if _time_order_ok_member(u, v, dt_raw):
                    time_ok = True
                    h = _member_hop(u, v)
                    if h is not None:
                        hops.append(int(h))
        if not time_ok:
            continue
        emb = float(np.dot(reps[i], reps[j]))
        A = _m4_A_from_reps(ws[i], ws[j], emb)
        hop = min(hops) if hops else None  # None = 拓撲不可達但仍保留（時間序邊）
        edges.append(
            {
                "_i": i,
                "_j": j,
                "from_super": sa.label,
                "to_super": sb.label,
                "from_members": list(sa.tids),
                "to_members": list(sb.tids),
                "dt": float(max(dt_raw, 0.0)),
                "dt_raw": float(dt_raw),
                "hop": hop,
                "hop_le2": hop is not None and hop <= 2,
                "emb": emb,
                "A": A,
                "w_u": float(ws[i]),
                "w_v": float(ws[j]),
                "n_crops_u": int(n_crops[i]),
                "n_crops_v": int(n_crops[j]),
            }
        )
    return edges


def _finite(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(xf):
        return None
    # M4/M5 把 ±inf 存成 ±1e300；統計須排除
    if abs(xf) >= 1e299:
        return None
    return xf


def _score_stats(vals: list) -> dict:
    arr = np.asarray([v for v in (_finite(x) for x in vals) if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "pos_rate": None,
            "n_pos": 0,
            "neg_rate": None,
            "n_neg": 0,
            "n_raw": len(vals),
            "n_excluded_inf": len(vals),
        }
    n_pos = int(np.sum(arr > 0))
    n_neg = int(np.sum(arr < 0))
    n_excl = int(len(vals) - arr.size)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "pos_rate": float(n_pos / arr.size),
        "n_pos": n_pos,
        "neg_rate": float(n_neg / arr.size),
        "n_neg": n_neg,
        "n_raw": len(vals),
        "n_excluded_inf": n_excl,
    }


def _fmt_stat(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
        return "+∞" if v > 0 else ("−∞" if v < 0 else "nan")
    if isinstance(v, float):
        return f"{v:+.{nd}f}"
    return str(v)


def _effect_size(st_gt: dict, st_ng: dict):
    """Cohen's d ≈ (mean_gt − mean_ng) / pooled_std。"""
    if st_gt.get("mean") is None or st_ng.get("mean") is None:
        return None, None
    mean_diff = float(st_gt["mean"]) - float(st_ng["mean"])
    n1, n2 = int(st_gt["n"]), int(st_ng["n"])
    s1 = float(st_gt.get("std") or 0.0)
    s2 = float(st_ng.get("std") or 0.0)
    if n1 + n2 < 3:
        return mean_diff, None
    if n1 < 2 and n2 < 2:
        return mean_diff, None
    if n1 < 2:
        pooled = s2
    elif n2 < 2:
        pooled = s1
    else:
        pooled = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled == 0.0:
        return mean_diff, None
    return mean_diff, mean_diff / pooled


def _rank_among_out(edges: list[dict], e: dict, key: str) -> tuple[int | None, int]:
    outs = [x for x in edges if x["_i"] == e["_i"]]
    outs.sort(
        key=lambda x: (
            -(_finite(x.get(key)) if _finite(x.get(key)) is not None else -1e300)
        )
    )
    for k, o in enumerate(outs, 1):
        if o["_j"] == e["_j"]:
            return k, len(outs)
    return None, len(outs)


def _lookalike_note(edge: dict, lookalike_tids: set[str]) -> str:
    if not lookalike_tids:
        return "否（本資料集無既知互像名單）"
    hit = sorted(
        {
            t
            for t in (edge.get("from_members") or []) + (edge.get("to_members") or [])
            if t in lookalike_tids
        }
    )
    if hit:
        return f"是（既知互像／誤標：{', '.join(hit)}）"
    return "否"


def _analyze_emb_edge_dataset(
    short: str,
    merge: Path,
    gt_path: Path,
    calib: dict,
    *,
    sim_min: float = 0.85,
) -> dict:
    tp.SIM_MIN = float(sim_min)
    tp.configure_for_input(str(merge))
    tracks = tp.load_tracks(str(merge))
    attach_crop_embs(tracks, merge)
    coexist_median = median_edge_emb(tracks)
    nodes, super_report = tp.build_supernodes(tracks, overlap_emb_min=coexist_median)

    edges = _build_time_only_emb_edges(nodes)
    emb_same = calib["emb_same"]
    emb_diff = calib["emb_diff"]
    for e in edges:
        e["LLR"] = float(tp.llr(emb_same, emb_diff, float(e["emb"])))

    by_tid = {t.tid: t for t in tracks}
    tid_to_sn = {}
    for i, sn in enumerate(nodes):
        for tid in sn.tids:
            tid_to_sn[tid] = i

    gt_obj = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_tids = list(gt_obj["person_tids"])
    lookalike_tids = set(gt_obj.get("removed_mislabel") or [])
    # 0507：K8-08_43 為既知互像誤標；confirmed_negatives 非互像，不併入

    gt_tracks = [by_tid[t] for t in gt_tids if t in by_tid]
    missing = [t for t in gt_tids if t not in by_tid]
    ordered = sorted(gt_tracks, key=lambda t: (t.t_start, t.t_end, t.tid))
    ej = {(e["_i"], e["_j"]): e for e in edges}

    gt_edge_recs = []
    same_sn = []
    no_edge = []
    for u, v in zip(ordered, ordered[1:]):
        iu, iv = tid_to_sn.get(u.tid), tid_to_sn.get(v.tid)
        if iu is None or iv is None:
            no_edge.append((u.tid, v.tid, "missing_sn"))
            continue
        if iu == iv:
            same_sn.append((u.tid, v.tid))
            continue
        e = ej.get((iu, iv))
        if e is None:
            no_edge.append((u.tid, v.tid, "no_time_edge"))
            continue
        gt_edge_recs.append(
            {
                "gt_from": u.tid,
                "gt_to": v.tid,
                "edge": e,
            }
        )

    gt_keys = {(r["edge"]["_i"], r["edge"]["_j"]) for r in gt_edge_recs}
    for e in edges:
        e["is_gt"] = (e["_i"], e["_j"]) in gt_keys

    def subset_pack(edge_list: list[dict]) -> dict:
        """統計與排名皆限定在 edge_list 內（全體或 hop≤2 子集）。"""
        gt_e = [e for e in edge_list if e["is_gt"]]
        ng_e = [e for e in edge_list if not e["is_gt"]]
        pack = {}
        for score_key in ("LLR", "A"):
            st_gt = _score_stats([e[score_key] for e in gt_e])
            st_ng = _score_stats([e[score_key] for e in ng_e])
            md, d = _effect_size(st_gt, st_ng)
            ranks = []
            for e in gt_e:
                rk, n_out = _rank_among_out(edge_list, e, score_key)
                ranks.append({"edge": e, "rank": rk, "n_out": n_out})
            n_gt = len(gt_e)
            r1 = sum(1 for r in ranks if r["rank"] == 1)
            r3 = sum(1 for r in ranks if r["rank"] is not None and r["rank"] <= 3)
            pack[score_key] = {
                "st_gt": st_gt,
                "st_ng": st_ng,
                "mean_diff": md,
                "effect_d": d,
                "ranks": ranks,
                "rank1_rate": (r1 / n_gt) if n_gt else None,
                "rank3_rate": (r3 / n_gt) if n_gt else None,
                "n_rank1": r1,
                "n_rank3": r3,
                "n_gt": n_gt,
            }
        pack["n_edges"] = len(edge_list)
        pack["n_gt"] = len(gt_e)
        pack["n_nongt"] = len(ng_e)
        pack["gt_edges"] = gt_e
        pack["nongt_edges"] = ng_e
        return pack

    all_pack = subset_pack(edges)
    hop2_edges = [e for e in edges if e["hop_le2"]]
    hop2_pack = subset_pack(hop2_edges)

    # 個案（以全體邊為準）
    gt_by_llr = sorted(
        all_pack["gt_edges"],
        key=lambda e: (
            _finite(e["LLR"]) if _finite(e["LLR"]) is not None else 1e300
        ),
    )
    ng_by_llr = sorted(
        all_pack["nongt_edges"],
        key=lambda e: -(
            _finite(e["LLR"]) if _finite(e["LLR"]) is not None else -1e300
        ),
    )
    cases = {
        "worst_gt_llr": gt_by_llr[:3],
        "best_nongt_llr": ng_by_llr[:5],
    }

    return {
        "short": short,
        "merge": str(merge),
        "n_tracks": len(tracks),
        "n_nodes": len(nodes),
        "n_edges_time": len(edges),
        "n_edges_hop_le2": len(hop2_edges),
        "coexist_median": float(coexist_median),
        "super_report": {
            "n_merged_pairs": super_report.get("n_merged_pairs"),
            "n_supernodes": super_report.get("n_supernodes"),
        },
        "gt_tids": gt_tids,
        "missing": missing,
        "n_gt_adj": max(0, len(ordered) - 1),
        "same_sn": same_sn,
        "no_edge": no_edge,
        "lookalike_tids": sorted(lookalike_tids),
        "calib_emb_same": {
            "mu": float(emb_same["mu"]),
            "sigma": float(emb_same["sigma"]),
            "n": emb_same.get("n"),
            "shrink_w": emb_same.get("shrink_w"),
        },
        "calib_emb_diff": {
            "mu": float(emb_diff["mu"]),
            "sigma": float(emb_diff["sigma"]),
            "n": emb_diff.get("n"),
            "shrink_w": emb_diff.get("shrink_w"),
        },
        "all": all_pack,
        "hop2": hop2_pack,
        "cases": cases,
        "edges": edges,
    }


def _render_emb_edge_report(pack: dict) -> str:
    short = pack["short"]
    lines = [
        f"# emb LLR vs A 邊層級外觀驗證 — {short}",
        "",
        "> **GT 僅用於分組評估，不參與計分。**",
        "",
        "圖規格（教授新提案）：超節點合併照舊；建邊**僅**時間順序"
        "（`v.t_start >= u.t_end`，重疊鏡頭對沿用交接容許）；"
        "**無 hop 限制、無 DT_MAX、無任何門檻**。",
        "",
        "emb 定義：節點代表向量 = kept crops 逐維平均後 L2；"
        "超節點 = 成員 crops 合併後再平均＋L2；`edge_emb` = 兩代表向量 cosine。",
        "",
        f"LLR = ln(f_same(emb)/f_diff(emb))，密度取自 `calibration_gt0507.pkl`："
        f" emb_same=N({pack['calib_emb_same']['mu']:.6f},"
        f"{pack['calib_emb_same']['sigma']:.6f})，"
        f" emb_diff=N({pack['calib_emb_diff']['mu']:.6f},"
        f"{pack['calib_emb_diff']['sigma']:.6f})。"
        " **不乘收縮權重**（單證據排名測試中 w 為共同倍率，不影響排序）。",
        "",
        "A = ln(((w_u+w_v)/2)/d_uv)，M4 自量尺；同批邊同步計算。",
        "",
        "## 1. 圖規模",
        "",
        f"- tracks：{pack['n_tracks']}；超節點：{pack['n_nodes']}",
        f"- 時間順序邊總數：**{pack['n_edges_time']}**",
        f"- 其中 hop≤2 的邊數（對照）：**{pack['n_edges_hop_le2']}**",
        f"- 共存合併中位數門檻（名單制）：{pack['coexist_median']:.4f}",
        "",
        "## 2. 邊分組",
        "",
        "- **GT 邊**：GT 清單依 `t_start` 排序後時間相鄰的真轉移，"
        "且兩端不在同一超節點，且圖上存在對應時間順序邊。",
        "- **非 GT 邊**：圖上其餘所有時間順序邊。",
        "- GT 僅用於分組評估，不參與計分。",
        "",
        f"- GT tids：{len(pack['gt_tids'])}（缺載入：{pack['missing'] or '無'}）",
        f"- GT 相鄰對數：{pack['n_gt_adj']}",
        f"- 同超節點相鄰（略）：{len(pack['same_sn'])} → {pack['same_sn']}",
        f"- 相鄰但無時間序邊：{len(pack['no_edge'])} → {pack['no_edge']}",
        f"- 既知互像／誤標名單：{pack['lookalike_tids'] or '（空）'}",
        f"- **GT 邊**：{pack['all']['n_gt']}；**非 GT 邊**：{pack['all']['n_nongt']}",
        f"- hop≤2 子集內 GT 邊：{pack['hop2']['n_gt']}；非 GT：{pack['hop2']['n_nongt']}",
        "",
    ]

    def _block(title: str, sub: dict):
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"邊數：{sub['n_edges']}（GT={sub['n_gt']}，非GT={sub['n_nongt']}）")
        lines.append("")
        for score_key, label in (("LLR", "emb LLR"), ("A", "A 自量尺")):
            s = sub[score_key]
            lines.append(f"### {label}")
            lines.append("")
            lines.append(
                "| 組別 | n | mean | median | min | max | >0 比例 |"
            )
            lines.append("|------|--:|-----:|-------:|----:|----:|--------:|")
            for tag, st in (("GT 邊", s["st_gt"]), ("非 GT 邊", s["st_ng"])):
                pr = (
                    f"{st['pos_rate']:.1%} ({st['n_pos']}/{st['n']})"
                    if st["pos_rate"] is not None
                    else "—"
                )
                lines.append(
                    f"| {tag} | {st['n']} | {_fmt_stat(st['mean'])} | "
                    f"{_fmt_stat(st['median'])} | {_fmt_stat(st['min'])} | "
                    f"{_fmt_stat(st['max'])} | {pr} |"
                )
            lines.append("")
            lines.append("GT 邊在同 u 出邊中的排名：")
            lines.append("")
            lines.append("| GT 邊 | 分數 | 排名 | 出邊數 |")
            lines.append("|-------|-----:|-----:|-------:|")
            for r in s["ranks"]:
                e = r["edge"]
                lines.append(
                    f"| `{e['from_super']}`→`{e['to_super']}` | "
                    f"{_fmt_stat(e[score_key])} | {r['rank']} | {r['n_out']} |"
                )
            lines.append("")
            r1 = s["rank1_rate"]
            r3 = s["rank3_rate"]
            lines.append(
                f"- 第 1 名比例：**{(f'{r1:.1%}' if r1 is not None else '—')}**"
                f"（{s['n_rank1']}/{s['n_gt']}）"
            )
            lines.append(
                f"- 前 3 名比例：**{(f'{r3:.1%}' if r3 is not None else '—')}**"
                f"（{s['n_rank3']}/{s['n_gt']}）"
            )
            lines.append("")

    _block("3. 對照統計 — 全體時間順序邊", pack["all"])
    _block("4. 對照統計 — hop≤2 子集（避免遠距爛邊灌高分離度）", pack["hop2"])

    lines += [
        "## 5. 個案",
        "",
        "### 5.1 LLR 最低的 3 條 GT 邊",
        "",
    ]
    for i, e in enumerate(pack["cases"]["worst_gt_llr"], 1):
        rk_llr, n_out = _rank_among_out(pack["edges"], e, "LLR")
        lines.append(
            f"#### ({i}) `{e['from_super']}` → `{e['to_super']}`"
        )
        lines.append("")
        lines.append(
            f"- tids：from={e['from_members']} → to={e['to_members']}"
        )
        lines.append(
            f"- emb={e['emb']:.4f}；LLR={_fmt_stat(e['LLR'])}；"
            f"A={_fmt_stat(e['A'])}；hop={e['hop']}；dt={e['dt']:.1f}s"
        )
        lines.append(f"- 同 u 出邊 LLR 排名：{rk_llr}/{n_out}")
        lines.append(f"- 既知互像路人：{_lookalike_note(e, set(pack['lookalike_tids']))}")
        lines.append("")

    lines += [
        "### 5.2 LLR 最高的 5 條非 GT 邊",
        "",
    ]
    for i, e in enumerate(pack["cases"]["best_nongt_llr"], 1):
        lines.append(
            f"#### ({i}) `{e['from_super']}` → `{e['to_super']}`"
        )
        lines.append("")
        lines.append(
            f"- tids：from={e['from_members']} → to={e['to_members']}"
        )
        lines.append(
            f"- emb={e['emb']:.4f}；LLR={_fmt_stat(e['LLR'])}；"
            f"A={_fmt_stat(e['A'])}；hop={e['hop']}；dt={e['dt']:.1f}s"
        )
        lines.append(f"- 既知互像路人：{_lookalike_note(e, set(pack['lookalike_tids']))}")
        lines.append("")

    lines += [
        "## 6. 頭對頭：LLR vs A",
        "",
        "| 範圍 | 分數 | GT mean | 非GT mean | mean差 | 效應量 d | 第1名比例 | 前3比例 |",
        "|------|------|--------:|----------:|-------:|--------:|----------:|--------:|",
    ]
    for scope_name, sub in (("全體邊", pack["all"]), ("hop≤2", pack["hop2"])):
        for score_key in ("LLR", "A"):
            s = sub[score_key]
            d = s["effect_d"]
            r1s = (
                f"{s['rank1_rate']:.1%}" if s["rank1_rate"] is not None else "—"
            )
            r3s = (
                f"{s['rank3_rate']:.1%}" if s["rank3_rate"] is not None else "—"
            )
            lines.append(
                f"| {scope_name} | {score_key} | "
                f"{_fmt_stat(s['st_gt']['mean'])} | {_fmt_stat(s['st_ng']['mean'])} | "
                f"{_fmt_stat(s['mean_diff'])} | "
                f"{_fmt_stat(d, 3) if d is not None else '—'} | "
                f"{r1s} | {r3s} |"
            )
    lines.append("")

    # 一句結論
    def _winner(sub):
        d_llr = sub["LLR"]["effect_d"]
        d_a = sub["A"]["effect_d"]
        r_llr = sub["LLR"]["rank1_rate"]
        r_a = sub["A"]["rank1_rate"]
        sep = "LLR" if (d_llr or -1e9) >= (d_a or -1e9) else "A"
        rank = "LLR" if (r_llr or -1) >= (r_a or -1) else "A"
        return sep, rank, d_llr, d_a, r_llr, r_a

    sep_all, rank_all, dL, dA, rL, rA = _winner(pack["all"])
    sep_h, rank_h, dLh, dAh, rLh, rAh = _winner(pack["hop2"])
    lines.append("### 結論")
    lines.append("")

    def _concl(scope, sep, rank, d_llr, d_a, r_llr, r_a):
        dpart = f"（LLR d={_fmt_stat(d_llr)}，A d={_fmt_stat(d_a)}）"
        if r_llr is not None and r_a is not None:
            rpart = f"（LLR {r_llr:.0%} vs A {r_a:.0%}）"
        else:
            rpart = ""
        return (
            f"- **{scope}**：分離度（效應量）較大者 = **{sep}**{dpart}；"
            f"第1名比例較高者 = **{rank}**{rpart}。"
        )

    lines.append(_concl("全體邊", sep_all, rank_all, dL, dA, rL, rA))
    lines.append(_concl("hop≤2 子集", sep_h, rank_h, dLh, dAh, rLh, rAh))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "實驗約束：純輸出、未改既有 M1–M4b 邏輯；未改 track_path.py；"
        "校準僅供本驗證讀取密度。"
    )
    lines.append("")
    return "\n".join(lines)


def cmd_validate_emb_edge(argv=None):
    """邊層級外觀計分驗證：emb LLR vs A（時間順序圖）。"""
    p = argparse.ArgumentParser(description="validate_emb_edge：emb LLR vs A 驗證")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m4_comparison",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "calibration_gt0507.pkl",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = tp.load_calibration(args.calibration.resolve())

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
        },
    ]

    packs = []
    for ds in datasets:
        print(f"\n===== validate_emb_edge {ds['short']} =====")
        pack = _analyze_emb_edge_dataset(
            ds["short"],
            ds["merge"],
            ds["gt"],
            calib,
            sim_min=float(args.sim_min),
        )
        text = _render_emb_edge_report(pack)
        path = out_dir / f"{ds['short']}_emb_llr_edge_validation.md"
        path.write_text(text, encoding="utf-8")
        pack["_report_text"] = text
        pack["_report_path"] = str(path)
        packs.append(pack)
        print(
            f"[{ds['short']}] nodes={pack['n_nodes']} "
            f"edges={pack['n_edges_time']} hop<=2={pack['n_edges_hop_le2']} "
            f"GT={pack['all']['n_gt']} → {path}"
        )

    combo = out_dir / "emb_llr_edge_validation.md"
    parts = [
        "# emb LLR vs A 邊層級外觀驗證",
        "",
        "> **GT 僅用於分組評估，不參與計分。**",
        "",
        "本檔含 0507 / 0528；分檔見 `0507_emb_llr_edge_validation.md`、"
        "`0528_emb_llr_edge_validation.md`。",
        "",
        f"生成時間：{datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for pack in packs:
        parts.append("---")
        parts.append("")
        parts.append(pack["_report_text"])
        parts.append("")
    combo.write_text("\n".join(parts), encoding="utf-8")
    print(f"\n合併報告：{combo}")

    # 精簡帳本
    account = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "calibration": str(args.calibration.resolve()),
        "note_no_shrink": (
            "LLR 不乘 shrink_w：單證據排名測試中 w 為共同倍率，不影響排序。"
        ),
        "datasets": {},
    }
    for pack in packs:
        account["datasets"][pack["short"]] = {
            "n_nodes": pack["n_nodes"],
            "n_edges_time": pack["n_edges_time"],
            "n_edges_hop_le2": pack["n_edges_hop_le2"],
            "n_gt": pack["all"]["n_gt"],
            "all_LLR_effect_d": pack["all"]["LLR"]["effect_d"],
            "all_A_effect_d": pack["all"]["A"]["effect_d"],
            "all_LLR_rank1": pack["all"]["LLR"]["rank1_rate"],
            "all_A_rank1": pack["all"]["A"]["rank1_rate"],
            "hop2_LLR_effect_d": pack["hop2"]["LLR"]["effect_d"],
            "hop2_A_effect_d": pack["hop2"]["A"]["effect_d"],
            "hop2_LLR_rank1": pack["hop2"]["LLR"]["rank1_rate"],
            "hop2_A_rank1": pack["hop2"]["A"]["rank1_rate"],
            "report": pack["_report_path"],
        }
    account_path = out_dir / "emb_llr_edge_validation_account.json"
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"帳本：{account_path}")
    return account


def _m5_gt_edge_keys(nodes, edges, gt_tids, by_tid):
    """GT 時間相鄰真轉移 ∩ 圖上邊。回傳 (gt_keys set of (i,j), same_sn, no_edge, missing)。"""
    tid_to_sn = {}
    for i, sn in enumerate(nodes):
        for tid in sn.tids:
            tid_to_sn[tid] = i
    gt_tracks = [by_tid[t] for t in gt_tids if t in by_tid]
    missing = [t for t in gt_tids if t not in by_tid]
    ordered = sorted(gt_tracks, key=lambda t: (t.t_start, t.t_end, t.tid))
    ej = {(e["_i"], e["_j"]): e for e in edges}
    gt_keys = set()
    same_sn = []
    no_edge = []
    for u, v in zip(ordered, ordered[1:]):
        iu, iv = tid_to_sn.get(u.tid), tid_to_sn.get(v.tid)
        if iu is None or iv is None:
            no_edge.append((u.tid, v.tid))
            continue
        if iu == iv:
            same_sn.append((u.tid, v.tid))
            continue
        if (iu, iv) not in ej:
            no_edge.append((u.tid, v.tid))
            continue
        gt_keys.add((iu, iv))
    return gt_keys, same_sn, no_edge, missing, ordered


def _m5_effect(st_gt, st_ng):
    if st_gt.get("mean") is None or st_ng.get("mean") is None:
        return None, None
    md = float(st_gt["mean"]) - float(st_ng["mean"])
    n1, n2 = int(st_gt["n"]), int(st_ng["n"])
    s1 = float(st_gt.get("std") or 0.0)
    s2 = float(st_ng.get("std") or 0.0)
    if n1 + n2 < 3:
        return md, None
    if n1 < 2:
        pooled = s2
    elif n2 < 2:
        pooled = s1
    else:
        pooled = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled == 0.0:
        return md, None
    return md, md / pooled


def _analyze_m5_edges(short, merge, gt_set, result_m5, result_m4=None):
    """邊層級：T / A / A+T vs M4(A+C−M)。"""
    nodes = result_m5["nodes"]
    tracks = result_m5["tracks"]
    by_tid = {t.tid: t for t in tracks}
    # 從 ranked/succ 重建邊：用 enumerate 再取 succ 太重；從 super_report 不夠
    # 直接重建 M5 圖
    attach_crop_embs(tracks, merge)
    fi = float(
        ((result_m5.get("super_report") or {}).get("m5_frame_interval"))
        or ((result_m5.get("options") or {}).get("stats") or {}).get("m5", {}).get(
            "frame_interval"
        )
        or (1.0 / 3.0)
    )
    # stats 在 result
    stats = (result_m5.get("options") or {}).get("score_stats") or result_m5.get(
        "stats"
    )
    if stats and stats.get("m5"):
        fi = float(stats["m5"].get("frame_interval") or fi)
    # re-get from super_report stored in result
    sr = result_m5.get("super_report") or {}
    if "m5_frame_interval" in sr:
        fi = float(sr["m5_frame_interval"])

    succ, _, n_legal, m5_meta = _build_succ_m5(nodes, fi)
    edges = []
    for i, items in enumerate(succ):
        for j, e in items:
            e2 = dict(e)
            e2["_i"] = i
            e2["_j"] = j
            edges.append(e2)

    gt_keys, same_sn, no_edge, missing, ordered = _m5_gt_edge_keys(
        nodes, edges, sorted(gt_set), by_tid
    )
    for e in edges:
        e["is_gt"] = (e["_i"], e["_j"]) in gt_keys
        e["AT"] = float(e.get("A", 0)) + float(e.get("T_score", 0))

    gt_e = [e for e in edges if e["is_gt"]]
    ng_e = [e for e in edges if not e["is_gt"]]

    def pack_score(key):
        st_gt = _score_stats([e.get(key) for e in gt_e])
        st_ng = _score_stats([e.get(key) for e in ng_e])
        md, d = _m5_effect(st_gt, st_ng)
        return {"st_gt": st_gt, "st_ng": st_ng, "mean_diff": md, "effect_d": d}

    scores = {
        "T_score": pack_score("T_score"),
        "A": pack_score("A"),
        "A+T": pack_score("AT"),
    }

    # M4 對照
    m4_cmp = None
    if result_m4 is not None:
        # 用已有 M4 結果的 succ？沒存。現場重建。
        attach_crop_embs(result_m4["tracks"], merge)
        nodes4 = result_m4["nodes"]
        succ4, _, n4, _ = _build_succ_m4(nodes4)
        edges4 = []
        for i, items in enumerate(succ4):
            for j, e in items:
                e2 = dict(e)
                e2["_i"] = i
                e2["_j"] = j
                # score 可能是 ±1e300
                sc = e2.get("score")
                if isinstance(sc, (int, float)) and abs(sc) >= 1e299:
                    e2["score"] = float("inf") if sc > 0 else float("-inf")
                edges4.append(e2)
        by4 = {t.tid: t for t in result_m4["tracks"]}
        gk4, _, _, _, _ = _m5_gt_edge_keys(nodes4, edges4, sorted(gt_set), by4)
        for e in edges4:
            e["is_gt"] = (e["_i"], e["_j"]) in gk4
        gt4 = [e for e in edges4 if e["is_gt"]]
        ng4 = [e for e in edges4 if not e["is_gt"]]
        st_gt = _score_stats([e.get("score") for e in gt4])
        st_ng = _score_stats([e.get("score") for e in ng4])
        md, d = _m5_effect(st_gt, st_ng)
        m4_cmp = {
            "n_edges": len(edges4),
            "n_gt": len(gt4),
            "n_nongt": len(ng4),
            "st_gt": st_gt,
            "st_ng": st_ng,
            "mean_diff": md,
            "effect_d": d,
        }

    # 個案
    worst_T = sorted(
        edges,
        key=lambda e: (
            _finite(e.get("T_score"))
            if _finite(e.get("T_score")) is not None
            else 0.0
        ),
    )[:5]
    steady = [
        e
        for e in edges
        if e.get("R") is not None
        and 0.8 <= float(e["R"]) <= 1.2
        and int(e.get("hop_count") or 0) >= 2
    ]
    steady.sort(key=lambda e: -float(e.get("dt") or 0))

    return {
        "short": short,
        "n_nodes": len(nodes),
        "n_edges": n_legal,
        "frame_interval": fi,
        "n_gt": len(gt_e),
        "n_nongt": len(ng_e),
        "same_sn": same_sn,
        "no_edge": no_edge,
        "missing": missing,
        "scores": scores,
        "m4_ACM": m4_cmp,
        "worst_T": worst_T,
        "steady_R": steady[:10],
        "edges": edges,
        "m5_meta": m5_meta,
    }


def cmd_compare_m5(argv=None):
    """M5（A+T）對照：邊層級 T/A/A+T vs M4；系統層 M0/M4/M5；0528 空窗。"""
    p = argparse.ArgumentParser(description="M5 A+T 對照實驗")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m5_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0507_top1.json",
            "m4": tp.OUTPUT_ROOT / "v1.0" / "m4_comparison" / "0507_M4_top1.json",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0528_top1.json",
            "m4": tp.OUTPUT_ROOT / "v1.0" / "m4_comparison" / "0528_M4_top1.json",
        },
    ]

    account = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "constants": [],
        "datasets": {},
    }
    results_by_short: dict = {}

    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])
        pack: dict = {"gt_set": sorted(gt_set)}

        m0 = json.loads(ds["m0_json"].read_text(encoding="utf-8"))
        pack["M0"] = _top_pack(m0, gt_set, {"source": str(ds["m0_json"])})

        # M4：引用既有或現場重跑
        print(f"\n===== {short} M4（對照） =====")
        if ds["m4"].is_file():
            # 仍需完整 result 做邊分析 → 現場跑 M4
            pass
        cfg4 = RunConfig(
            scoring="m4",
            node_score=False,
            dt_max=None,
            sim_min=float(args.sim_min),
            variant_tag="M4",
        )
        result4 = run_with_config(ds["merge"], cfg4)
        summary4 = _save_summary(result4, ds["merge"], out_dir, f"{short}_M4_top1")
        pack["M4"] = _top_pack(summary4, gt_set, {"constants": []})

        print(f"\n===== {short} M5 =====")
        cfg5 = RunConfig(
            scoring="m5",
            node_score=False,
            dt_max=None,
            sim_min=float(args.sim_min),
            variant_tag="M5",
        )
        result5 = run_with_config(ds["merge"], cfg5)
        summary5 = _save_summary(result5, ds["merge"], out_dir, f"{short}_M5_top1")
        pack["M5"] = _top_pack(summary5, gt_set, {"constants": []})

        # 診斷：最佳「至少 1 條邊」路徑（單節點 score=0 在 A+T≤0 時會壓過負分多邊）
        best_edged = None
        for h in result5["ranked"]:
            n_e = 0
            for seg in h.get("segments") or []:
                n_e += len(seg.get("edges") or [])
            if n_e == 0 and h.get("edges"):
                n_e = len(h["edges"])
            if n_e >= 1:
                best_edged = h
                break
        if best_edged is None:
            # fallback：scored maximal 裡找
            for p in result5.get("scored") or []:
                if p.get("edges"):
                    best_edged = p
                    break
        pack["M5_best_edged"] = (
            _top_pack(
                {
                    "tids": best_edged.get("tids"),
                    "path": best_edged.get("path")
                    or " -> ".join(
                        best_edged.get("super_labels") or best_edged.get("tids") or []
                    ),
                    "score": best_edged.get("score"),
                    "path_probability": best_edged.get("path_probability"),
                    "n_segments": best_edged.get("n_segments") or 1,
                },
                gt_set,
                {"note": "ranked 中第一條含 ≥1 邊者"},
            )
            if best_edged
            else {}
        )
        pack["M5_top1_n_edges"] = 0
        top5 = result5["ranked"][0] if result5["ranked"] else None
        if top5:
            ne = 0
            for seg in top5.get("segments") or []:
                ne += len(seg.get("edges") or [])
            if ne == 0:
                ne = len(top5.get("edges") or [])
            pack["M5_top1_n_edges"] = ne
            pack["M5_top1_is_singleton"] = ne == 0

        edge_pack = _analyze_m5_edges(
            short, ds["merge"], gt_set, result5, result_m4=result4
        )
        pack["edge"] = {
            k: v
            for k, v in edge_pack.items()
            if k not in ("edges", "worst_T", "steady_R")
        }
        # 序列化個案
        def _edge_brief(e):
            return {
                "from_super": e.get("from_super"),
                "to_super": e.get("to_super"),
                "from": e.get("from"),
                "to": e.get("to"),
                "dt": e.get("dt"),
                "hop": e.get("hop_count"),
                "A": e.get("A"),
                "T_score": e.get("T_score"),
                "R": e.get("R"),
                "T_bar": e.get("T_bar"),
                "T_u_raw": e.get("T_u_raw"),
                "T_v_raw": e.get("T_v_raw"),
                "is_gt": e.get("is_gt"),
                "both_degenerate": e.get("both_degenerate"),
            }

        pack["worst_T"] = [_edge_brief(e) for e in edge_pack["worst_T"]]
        pack["steady_R"] = [_edge_brief(e) for e in edge_pack["steady_R"]]

        # 0528 空窗
        gap_diag = None
        if short == "0528":
            m0_top = (m0.get("ranked") or m0.get("top") or [m0])[0] if isinstance(m0, dict) else {}
            # M0 分段：從 summary / ranked
            segs = []
            if isinstance(m0, dict):
                ranked0 = m0.get("ranked") or []
                if ranked0:
                    segs = ranked0[0].get("segments") or []
                elif m0.get("segments"):
                    segs = m0["segments"]
            gap_lo, gap_hi = 394.09, 572.38  # 既知 M0 空窗
            if len(segs) >= 2:
                s1, s2 = segs[0], segs[1]
                gap_lo = float(s1.get("t_end") or gap_lo)
                gap_hi = float(s2.get("t_start") or gap_hi)
            gap_w = gap_hi - gap_lo
            # 橋接邊：from.t_end <= gap_lo 且 to.t_start >= gap_hi（跨越空窗）
            nodes = result5["nodes"]
            bridge = []
            for e in edge_pack["edges"]:
                sa, sb = nodes[e["_i"]], nodes[e["_j"]]
                if float(sa.t_end) <= gap_lo + 1e-6 and float(sb.t_start) >= gap_hi - 1e-6:
                    bridge.append(_edge_brief(e))
            # 特別標 09_96 / 07_139
            fake = [
                e
                for e in edge_pack["edges"]
                if (
                    e.get("from") in ("K8-09_96", "K8-07_139")
                    or e.get("to") in ("K8-09_96", "K8-07_139")
                    or "K8-09_96" in (e.get("from_members") or [])
                    or "K8-07_139" in (e.get("to_members") or [])
                    or "K8-09_96" in (e.get("to_members") or [])
                    or "K8-07_139" in (e.get("from_members") or [])
                )
            ]
            # 空窗相關假橋邊
            focus = []
            for e in edge_pack["edges"]:
                fs, ts = e.get("from_super"), e.get("to_super")
                pair = (e.get("from"), e.get("to"))
                if pair in (
                    ("K8-07_93", "K8-09_96"),
                    ("K8-09_96", "K8-07_139"),
                    ("K8-07_139", "K8-09_142"),
                ) or (
                    "K8-09_96" in (e.get("from_members") or [])
                    and "K8-07_139" in (e.get("to_members") or [])
                ):
                    focus.append(_edge_brief(e))
            top5 = result5["ranked"][0] if result5["ranked"] else None
            has_fake = False
            if top5:
                tids = set(top5.get("tids") or [])
                has_fake = ("K8-09_96" in tids) or ("K8-07_139" in tids)
            gap_diag = {
                "gap_lo": gap_lo,
                "gap_hi": gap_hi,
                "gap_width": gap_w,
                "n_bridge_edges": len(bridge),
                "bridge_edges": sorted(
                    bridge, key=lambda x: (x.get("T_score") if x.get("T_score") is not None else 0)
                )[:15],
                "focus_fake_bridge": focus,
                "top1_has_09_96_or_07_139": has_fake,
                "top1_path": top5.get("path") if top5 else None,
            }
            pack["gap_178"] = gap_diag

        results_by_short[short] = {
            "result": result5,
            "result_m4": result4,
            "summary": summary5,
            "gt_set": gt_set,
            "merge": ds["merge"],
            "edge_pack": edge_pack,
            "gap_diag": gap_diag,
        }
        account["datasets"][short] = pack

    _render_m5_collages(results_by_short, out_dir)
    report_path = args.report.resolve() if args.report else (out_dir / "comparison_m5.md")
    root_report = tp.REPO_ROOT.parent / "comparison_m5.md"
    text = _render_m5_report(account, results_by_short)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    account_path = out_dir / "comparison_m5_account.json"
    # 精簡：去掉巨大 edge 列表
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n報告：{report_path}")
    print(f"副本：{root_report}")
    print(f"帳本：{account_path}")
    return account


def _render_m5_collages(results_by_short: dict, out_dir: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[m5_collage] Pillow 不可用")
        return

    def _font(size: int):
        for name in (
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ):
            if Path(name).is_file():
                try:
                    return ImageFont.truetype(name, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    FONT_XS = _font(10)

    def members_of_label(lab):
        if lab.startswith("{") and lab.endswith("}"):
            return [x.strip() for x in lab[1:-1].split(",") if x.strip()]
        return [lab]

    def thumb(path, size):
        if path is None or not Path(path).is_file():
            return Image.new("RGB", size, (230, 230, 230))
        im = Image.open(path).convert("RGB")
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (236, 236, 236))
        canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
        return canvas

    def _fmt(v):
        if isinstance(v, (int, float)) and abs(v) >= 1e299:
            return "+∞" if v > 0 else "−∞"
        if v is None:
            return "—"
        return f"{v:+.2f}"

    for short, tag_name in (("0507", "人員追蹤_20260507"), ("0528", "人員追蹤_20260528")):
        if short not in results_by_short:
            continue
        data = results_by_short[short]
        result = data["result"]
        gt_set = data["gt_set"]
        merge = data["merge"]
        if not result["ranked"]:
            continue
        top = result["ranked"][0]
        by_tid = {t.tid: t for t in result["tracks"]}
        segs = sorted(top.get("segments") or [], key=lambda s: int(s.get("segment") or 1))
        if not segs:
            segs = [{
                "segment": 1,
                "super_labels": top.get("super_labels") or [],
                "edges": top.get("edges") or [],
                "gap_after_prev_sec": None,
            }]

        tw, th = 100, 130
        margin, title_h, row_gap, cell_gap = 12, 80, 20, 72
        max_cols = 8
        rows = []
        for si, seg in enumerate(segs):
            if si > 0:
                g = seg.get("gap_after_prev_sec")
                g = 0.0 if g is None else float(g)
                rows.append(("gap", f"觀測空窗 {g:.1f} 秒"))
            labels = seg.get("super_labels") or []
            edges = seg.get("edges") or []
            if not edges and int(seg.get("segment") or 1) == 1 and int(top.get("n_segments") or 1) == 1:
                edges = top.get("edges") or []
            nodes_list = []
            for i, lab in enumerate(labels):
                mems = members_of_label(lab)
                if i == 0 and edges and edges[0].get("from_members"):
                    mems = list(edges[0]["from_members"])
                elif i > 0 and i - 1 < len(edges) and edges[i - 1].get("to_members"):
                    mems = list(edges[i - 1]["to_members"])
                e = edges[i] if i < len(edges) else None
                nodes_list.append((mems, e))
            for st in range(0, len(nodes_list), max_cols):
                rows.append(("nodes", nodes_list[st: st + max_cols]))

        rh, rw = [], []
        for row in rows:
            if row[0] == "gap":
                rh.append(30)
                rw.append(900)
                continue
            hmax = th + 72
            wsum = 0
            for mems, e in row[1]:
                cw = max(140, len(mems) * (tw + 8) + 12)
                wsum += cw + (cell_gap if e is not None else 0)
            rh.append(hmax)
            rw.append(wsum)

        W = margin * 2 + max(rw + [800])
        H = title_h + sum(rh) + row_gap * max(0, len(rh) - 1) + margin
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        pr = precision_recall_vs_gt(top.get("tids") or [], gt_set)
        for i, line in enumerate([
            f"M5（A+T）{short} Top-1  n_seg={top.get('n_segments')}  "
            f"score={top.get('score', 0):.4f}",
            f"prec={pr['precision']:.4f} ({pr['n_hit']}/{pr['n_path']})  "
            f"rec={pr['recall']:.4f} ({pr['n_hit']}/{pr['n_gt']})  "
            f"P={top.get('path_probability', 0):.6f}",
            "邊分 = A + T；T=−max(0,ln R)；R=dt/(hop×T̄)  GT 僅評估著色",
        ]):
            draw.text((margin, 6 + i * 14), line, fill=(10, 10, 10), font=FONT_XS)

        y = title_h
        for row in rows:
            if row[0] == "gap":
                draw.rectangle(
                    [margin, y, W - margin, y + 26],
                    fill=(255, 245, 230),
                    outline=(200, 120, 40),
                )
                draw.text((margin + 6, y + 6), row[1], fill=(160, 80, 0), font=FONT_XS)
                y += 30 + row_gap
                continue
            x = margin
            hmax = th + 72
            for mems, e in row[1]:
                cw = max(140, len(mems) * (tw + 8) + 12)
                bc = (0, 160, 0) if all(m in gt_set for m in mems) else (200, 40, 40)
                draw.rectangle([x, y, x + cw - 1, y + hmax - 1], outline=bc, width=3)
                for i, tid in enumerate(mems):
                    t = by_tid.get(tid)
                    cam, tid_s = tid.rsplit("_", 1)
                    try:
                        _, crops = tp._crop_paths_for_track(merge, cam, int(tid_s))
                        rep = tp._pick_rep_crop(crops)
                    except Exception:
                        rep = None
                    sx = x + 4 + i * (tw + 8)
                    sy = y + 16
                    img.paste(thumb(rep, (tw, th)), (sx, sy))
                    mbc = (0, 160, 0) if tid in gt_set else (200, 40, 40)
                    draw.rectangle(
                        [sx, sy, sx + tw - 1, sy + th - 1], outline=mbc, width=2
                    )
                    if t:
                        txt = f"{tid}\n{cam}\n[{t.t_start:.1f}-{t.t_end:.1f}]"
                    else:
                        txt = tid
                    ly = sy + th + 2
                    for line in txt.split("\n"):
                        draw.text((sx, ly), line, fill=(20, 20, 20), font=FONT_XS)
                        ly += 11
                x += cw
                if e is not None:
                    mid = y + hmax // 2
                    draw.line(
                        [(x + 4, mid), (x + cell_gap - 8, mid)],
                        fill=(40, 40, 40),
                        width=2,
                    )
                    A = e.get("A", 0)
                    T = e.get("T_score", 0)
                    sc = e.get("score", 0)
                    R = e.get("R")
                    sc_col = (
                        (0, 140, 0)
                        if (isinstance(sc, (int, float)) and sc >= 0)
                        else (200, 0, 0)
                    )
                    rtxt = f"R={R:.2f}" if isinstance(R, (int, float)) else "R=—"
                    draw.multiline_text(
                        (x + 2, mid - 28),
                        f"A={_fmt(A)}\nT={_fmt(T)}\n{rtxt}",
                        fill=(40, 40, 140),
                        font=FONT_XS,
                    )
                    draw.text(
                        (x + 2, mid + 18), f"Σ={_fmt(sc)}", fill=sc_col, font=FONT_XS
                    )
                    x += cell_gap
            y += hmax + row_gap

        out_png = out_dir / f"{tag_name}_m5_top1_collage.png"
        img.save(out_png)
        print(f"拼圖：{out_png}")


def _render_m5_report(account: dict, results_by_short: dict) -> str:
    lines = [
        "# M5（A + T 自時鐘）對照實驗",
        "",
        f"生成時間：{account.get('generated_at')}",
        "",
        "> **GT 僅用於評估與著色，不參與計分。**",
        "",
        "邊分 = A + T；T = −max(0, ln R)；R = dt / (hop_count × T̄)；"
        "建邊僅時間順序；不用 C、不用 M。",
        "",
        "## 1. 系統層級總表：M0 / M4 / M5",
        "",
        "| 資料集 | 版 | precision | recall | P | n_seg | n_path | n_hit |",
        "|--------|----|-----------:|-------:|--:|------:|-------:|------:|",
    ]
    for short, pack in account["datasets"].items():
        for key in ("M0", "M4", "M5"):
            d = pack.get(key) or {}
            if not d:
                lines.append(f"| {short} | {key} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {short} | {key} | {_fmt_pct(d.get('precision', 0))} | "
                f"{_fmt_pct(d.get('recall', 0))} | {float(d.get('P') or 0):.6f} | "
                f"{d.get('n_segments')} | {d.get('n_path')} | {d.get('n_hit')} |"
            )
        be = pack.get("M5_best_edged") or {}
        if be:
            mark = ""
            if pack.get("M5_top1_is_singleton"):
                mark = "（Top-1 為 0 邊單節點；此列=第一條有邊假設）"
            lines.append(
                f"| {short} | M5≥1邊{mark} | {_fmt_pct(be.get('precision', 0))} | "
                f"{_fmt_pct(be.get('recall', 0))} | {float(be.get('P') or 0):.6f} | "
                f"{be.get('n_segments')} | {be.get('n_path')} | {be.get('n_hit')} |"
            )
    lines.append("")
    lines.append(
        "> 註：T_score = −max(0,ln R) **恆 ≤ 0**；當多數邊 A+T < 0 時，"
        "score=0 的終端單節點會贏過有邊路徑（非調參，是計分符號後果）。"
    )
    lines.append("")

    # 邊層級
    lines += ["## 2. 邊層級：T / A / A+T vs M4(A+C−M)", ""]
    for short, pack in account["datasets"].items():
        ep = pack.get("edge") or {}
        lines.append(f"### {short}")
        lines.append("")
        lines.append(
            f"- 超節點：{ep.get('n_nodes')}；時間序邊：{ep.get('n_edges')}；"
            f"GT 邊：{ep.get('n_gt')}；非 GT：{ep.get('n_nongt')}；"
            f"幀間隔：{ep.get('frame_interval')}"
        )
        lines.append("")
        lines.append(
            "| 分數 | GT mean | 非GT mean | mean差 | 效應量 d | GT>0 | 非GT>0 |"
        )
        lines.append(
            "|------|--------:|----------:|-------:|--------:|-----:|-------:|"
        )
        scores = ep.get("scores") or {}
        for key in ("T_score", "A", "A+T"):
            s = scores.get(key) or {}
            stg, stn = s.get("st_gt") or {}, s.get("st_ng") or {}
            prg = (
                f"{stg.get('pos_rate'):.1%}"
                if stg.get("pos_rate") is not None
                else "—"
            )
            prn = (
                f"{stn.get('pos_rate'):.1%}"
                if stn.get("pos_rate") is not None
                else "—"
            )
            lines.append(
                f"| {key} | {_fmt_stat(stg.get('mean'))} | {_fmt_stat(stn.get('mean'))} | "
                f"{_fmt_stat(s.get('mean_diff'))} | {_fmt_stat(s.get('effect_d'))} | "
                f"{prg} | {prn} |"
            )
        m4 = ep.get("m4_ACM")
        if m4:
            lines.append(
                f"| M4 A+C−M | {_fmt_stat((m4.get('st_gt') or {}).get('mean'))} | "
                f"{_fmt_stat((m4.get('st_ng') or {}).get('mean'))} | "
                f"{_fmt_stat(m4.get('mean_diff'))} | {_fmt_stat(m4.get('effect_d'))} | "
                f"{(m4.get('st_gt') or {}).get('pos_rate', 0):.1%} | "
                f"{(m4.get('st_ng') or {}).get('pos_rate', 0):.1%} |"
            )
        lines.append("")
        # T 分布細節
        st_t_gt = (scores.get("T_score") or {}).get("st_gt") or {}
        st_t_ng = (scores.get("T_score") or {}).get("st_ng") or {}
        def _neg_pct(st):
            r = st.get("neg_rate")
            return f"{r:.1%}" if r is not None else "—"

        lines.append(
            f"- T_score GT：mean={_fmt_stat(st_t_gt.get('mean'))} "
            f"median={_fmt_stat(st_t_gt.get('median'))} "
            f">0={st_t_gt.get('pos_rate')}；<0（有罰）={_neg_pct(st_t_gt)}"
        )
        lines.append(
            f"- T_score 非GT：mean={_fmt_stat(st_t_ng.get('mean'))} "
            f"median={_fmt_stat(st_t_ng.get('median'))} "
            f">0={st_t_ng.get('pos_rate')}；<0（有罰）={_neg_pct(st_t_ng)}"
        )
        # 誰更會分
        d_t = (scores.get("T_score") or {}).get("effect_d")
        d_a = (scores.get("A") or {}).get("effect_d")
        d_at = (scores.get("A+T") or {}).get("effect_d")
        d_m4 = m4.get("effect_d") if m4 else None
        lines.append(
            f"- 效應量對照：T={_fmt_stat(d_t)}，A={_fmt_stat(d_a)}，"
            f"A+T={_fmt_stat(d_at)}，M4(A+C−M)={_fmt_stat(d_m4)}"
        )
        lines.append("")

    # 0528 空窗
    lines += ["## 3. 0528 空窗（~178s）與假橋", ""]
    g = (account["datasets"].get("0528") or {}).get("gap_178") or {}
    if not g:
        lines.append("（無）")
    else:
        lines.append(
            f"- 空窗區間：**[{g.get('gap_lo'):.2f}, {g.get('gap_hi'):.2f}]** "
            f"寬度 **{g.get('gap_width'):.2f}s**"
        )
        lines.append(
            f"- Top-1 是否含 `K8-09_96` / `K8-07_139`："
            f"**{'是 ★' if g.get('top1_has_09_96_or_07_139') else '否'}**"
        )
        lines.append(f"- Top-1 路徑：`{g.get('top1_path')}`")
        lines.append("")
        lines.append("### 假橋焦點邊（R / T_score）")
        lines.append("")
        lines.append(
            "| from→to | dt | hop | T̄ | R | T_score | A | Σ |"
        )
        lines.append(
            "|---------|---:|----:|---:|--:|--------:|--:|--:|"
        )
        for e in g.get("focus_fake_bridge") or []:
            lines.append(
                f"| `{e.get('from')}`→`{e.get('to')}` | {e.get('dt', 0):.1f} | "
                f"{e.get('hop')} | {_fmt_stat(e.get('T_bar'))} | "
                f"{_fmt_stat(e.get('R'), 3)} | {_fmt_stat(e.get('T_score'))} | "
                f"{_fmt_stat(e.get('A'))} | "
                f"{_fmt_stat((e.get('A') or 0) + (e.get('T_score') or 0))} |"
            )
        lines.append("")
        lines.append(
            f"跨越空窗的橋接邊數：{g.get('n_bridge_edges')}（列 T 最重罰前數條）："
        )
        lines.append("")
        for e in (g.get("bridge_edges") or [])[:8]:
            lines.append(
                f"- `{e.get('from_super')}`→`{e.get('to_super')}` "
                f"dt={e.get('dt'):.1f} hop={e.get('hop')} "
                f"R={_fmt_stat(e.get('R'), 3)} T={_fmt_stat(e.get('T_score'))} "
                f"A={_fmt_stat(e.get('A'))} GT={e.get('is_gt')}"
            )
        lines.append("")

    # 個案
    lines += ["## 4. 個案", ""]
    for short, pack in account["datasets"].items():
        lines.append(f"### {short}")
        lines.append("")
        lines.append("#### T_score 最重罰的 5 條邊")
        lines.append("")
        lines.append("| from→to | dt | hop | T̄ | R | T_score | A | GT? |")
        lines.append("|---------|---:|----:|---:|--:|--------:|--:|:---:|")
        for e in pack.get("worst_T") or []:
            lines.append(
                f"| `{e.get('from_super')}`→`{e.get('to_super')}` | "
                f"{e.get('dt', 0):.1f} | {e.get('hop')} | "
                f"{_fmt_stat(e.get('T_bar'))} | {_fmt_stat(e.get('R'), 3)} | "
                f"{_fmt_stat(e.get('T_score'))} | {_fmt_stat(e.get('A'))} | "
                f"{'✓' if e.get('is_gt') else ''} |"
            )
        lines.append("")
        lines.append("#### R∈[0.8,1.2] 且 hop≥2 的長途邊（步調一致清白案例）")
        lines.append("")
        steady = pack.get("steady_R") or []
        if not steady:
            lines.append("（無）")
        else:
            lines.append("| from→to | dt | hop | T̄ | R | T_score | A | GT? |")
            lines.append("|---------|---:|----:|---:|--:|--------:|--:|:---:|")
            for e in steady:
                lines.append(
                    f"| `{e.get('from_super')}`→`{e.get('to_super')}` | "
                    f"{e.get('dt', 0):.1f} | {e.get('hop')} | "
                    f"{_fmt_stat(e.get('T_bar'))} | {_fmt_stat(e.get('R'), 3)} | "
                    f"{_fmt_stat(e.get('T_score'))} | {_fmt_stat(e.get('A'))} | "
                    f"{'✓' if e.get('is_gt') else ''} |"
                )
        lines.append("")

    lines += [
        "## 5. 常數清單",
        "",
        "**`[]`（空）** — 幀間隔由 foots 時間差中位數估計（資料觀測，非手調門檻）；"
        "T 公式無額外常數。",
        "",
        "## 6. 拼圖",
        "",
        "- `output/v1.0/m5_comparison/人員追蹤_20260507_m5_top1_collage.png`",
        "- `output/v1.0/m5_comparison/人員追蹤_20260528_m5_top1_collage.png`",
        "",
        "---",
        "",
        "實驗約束：GT 僅評估；未改既有 M1–M4b；未改 track_path.py；未調參。",
        "",
    ]
    return "\n".join(lines)


def _analyze_m6_edges(short, merge, gt_set, result_m6, result_m4=None):
    nodes = result_m6["nodes"]
    tracks = result_m6["tracks"]
    by_tid = {t.tid: t for t in tracks}
    attach_crop_embs(tracks, merge)
    succ, _, n_legal, m6_meta = _build_succ_m6(nodes)
    edges = []
    for i, items in enumerate(succ):
        for j, e in items:
            e2 = dict(e)
            e2["_i"] = i
            e2["_j"] = j
            # A+C+S 以邊 score 為準（已含 inf 編碼）
            e2["ACS"] = e.get("score")
            edges.append(e2)

    gt_keys, same_sn, no_edge, missing, ordered = _m5_gt_edge_keys(
        nodes, edges, sorted(gt_set), by_tid
    )
    for e in edges:
        e["is_gt"] = (e["_i"], e["_j"]) in gt_keys

    gt_e = [e for e in edges if e["is_gt"]]
    ng_e = [e for e in edges if not e["is_gt"]]

    def pack_score(key):
        st_gt = _score_stats([e.get(key) for e in gt_e])
        st_ng = _score_stats([e.get(key) for e in ng_e])
        md, d = _m5_effect(st_gt, st_ng)
        return {"st_gt": st_gt, "st_ng": st_ng, "mean_diff": md, "effect_d": d}

    scores = {
        "S": pack_score("S"),
        "A": pack_score("A"),
        "C": pack_score("C"),
        "A+C+S": pack_score("ACS"),
    }

    m4_cmp = None
    if result_m4 is not None:
        attach_crop_embs(result_m4["tracks"], merge)
        nodes4 = result_m4["nodes"]
        succ4, _, _, _ = _build_succ_m4(nodes4)
        edges4 = []
        for i, items in enumerate(succ4):
            for j, e in items:
                e2 = dict(e)
                e2["_i"] = i
                e2["_j"] = j
                edges4.append(e2)
        by4 = {t.tid: t for t in result_m4["tracks"]}
        gk4, _, _, _, _ = _m5_gt_edge_keys(nodes4, edges4, sorted(gt_set), by4)
        for e in edges4:
            e["is_gt"] = (e["_i"], e["_j"]) in gk4
        gt4 = [e for e in edges4 if e["is_gt"]]
        ng4 = [e for e in edges4 if not e["is_gt"]]
        st_gt = _score_stats([e.get("score") for e in gt4])
        st_ng = _score_stats([e.get("score") for e in ng4])
        md, d = _m5_effect(st_gt, st_ng)
        m4_cmp = {
            "n_edges": len(edges4),
            "n_gt": len(gt4),
            "n_nongt": len(ng4),
            "st_gt": st_gt,
            "st_ng": st_ng,
            "mean_diff": md,
            "effect_d": d,
        }

    # 教授情境：跳過>=2 但 S 罰款 <0.2（−S < 0.2 ⇒ S > −0.2）
    soft_skip_gt = []
    for e in gt_e:
        nsk = int(e.get("n_skipped") or 0)
        S = _finite(e.get("S"))
        if nsk >= 2 and S is not None and (-S) < 0.2:
            soft_skip_gt.append(e)

    return {
        "short": short,
        "n_nodes": len(nodes),
        "n_edges": n_legal,
        "n_degenerate": m6_meta.get("n_degenerate"),
        "n_gt": len(gt_e),
        "n_nongt": len(ng_e),
        "same_sn": same_sn,
        "no_edge": no_edge,
        "missing": missing,
        "scores": scores,
        "m4_ACM": m4_cmp,
        "soft_skip_gt": soft_skip_gt,
        "gt_edges": gt_e,
        "edges": edges,
        "m6_meta": m6_meta,
    }


def cmd_compare_m6(argv=None):
    """M6（A+C+S）對照驗證。"""
    p = argparse.ArgumentParser(description="M6 A+C+S 對照實驗")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m6_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0507_top1.json",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0528_top1.json",
        },
    ]

    account = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "constants": [],
        "datasets": {},
    }
    results_by_short = {}

    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])
        pack: dict = {"gt_set": sorted(gt_set)}

        m0 = json.loads(ds["m0_json"].read_text(encoding="utf-8"))
        pack["M0"] = _top_pack(m0, gt_set, {"source": str(ds["m0_json"])})

        print(f"\n===== {short} M4（對照） =====")
        result4 = run_with_config(
            ds["merge"],
            RunConfig(
                scoring="m4",
                node_score=False,
                dt_max=None,
                sim_min=float(args.sim_min),
                variant_tag="M4",
            ),
        )
        summary4 = _save_summary(result4, ds["merge"], out_dir, f"{short}_M4_top1")
        pack["M4"] = _top_pack(summary4, gt_set, {"constants": []})

        print(f"\n===== {short} M6 =====")
        result6 = run_with_config(
            ds["merge"],
            RunConfig(
                scoring="m6",
                node_score=False,
                dt_max=None,
                sim_min=float(args.sim_min),
                variant_tag="M6",
            ),
        )
        summary6 = _save_summary(result6, ds["merge"], out_dir, f"{short}_M6_top1")
        pack["M6"] = _top_pack(summary6, gt_set, {"constants": []})

        # 單節點壓制？
        top6 = result6["ranked"][0] if result6["ranked"] else None
        n_edges_top = 0
        if top6:
            for seg in top6.get("segments") or []:
                n_edges_top += len(seg.get("edges") or [])
            if n_edges_top == 0:
                n_edges_top = len(top6.get("edges") or [])
        pack["M6_top1_n_edges"] = n_edges_top
        pack["M6_top1_is_singleton"] = n_edges_top == 0
        pack["M6_top1_min_A"] = top6.get("min_A") if top6 else None
        pack["M6_top1_path"] = (
            top6.get("path")
            or " -> ".join(top6.get("super_labels") or top6.get("tids") or [])
            if top6
            else None
        )
        pack["M6_top1_score"] = top6.get("score") if top6 else None
        pack["M6_top1_P"] = top6.get("path_probability") if top6 else None

        # 次名 min-A
        runner = result6["ranked"][1] if len(result6["ranked"]) > 1 else None
        pack["M6_runner"] = {
            "path": (
                runner.get("path")
                or " -> ".join(
                    runner.get("super_labels") or runner.get("tids") or []
                )
                if runner
                else None
            ),
            "score": runner.get("score") if runner else None,
            "P": runner.get("path_probability") if runner else None,
            "min_A": runner.get("min_A") if runner else None,
            "n_segments": runner.get("n_segments") if runner else None,
        }

        # 最佳有邊
        best_edged = None
        for h in result6["ranked"]:
            ne = 0
            for seg in h.get("segments") or []:
                ne += len(seg.get("edges") or [])
            if ne == 0:
                ne = len(h.get("edges") or [])
            if ne >= 1:
                best_edged = h
                break
        pack["M6_best_edged"] = (
            {
                "path": best_edged.get("path")
                or " -> ".join(
                    best_edged.get("super_labels") or best_edged.get("tids") or []
                ),
                "score": best_edged.get("score"),
                "P": best_edged.get("path_probability"),
                "min_A": best_edged.get("min_A"),
                **precision_recall_vs_gt(best_edged.get("tids") or [], gt_set),
            }
            if best_edged
            else {}
        )

        edge_pack = _analyze_m6_edges(
            short, ds["merge"], gt_set, result6, result_m4=result4
        )
        pack["edge"] = {
            k: v
            for k, v in edge_pack.items()
            if k not in ("edges", "gt_edges", "soft_skip_gt")
        }

        def _eb(e):
            return {
                "from_super": e.get("from_super"),
                "to_super": e.get("to_super"),
                "from": e.get("from"),
                "to": e.get("to"),
                "dt": e.get("dt"),
                "A": e.get("A"),
                "C": e.get("C"),
                "S": e.get("S"),
                "score": e.get("score"),
                "n_skipped": e.get("n_skipped"),
                "sum_P_skipped": e.get("sum_P_skipped"),
                "P_fwd": e.get("P_fwd"),
                "is_gt": e.get("is_gt"),
                "skipped_top": (e.get("skipped") or [])[:8],
            }

        pack["soft_skip_gt"] = [_eb(e) for e in edge_pack["soft_skip_gt"]]
        pack["gt_edges_brief"] = [_eb(e) for e in edge_pack["gt_edges"]]

        # 0528 假橋
        if short == "0528":
            focus_pairs = {
                ("K8-07_93", "K8-09_96"),
                ("K8-09_96", "K8-07_139"),
                ("K8-07_139", "K8-09_142"),
            }
            focus = []
            for e in edge_pack["edges"]:
                pair = (e.get("from"), e.get("to"))
                if pair in focus_pairs or (
                    "K8-09_96" in (e.get("from_members") or [])
                    and "K8-07_139" in (e.get("to_members") or [])
                ):
                    focus.append(_eb(e))
            tids = set(top6.get("tids") or []) if top6 else set()
            pack["fake_bridge"] = {
                "focus": focus,
                "top1_has_09_96_or_07_139": ("K8-09_96" in tids)
                or ("K8-07_139" in tids),
                "top1_path": pack["M6_top1_path"],
            }

        results_by_short[short] = {
            "result": result6,
            "result_m4": result4,
            "gt_set": gt_set,
            "merge": ds["merge"],
            "edge_pack": edge_pack,
        }
        account["datasets"][short] = pack

    _render_m6_collages(results_by_short, out_dir)
    report_path = args.report.resolve() if args.report else (out_dir / "comparison_m6.md")
    root_report = tp.REPO_ROOT.parent / "comparison_m6.md"
    text = _render_m6_report(account)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    account_path = out_dir / "comparison_m6_account.json"
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n報告：{report_path}")
    print(f"副本：{root_report}")
    print(f"帳本：{account_path}")
    return account


def _render_m6_collages(results_by_short: dict, out_dir: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[m6_collage] Pillow 不可用")
        return

    def _font(size: int):
        for name in (
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ):
            if Path(name).is_file():
                try:
                    return ImageFont.truetype(name, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    FONT_XS = _font(10)

    def members_of_label(lab):
        if lab.startswith("{") and lab.endswith("}"):
            return [x.strip() for x in lab[1:-1].split(",") if x.strip()]
        return [lab]

    def thumb(path, size):
        if path is None or not Path(path).is_file():
            return Image.new("RGB", size, (230, 230, 230))
        im = Image.open(path).convert("RGB")
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (236, 236, 236))
        canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
        return canvas

    def _fmt(v):
        if isinstance(v, (int, float)) and abs(v) >= 1e299:
            return "+∞" if v > 0 else "−∞"
        if v is None:
            return "—"
        return f"{v:+.2f}"

    for short, tag_name in (("0507", "人員追蹤_20260507"), ("0528", "人員追蹤_20260528")):
        if short not in results_by_short:
            continue
        data = results_by_short[short]
        result = data["result"]
        gt_set = data["gt_set"]
        merge = data["merge"]
        if not result["ranked"]:
            continue
        top = result["ranked"][0]
        by_tid = {t.tid: t for t in result["tracks"]}
        segs = sorted(top.get("segments") or [], key=lambda s: int(s.get("segment") or 1))
        if not segs:
            segs = [{
                "segment": 1,
                "super_labels": top.get("super_labels") or [],
                "edges": top.get("edges") or [],
                "gap_after_prev_sec": None,
            }]

        tw, th = 100, 130
        margin, title_h, row_gap, cell_gap = 12, 80, 20, 80
        max_cols = 8
        rows = []
        for si, seg in enumerate(segs):
            if si > 0:
                g = seg.get("gap_after_prev_sec")
                g = 0.0 if g is None else float(g)
                rows.append(("gap", f"觀測空窗 {g:.1f} 秒"))
            labels = seg.get("super_labels") or []
            edges = seg.get("edges") or []
            if (
                not edges
                and int(seg.get("segment") or 1) == 1
                and int(top.get("n_segments") or 1) == 1
            ):
                edges = top.get("edges") or []
            nodes_list = []
            for i, lab in enumerate(labels):
                mems = members_of_label(lab)
                if i == 0 and edges and edges[0].get("from_members"):
                    mems = list(edges[0]["from_members"])
                elif i > 0 and i - 1 < len(edges) and edges[i - 1].get("to_members"):
                    mems = list(edges[i - 1]["to_members"])
                e = edges[i] if i < len(edges) else None
                nodes_list.append((mems, e))
            for st in range(0, len(nodes_list), max_cols):
                rows.append(("nodes", nodes_list[st : st + max_cols]))

        rh, rw = [], []
        for row in rows:
            if row[0] == "gap":
                rh.append(30)
                rw.append(900)
                continue
            hmax = th + 72
            wsum = 0
            for mems, e in row[1]:
                cw = max(140, len(mems) * (tw + 8) + 12)
                wsum += cw + (cell_gap if e is not None else 0)
            rh.append(hmax)
            rw.append(wsum)

        W = margin * 2 + max(rw + [800])
        H = title_h + sum(rh) + row_gap * max(0, len(rh) - 1) + margin
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        pr = precision_recall_vs_gt(top.get("tids") or [], gt_set)
        for i, line in enumerate(
            [
                f"M6（A+C+S）{short} Top-1  n_seg={top.get('n_segments')}  "
                f"score={top.get('score', 0):.4f}  minA={_fmt(top.get('min_A'))}",
                f"prec={pr['precision']:.4f} ({pr['n_hit']}/{pr['n_path']})  "
                f"rec={pr['recall']:.4f} ({pr['n_hit']}/{pr['n_gt']})  "
                f"P={top.get('path_probability', 0):.6f}",
                "邊分=A+C+S；S=ln(1−ΣP_earlier)；hop不計分  GT僅評估",
            ]
        ):
            draw.text((margin, 6 + i * 14), line, fill=(10, 10, 10), font=FONT_XS)

        y = title_h
        for row in rows:
            if row[0] == "gap":
                draw.rectangle(
                    [margin, y, W - margin, y + 26],
                    fill=(255, 245, 230),
                    outline=(200, 120, 40),
                )
                draw.text((margin + 6, y + 6), row[1], fill=(160, 80, 0), font=FONT_XS)
                y += 30 + row_gap
                continue
            x = margin
            hmax = th + 72
            for mems, e in row[1]:
                cw = max(140, len(mems) * (tw + 8) + 12)
                bc = (0, 160, 0) if all(m in gt_set for m in mems) else (200, 40, 40)
                draw.rectangle([x, y, x + cw - 1, y + hmax - 1], outline=bc, width=3)
                for i, tid in enumerate(mems):
                    t = by_tid.get(tid)
                    cam, tid_s = tid.rsplit("_", 1)
                    try:
                        _, crops = tp._crop_paths_for_track(merge, cam, int(tid_s))
                        rep = tp._pick_rep_crop(crops)
                    except Exception:
                        rep = None
                    sx = x + 4 + i * (tw + 8)
                    sy = y + 16
                    img.paste(thumb(rep, (tw, th)), (sx, sy))
                    mbc = (0, 160, 0) if tid in gt_set else (200, 40, 40)
                    draw.rectangle(
                        [sx, sy, sx + tw - 1, sy + th - 1], outline=mbc, width=2
                    )
                    txt = f"{tid}\n{cam}" if t else tid
                    ly = sy + th + 2
                    for line in txt.split("\n"):
                        draw.text((sx, ly), line, fill=(20, 20, 20), font=FONT_XS)
                        ly += 11
                x += cw
                if e is not None:
                    mid = y + hmax // 2
                    draw.line(
                        [(x + 4, mid), (x + cell_gap - 8, mid)],
                        fill=(40, 40, 40),
                        width=2,
                    )
                    sc = e.get("score", 0)
                    sc_col = (
                        (0, 140, 0)
                        if (isinstance(sc, (int, float)) and sc >= 0)
                        else (200, 0, 0)
                    )
                    draw.multiline_text(
                        (x + 2, mid - 32),
                        f"A={_fmt(e.get('A'))}\nC={_fmt(e.get('C'))}\n"
                        f"S={_fmt(e.get('S'))}",
                        fill=(40, 40, 140),
                        font=FONT_XS,
                    )
                    draw.text(
                        (x + 2, mid + 18), f"Σ={_fmt(sc)}", fill=sc_col, font=FONT_XS
                    )
                    x += cell_gap
            y += hmax + row_gap

        out_png = out_dir / f"{tag_name}_m6_top1_collage.png"
        img.save(out_png)
        print(f"拼圖：{out_png}")


def _m6_collage_helpers():
    """共用：字型 / 縮圖 / 格式。"""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    def _font(size: int):
        for name in (
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ):
            if Path(name).is_file():
                try:
                    return ImageFont.truetype(name, size)
                except OSError:
                    continue
        return ImageFont.load_default()

    return Image, ImageDraw, ImageFont, _font(10)


def _render_one_m6_collage(
    *,
    hyp: dict,
    rank: int,
    short: str,
    dataset_tag: str,
    gt_set: set,
    merge: Path,
    by_tid: dict,
    out_png: Path,
    mode_label: str = "M6（A+C+S）",
    app_key: str = "A",
    app_tag: str = "A",
) -> None:
    """單張 Top-k 拼圖（規格：crop+tid/鏡頭/時間、GT綠/非GT紅、邊標 app/C/S/Σ）。"""
    helpers = _m6_collage_helpers()
    if helpers is None:
        print(f"[m6_collage] Pillow 不可用，跳過 {out_png.name}")
        return
    Image, ImageDraw, ImageFont, FONT_XS = helpers

    def members_of_label(lab):
        if lab.startswith("{") and lab.endswith("}"):
            return [x.strip() for x in lab[1:-1].split(",") if x.strip()]
        return [lab]

    def thumb(path, size):
        if path is None or not Path(path).is_file():
            return Image.new("RGB", size, (230, 230, 230))
        im = Image.open(path).convert("RGB")
        im.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (236, 236, 236))
        canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
        return canvas

    def _fmt(v):
        if isinstance(v, (int, float)) and abs(v) >= 1e299:
            return "+∞" if v > 0 else "−∞"
        if v is None:
            return "—"
        return f"{v:+.2f}"

    segs = sorted(hyp.get("segments") or [], key=lambda s: int(s.get("segment") or 1))
    if not segs:
        segs = [
            {
                "segment": 1,
                "super_labels": hyp.get("super_labels") or [],
                "edges": hyp.get("edges") or [],
                "gap_after_prev_sec": None,
            }
        ]

    tw, th = 100, 130
    margin, title_h, row_gap, cell_gap = 12, 80, 20, 80
    max_cols = 8
    rows = []
    for si, seg in enumerate(segs):
        if si > 0:
            g = seg.get("gap_after_prev_sec")
            g = 0.0 if g is None else float(g)
            rows.append(("gap", f"觀測空窗 {g:.1f} 秒"))
        labels = seg.get("super_labels") or []
        edges = seg.get("edges") or []
        if (
            not edges
            and int(seg.get("segment") or 1) == 1
            and int(hyp.get("n_segments") or 1) == 1
        ):
            edges = hyp.get("edges") or []
        nodes_list = []
        for i, lab in enumerate(labels):
            mems = members_of_label(lab)
            if i == 0 and edges and edges[0].get("from_members"):
                mems = list(edges[0]["from_members"])
            elif i > 0 and i - 1 < len(edges) and edges[i - 1].get("to_members"):
                mems = list(edges[i - 1]["to_members"])
            e = edges[i] if i < len(edges) else None
            nodes_list.append((mems, e))
        for st in range(0, len(nodes_list), max_cols):
            rows.append(("nodes", nodes_list[st : st + max_cols]))

    rh, rw = [], []
    for row in rows:
        if row[0] == "gap":
            rh.append(30)
            rw.append(900)
            continue
        hmax = th + 84
        wsum = 0
        for mems, e in row[1]:
            cw = max(140, len(mems) * (tw + 8) + 12)
            wsum += cw + (cell_gap if e is not None else 0)
        rh.append(hmax)
        rw.append(wsum)

    W = margin * 2 + max(rw + [800])
    H = title_h + sum(rh) + row_gap * max(0, len(rh) - 1) + margin
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pr = precision_recall_vs_gt(hyp.get("tids") or [], gt_set)
    for i, line in enumerate(
        [
            f"{mode_label} {short} Top-{rank}  n_seg={hyp.get('n_segments')}  "
            f"score={hyp.get('score', 0):.4f}",
            f"prec={pr['precision']:.4f} ({pr['n_hit']}/{pr['n_path']})  "
            f"rec={pr['recall']:.4f} ({pr['n_hit']}/{pr['n_gt']})  "
            f"P={hyp.get('path_probability', 0):.6f}",
            f"邊標 {app_tag} / C / S / Σ    GT 綠框 / 非GT 紅框    GT 僅評估",
        ]
    ):
        draw.text((margin, 6 + i * 14), line, fill=(10, 10, 10), font=FONT_XS)

    y = title_h
    for row in rows:
        if row[0] == "gap":
            draw.rectangle(
                [margin, y, W - margin, y + 26],
                fill=(255, 245, 230),
                outline=(200, 120, 40),
            )
            draw.text((margin + 6, y + 6), row[1], fill=(160, 80, 0), font=FONT_XS)
            y += 30 + row_gap
            continue
        x = margin
        hmax = th + 84
        for mems, e in row[1]:
            cw = max(140, len(mems) * (tw + 8) + 12)
            bc = (0, 160, 0) if all(m in gt_set for m in mems) else (200, 40, 40)
            draw.rectangle([x, y, x + cw - 1, y + hmax - 1], outline=bc, width=3)
            if len(mems) > 1:
                draw.text((x + 4, y + 2), "共存合併", fill=bc, font=FONT_XS)
            for i, tid in enumerate(mems):
                t = by_tid.get(tid)
                cam, tid_s = tid.rsplit("_", 1)
                try:
                    _, crops = tp._crop_paths_for_track(merge, cam, int(tid_s))
                    rep = tp._pick_rep_crop(crops)
                except Exception:
                    rep = None
                sx = x + 4 + i * (tw + 8)
                sy = y + 16
                img.paste(thumb(rep, (tw, th)), (sx, sy))
                mbc = (0, 160, 0) if tid in gt_set else (200, 40, 40)
                draw.rectangle(
                    [sx, sy, sx + tw - 1, sy + th - 1], outline=mbc, width=2
                )
                if t:
                    txt = (
                        f"{tid}\n{cam}\n"
                        f"[{t.t_start:.1f}-{t.t_end:.1f}]"
                    )
                else:
                    txt = tid
                ly = sy + th + 2
                for line in txt.split("\n"):
                    draw.text((sx, ly), line, fill=(20, 20, 20), font=FONT_XS)
                    ly += 11
            x += cw
            if e is not None:
                mid = y + hmax // 2
                draw.line(
                    [(x + 4, mid), (x + cell_gap - 8, mid)],
                    fill=(40, 40, 40),
                    width=2,
                )
                sc = e.get("score", 0)
                sc_col = (
                    (0, 140, 0)
                    if (isinstance(sc, (int, float)) and sc >= 0)
                    else (200, 0, 0)
                )
                draw.multiline_text(
                    (x + 2, mid - 36),
                    f"{app_tag}={_fmt(e.get(app_key))}\nC={_fmt(e.get('C'))}\n"
                    f"S={_fmt(e.get('S'))}",
                    fill=(40, 40, 140),
                    font=FONT_XS,
                )
                draw.text(
                    (x + 2, mid + 16), f"Σ={_fmt(sc)}", fill=sc_col, font=FONT_XS
                )
                x += cell_gap
        y += hmax + row_gap

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    print(f"拼圖：{out_png}")


def _hyp_flat_edges(hyp: dict) -> list:
    edges = []
    for seg in sorted(hyp.get("segments") or [], key=lambda s: int(s.get("segment") or 1)):
        edges.extend(seg.get("edges") or [])
    if not edges:
        edges = list(hyp.get("edges") or [])
    return edges


def _pairwise_dist_list(embs: list) -> tuple[float, list[float], int]:
    """回傳 (mean_dist, 各對距離列表, n_crops)。"""
    n = len(embs)
    if n < 2:
        return 0.0, [], n
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(_cos_dist(embs[i], embs[j]))
    return (sum(dists) / len(dists), dists, n)


def _write_m6_edge_walkthrough(
    out_path: Path,
    *,
    short: str,
    hyp: dict,
    edge: dict,
    nodes: list,
    tracks: list,
    merge: Path,
) -> None:
    """單邊 A/C/S 完整展開（與程式輸出交叉驗證）。"""
    by_tid = {t.tid: t for t in tracks}
    # 確保 crop_embs
    attach_crop_embs(tracks, merge)

    u = by_tid.get(edge.get("from"))
    v = by_tid.get(edge.get("to"))
    if u is None or v is None:
        raise RuntimeError(
            f"inspect_m6 異常：邊成員找不到 {edge.get('from')} / {edge.get('to')}"
        )

    # --- A ---
    embs_u = u.meta.get("crop_embs") or []
    embs_v = v.meta.get("crop_embs") or []
    w_u, dists_u, n_u = _pairwise_dist_list(embs_u)
    w_v, dists_v, n_v = _pairwise_dist_list(embs_v)
    d_uv = _cos_dist(np.asarray(u.emb), np.asarray(v.emb))
    A_re = m4_A(u, v)
    A_prog = edge.get("A")

    # --- 重建圖上的 P（與 C/S 一致）---
    # 用節點找 from_super / to_super
    tid_to_sn = {}
    for i, sn in enumerate(nodes):
        for tid in sn.tids:
            tid_to_sn[tid] = i
    i = tid_to_sn[u.tid]
    j = tid_to_sn[v.tid]
    succ, _, _, _ = _build_succ_m6(nodes)
    # 找到對應邊
    e_live = None
    outs = []
    for jj, ee in succ[i]:
        outs.append((jj, ee))
        if jj == j:
            e_live = ee
    if e_live is None:
        raise RuntimeError(
            f"inspect_m6 異常：圖上找不到邊 {nodes[i].label}→{nodes[j].label}"
        )

    # 正向競爭者
    outs_sorted = sorted(outs, key=lambda x: -float(x[1].get("P_fwd") or 0))
    N_u = len(outs)
    # 反向
    inns = []
    for ii, items in enumerate(succ):
        for jj, ee in items:
            if jj == j:
                inns.append((ii, ee))
    inns_sorted = sorted(inns, key=lambda x: -float(x[1].get("P_bwd") or 0))
    N_v = len(inns)

    lines = [
        "# M6 單邊算法完整展開",
        "",
        "> **GT 僅用於評估標注，不參與計分。** 本檔為純輸出，不改邏輯。",
        "",
        f"資料集：**{short}**；取自 **Top-1** 路徑中之一邊。",
        "",
        f"- Top-1 路徑：`{hyp.get('path') or ' -> '.join(hyp.get('super_labels') or [])}`",
        f"- 展開邊：`{edge.get('from_super')}` → `{edge.get('to_super')}`"
        f"（成員 via `{edge.get('from')}`→`{edge.get('to')}`）",
        f"- dt={edge.get('dt'):.2f}s；程式邊分 score={_fmt_stat(edge.get('score'))}",
        "",
        "---",
        "",
        "## 1. A = ln(((w_u+w_v)/2) / d_uv)",
        "",
        f"### 1.1 w_u（`{u.tid}` kept crops 兩兩 cosine distance 平均）",
        "",
        f"- kept 張數 n={n_u}",
        f"- 兩兩對數={len(dists_u)}；**w_u = {w_u:.6f}**",
        f"- Track.meta['w_intra']（程式）= {float(u.meta.get('w_intra', 0)):.6f}",
        "",
    ]
    if dists_u:
        show = dists_u if len(dists_u) <= 20 else dists_u[:20]
        lines.append(
            f"- 各對距離（{'全部' if len(dists_u) <= 20 else f'前 20 / 共 {len(dists_u)}'}）："
            + ", ".join(f"{d:.4f}" for d in show)
        )
        lines.append("")
    else:
        lines.append("- n<2 → w_u=0（無可觀測內部波動）")
        lines.append("")

    lines += [
        f"### 1.2 w_v（`{v.tid}`）",
        "",
        f"- kept 張數 n={n_v}",
        f"- 兩兩對數={len(dists_v)}；**w_v = {w_v:.6f}**",
        f"- Track.meta['w_intra']（程式）= {float(v.meta.get('w_intra', 0)):.6f}",
        "",
    ]
    if dists_v:
        show = dists_v if len(dists_v) <= 20 else dists_v[:20]
        lines.append(
            f"- 各對距離（{'全部' if len(dists_v) <= 20 else f'前 20 / 共 {len(dists_v)}'}）："
            + ", ".join(f"{d:.4f}" for d in show)
        )
        lines.append("")
    else:
        lines.append("- n<2 → w_v=0")
        lines.append("")

    num = 0.5 * (w_u + w_v)
    lines += [
        "### 1.3 d_uv 與 A",
        "",
        f"- 代表向量 cosine 相似度 cos = {float(np.dot(u.emb, v.emb)):.6f}",
        f"- **d_uv = 1 − cos = {d_uv:.6f}**",
        f"- (w_u+w_v)/2 = {num:.6f}",
        f"- 手算 A = ln({num:.6f}/{d_uv:.6f}) = **{_fmt_stat(A_re)}**",
        f"- 程式 edge['A'] = **{_fmt_stat(A_prog)}**；"
        f"一致={'✓' if _finite(A_re) is not None and _finite(A_prog) is not None and abs(A_re - float(A_prog)) < 1e-6 else ('✓(同號inf)' if (A_re == float('inf') and edge.get('A_inf') and float(A_prog)>0) or (A_re == float('-inf') and edge.get('A_inf') and float(A_prog)<0) else '核对')}",
        f"- 程式 edge['d_uv'] = {_fmt_stat(edge.get('d_uv'), 6)}；edge['w_u']={_fmt_stat(edge.get('w_u'), 6)}；edge['w_v']={_fmt_stat(edge.get('w_v'), 6)}",
        "",
        "---",
        "",
        "## 2. C = ln(N_u·P(v|u)) + ln(N_v·P(u|v))",
        "",
        f"### 2.1 正向：from_super=`{nodes[i].label}` 的全部下家（N_u={N_u}）",
        "",
        f"- 目標 P(v|u) = **{_fmt_stat(e_live.get('P_fwd'), 4)}**",
        f"- C_fwd = ln(N_u·P) = **{_fmt_stat(e_live.get('C_fwd'))}**",
        "",
        "| 排名 | to_super | A | P(v|u) | 目標 |",
        "|-----:|----------|--:|------:|:----:|",
    ]
    for k, (jj, ee) in enumerate(outs_sorted[:5], 1):
        mark = "★" if jj == j else ""
        lines.append(
            f"| {k} | `{ee.get('to_super')}` | {_fmt_stat(ee.get('A'))} | "
            f"{_fmt_stat(ee.get('P_fwd'), 4)} | {mark} |"
        )
    if N_u > 5:
        rest_p = sum(float(ee.get("P_fwd") or 0) for _, ee in outs_sorted[5:])
        lines.append(
            f"| … | （其餘 {N_u - 5} 名合計） | — | {_fmt_stat(rest_p, 4)} | |"
        )
    lines.append("")

    lines += [
        f"### 2.2 反向：to_super=`{nodes[j].label}` 的全部前家（N_v={N_v}）",
        "",
        f"- 目標 P(u|v) = **{_fmt_stat(e_live.get('P_bwd'), 4)}**",
        f"- C_bwd = ln(N_v·P) = **{_fmt_stat(e_live.get('C_bwd'))}**",
        "",
        "| 排名 | from_super | A | P(u|v) | 目標 |",
        "|-----:|------------|--:|------:|:----:|",
    ]
    for k, (ii, ee) in enumerate(inns_sorted[:5], 1):
        mark = "★" if ii == i else ""
        lines.append(
            f"| {k} | `{ee.get('from_super')}` | {_fmt_stat(ee.get('A'))} | "
            f"{_fmt_stat(ee.get('P_bwd'), 4)} | {mark} |"
        )
    if N_v > 5:
        rest_p = sum(float(ee.get("P_bwd") or 0) for _, ee in inns_sorted[5:])
        lines.append(
            f"| … | （其餘 {N_v - 5} 名合計） | — | {_fmt_stat(rest_p, 4)} | |"
        )
    lines.append("")
    lines.append(
        f"- **C 合計 = {_fmt_stat(e_live.get('C'))}**"
        f"（程式 edge['C']={_fmt_stat(edge.get('C'))}）"
    )
    lines.append("")

    # --- S ---
    skipped = e_live.get("skipped") or []
    sum_p = float(e_live.get("sum_P_skipped") or 0.0)
    S_re = float(math.log(1.0 - sum_p)) if sum_p < 1.0 else float("-inf")
    lines += [
        "---",
        "",
        "## 3. S = ln(1 − Σ P(w|u))",
        "",
        f"被跳過者：u 的下家中 `t_start < v.t_start`（v=`{nodes[j].label}`，"
        f"t_start={nodes[j].t_start:.2f}），且 ≠ v。",
        "",
        f"- n_skipped = {len(skipped)}",
        f"- **Σ P = {sum_p:.6f}**",
        f"- 手算 S = ln(1−{sum_p:.6f}) = **{_fmt_stat(S_re)}**",
        f"- 程式 edge['S'] = **{_fmt_stat(edge.get('S'))}**",
        "",
        "| to_super | t_start | P | A |",
        "|----------|--------:|--:|--:|",
    ]
    if not skipped:
        lines.append("| （無） | | | |")
    for sk in skipped:
        lines.append(
            f"| `{sk.get('to_super')}` | {sk.get('t_start', 0):.2f} | "
            f"{_fmt_stat(sk.get('P'), 4)} | {_fmt_stat(sk.get('A'))} |"
        )
    lines.append("")

    # --- 合計 ---
    A_f = _finite(e_live.get("A"))
    C_f = _finite(e_live.get("C"))
    S_f = _finite(e_live.get("S"))
    if A_f is not None and C_f is not None and S_f is not None:
        hand = A_f + C_f + S_f
    else:
        hand = None
    prog = e_live.get("score")
    lines += [
        "---",
        "",
        "## 4. 三項合計 = 邊分",
        "",
        f"- A + C + S = {_fmt_stat(A_f)} + {_fmt_stat(C_f)} + {_fmt_stat(S_f)}"
        f" = **{_fmt_stat(hand)}**",
        f"- 程式 edge['score'] = **{_fmt_stat(prog)}**",
        f"- 路徑 Top-1 內該邊 score（假設物件）= **{_fmt_stat(edge.get('score'))}**",
        "",
    ]
    ok = (
        hand is not None
        and _finite(prog) is not None
        and abs(hand - float(prog)) < 1e-5
    )
    lines.append(f"**交叉驗證：{'一致 ✓' if ok else '不一致 ★（請查）'}**")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"單邊展開：{out_path}")


def _write_m6_pair_stats(
    out_path: Path,
    *,
    results_by_short: dict,
) -> None:
    """六張拼圖涉及節點的 GT–GT / GT–非GT / 非GT–非GT 配對統計。"""
    lines = [
        "# M6 配對分數統計表",
        "",
        "> **GT 僅用於分組評估，不參與計分。**",
        "",
        "範圍：0507/0528 的 Top-1～Top-3 拼圖所涉全部超節點之間，"
        "在 M6 時間序圖上的合法邊。",
        "",
    ]

    for short, data in results_by_short.items():
        result = data["result"]
        gt_set = data["gt_set"]
        merge = data["merge"]
        nodes = result["nodes"]
        tracks = result["tracks"]
        attach_crop_embs(tracks, merge)
        by_tid = {t.tid: t for t in tracks}
        tid_to_sn = {}
        for i, sn in enumerate(nodes):
            for tid in sn.tids:
                tid_to_sn[tid] = i

        # 六張拼圖涉及的節點 index
        involved = set()
        for hyp in result["ranked"][:3]:
            for tid in hyp.get("tids") or []:
                if tid in tid_to_sn:
                    involved.add(tid_to_sn[tid])
            for lab in hyp.get("super_labels") or []:
                # also via edges
                pass
            for e in _hyp_flat_edges(hyp):
                for tid in (e.get("from_members") or []) + (e.get("to_members") or []):
                    if tid in tid_to_sn:
                        involved.add(tid_to_sn[tid])

        succ, _, _, _ = _build_succ_m6(nodes)
        # 邊：兩端超節點都在 involved
        pairs = {"GT-GT": [], "GT-非GT": [], "非GT-非GT": []}
        for i, items in enumerate(succ):
            if i not in involved:
                continue
            for j, e in items:
                if j not in involved:
                    continue
                from_gt = all(t in gt_set for t in (e.get("from_members") or []))
                to_gt = all(t in gt_set for t in (e.get("to_members") or []))
                # 端點分類：任一 member 在 GT → 該超節點算 GT 側？
                # 使用者：兩端皆 GT / GT–非GT / 非GT–非GT
                # 採超節點：全成員皆 GT → GT；否則非GT（與拼圖著色一致：all GT = 綠）
                sa_gt = all(t in gt_set for t in nodes[i].tids)
                sb_gt = all(t in gt_set for t in nodes[j].tids)
                if sa_gt and sb_gt:
                    key = "GT-GT"
                elif sa_gt or sb_gt:
                    key = "GT-非GT"
                else:
                    key = "非GT-非GT"
                emb = e.get("emb")
                pairs[key].append(
                    {
                        "from": e.get("from_super"),
                        "to": e.get("to_super"),
                        "via": f"{e.get('from')}→{e.get('to')}",
                        "score": e.get("score"),
                        "A": e.get("A"),
                        "C": e.get("C"),
                        "S": e.get("S"),
                        "emb": emb,
                    }
                )

        lines.append(f"## {short}")
        lines.append("")
        lines.append(
            f"- 拼圖涉及超節點數：{len(involved)} / 全圖 {len(nodes)}"
        )
        lines.append("")

        for key in ("GT-GT", "GT-非GT", "非GT-非GT"):
            rows = pairs[key]
            lines.append(f"### {key}（n={len(rows)}）")
            lines.append("")
            if not rows:
                lines.append("（無合法邊）")
                lines.append("")
                continue
            st_sc = _score_stats([r["score"] for r in rows])
            st_a = _score_stats([r["A"] for r in rows])
            embs = [float(r["emb"]) for r in rows if r.get("emb") is not None]
            lines.append(
                f"- **A+C+S**：mean={_fmt_stat(st_sc.get('mean'))}  "
                f"min={_fmt_stat(st_sc.get('min'))}  max={_fmt_stat(st_sc.get('max'))}  "
                f"median={_fmt_stat(st_sc.get('median'))}"
            )
            lines.append(
                f"- **A 單獨**：mean={_fmt_stat(st_a.get('mean'))}  "
                f"min={_fmt_stat(st_a.get('min'))}  max={_fmt_stat(st_a.get('max'))}  "
                f"median={_fmt_stat(st_a.get('median'))}"
            )
            if embs:
                lines.append(
                    f"- **emb 原始值範圍**：[{min(embs):.4f}, {max(embs):.4f}]  "
                    f"mean={sum(embs)/len(embs):.4f}"
                )
            else:
                lines.append("- emb：—")
            lines.append("")
            lines.append(
                "| from → to | via | emb | A | C | S | A+C+S |"
            )
            lines.append(
                "|-----------|-----|----:|--:|--:|--:|------:|"
            )
            # 按 score 降序列每筆
            rows_sorted = sorted(
                rows,
                key=lambda r: -(
                    _finite(r["score"])
                    if _finite(r["score"]) is not None
                    else -1e300
                ),
            )
            for r in rows_sorted:
                emb_s = (
                    f"{float(r['emb']):.4f}" if r.get("emb") is not None else "—"
                )
                lines.append(
                    f"| `{r['from']}`→`{r['to']}` | `{r['via']}` | "
                    f"{emb_s} | "
                    f"{_fmt_stat(r['A'])} | {_fmt_stat(r['C'])} | {_fmt_stat(r['S'])} | "
                    f"{_fmt_stat(r['score'])} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("GT 僅評估；純輸出、未改 M6 邏輯。")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"配對統計：{out_path}")


def cmd_inspect_m6(argv=None):
    """M6 檢視包：Top-1/2/3 拼圖 + 單邊展開 + 配對統計（純輸出）。"""
    p = argparse.ArgumentParser(description="M6 檢視包（純輸出）")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m6_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "tag": "人員追蹤_20260507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
        },
        {
            "short": "0528",
            "tag": "人員追蹤_20260528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
        },
    ]

    results_by_short = {}
    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])
        print(f"\n===== inspect_m6 {short} =====")
        result = run_with_config(
            ds["merge"],
            RunConfig(
                scoring="m6",
                node_score=False,
                dt_max=None,
                sim_min=float(args.sim_min),
                variant_tag="M6",
            ),
        )
        by_tid = {t.tid: t for t in result["tracks"]}
        ranked = result["ranked"]
        for k in (1, 2, 3):
            if k - 1 >= len(ranked):
                print(f"[{short}] 無 Top-{k}")
                continue
            hyp = ranked[k - 1]
            out_png = out_dir / f"{short}_m6_top{k}_collage.png"
            _render_one_m6_collage(
                hyp=hyp,
                rank=k,
                short=short,
                dataset_tag=ds["tag"],
                gt_set=gt_set,
                merge=ds["merge"],
                by_tid=by_tid,
                out_png=out_png,
            )
        results_by_short[short] = {
            "result": result,
            "gt_set": gt_set,
            "merge": ds["merge"],
        }

    # 單邊展開：優先 0528 Top-1 中 n_skipped>0 且有完整 A/C/S 的邊；
    # 否則 0507；再否則第一條邊
    walk_short = None
    walk_hyp = None
    walk_edge = None
    for short in ("0528", "0507"):
        if short not in results_by_short:
            continue
        hyp = results_by_short[short]["result"]["ranked"][0]
        edges = _hyp_flat_edges(hyp)
        cand = [e for e in edges if int(e.get("n_skipped") or 0) > 0]
        if not cand:
            cand = edges
        if cand:
            # 取 |S| 中等、資訊量高者：n_skipped 最大
            cand.sort(key=lambda e: (-int(e.get("n_skipped") or 0), -abs(float(e.get("S") or 0))))
            walk_short, walk_hyp, walk_edge = short, hyp, cand[0]
            break
    if walk_edge is None:
        raise RuntimeError("inspect_m6 異常：找不到可展開的邊")

    _write_m6_edge_walkthrough(
        out_dir / "edge_walkthrough_m6.md",
        short=walk_short,
        hyp=walk_hyp,
        edge=walk_edge,
        nodes=results_by_short[walk_short]["result"]["nodes"],
        tracks=results_by_short[walk_short]["result"]["tracks"],
        merge=results_by_short[walk_short]["merge"],
    )
    _write_m6_pair_stats(out_dir / "pair_score_stats_m6.md", results_by_short=results_by_short)
    print(f"\n檢視包完成 → {out_dir}")
    return out_dir


def _w_hist_png(ws: list[float], out_png: Path, *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(ws, bins=min(30, max(8, len(ws) // 2)), color="#4a7c9b", edgecolor="white")
    ax.set_xlabel("w (mean pairwise cosine distance of kept crops)")
    ax.set_ylabel("count")
    ax.set_title(title)
    if ws:
        mu = float(np.mean(ws))
        ax.axvline(mu, color="#c0392b", linestyle="--", linewidth=1.2, label=f"mean={mu:.4f}")
        ax.legend(loc="upper right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _save_rep_crop_thumb(merge: Path, tid: str, out_png: Path, size=(160, 200)) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False
    cam, tid_s = tid.rsplit("_", 1)
    try:
        _, crops = tp._crop_paths_for_track(merge, cam, int(tid_s))
        rep = tp._pick_rep_crop(crops)
    except Exception:
        rep = None
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if rep is None or not Path(rep).is_file():
        Image.new("RGB", size, (230, 230, 230)).save(out_png)
        return False
    im = Image.open(rep).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (236, 236, 236))
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    canvas.save(out_png)
    return True


def _w_stats_block(ws: list[float]) -> dict:
    if not ws:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "cv": None,
        }
    arr = np.asarray(ws, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    cv = (std / mean) if mean != 0.0 else None
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "cv": cv,
    }


def cmd_w_distribution(argv=None):
    """純輸出：0507/0528 全部候選軌（含超節點成員）的 w 分布。"""
    p = argparse.ArgumentParser(description="w 分布統計（純輸出）")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m6_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    extreme_dir = out_dir / "w_extremes"
    extreme_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
        },
    ]

    lines = [
        "# w 分布統計",
        "",
        f"生成時間：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "> 純輸出、不改計分邏輯。**GT 不參與。**",
        "",
        "定義：`w` = 該軌 kept crops 兩兩 cosine distance 平均"
        "（與 `attach_crop_embs` / `Track.meta['w_intra']` 相同）；"
        "n_crop < 2 → w = 0。",
        "",
        "範圍：與 M6 相同候選池（`SIM_MIN` 過濾後的全部軌，含後續併入超節點者）。",
        "超節點多成員另附合併 crops 的 w，供對照。",
        "",
    ]

    for ds in datasets:
        short = ds["short"]
        merge = ds["merge"]
        print(f"\n===== w_distribution {short} =====")
        tp.SIM_MIN = float(args.sim_min)
        tp.configure_for_input(str(merge))
        tracks = tp.load_tracks(str(merge))
        attach_crop_embs(tracks, merge)
        coexist_median = median_edge_emb(tracks)
        nodes, super_report = tp.build_supernodes(
            tracks, overlap_emb_min=coexist_median
        )

        rows = []
        for t in tracks:
            embs = t.meta.get("crop_embs") or []
            # n_crop：kept 有 emb 的張數（與 w 計算一致）；fallback 僅代表向量時 n=1
            n_crop = len(embs)
            w = float(t.meta.get("w_intra", 0.0))
            rows.append(
                {
                    "tid": t.tid,
                    "cam": t.cam,
                    "n_crop": n_crop,
                    "w": w,
                    "t_start": float(t.t_start),
                    "t_end": float(t.t_end),
                    "dur": float(t.t_end - t.t_start),
                    "sim": float(t.sim) if t.sim is not None else None,
                }
            )
        rows.sort(key=lambda r: (-r["w"], r["tid"]))
        ws = [r["w"] for r in rows]
        st = _w_stats_block(ws)

        hist_name = f"w_hist_{short}.png"
        hist_png = out_dir / hist_name
        _w_hist_png(
            ws,
            hist_png,
            title=f"{short} w distribution (n={st['n']})",
        )
        print(f"直方圖：{hist_png}")

        # 極端軌：最大/最小各 3（n_crop>=2 優先，否則含 w=0）
        with_pair = [r for r in rows if r["n_crop"] >= 2]
        pool = with_pair if len(with_pair) >= 3 else rows
        top_max = sorted(pool, key=lambda r: (-r["w"], r["tid"]))[:3]
        top_min = sorted(pool, key=lambda r: (r["w"], r["tid"]))[:3]

        lines += [
            f"## {short}",
            "",
            f"- 候選軌數 n={st['n']}；超節點數={super_report.get('n_supernodes')}；"
            f"多成員合併對數={super_report.get('n_merged_pairs')}",
            f"- **mean**={_fmt_stat(st['mean'], 6)}  "
            f"**std**={_fmt_stat(st['std'], 6)}  "
            f"**min**={_fmt_stat(st['min'], 6)}  "
            f"**max**={_fmt_stat(st['max'], 6)}",
            f"- **變異係數 CV = std/mean** = "
            f"{_fmt_stat(st['cv'], 4) if st['cv'] is not None else '—（mean=0）'}",
            f"- 直方圖：`{hist_name}`",
            "",
            f"![w hist {short}]({hist_name})",
            "",
            "### w 最大 3 條（人工看：是否轉身／模糊）",
            "",
            "| # | tid | n_crop | w | 時長(s) | 代表 crop |",
            "|--:|-----|-------:|--:|--------:|-----------|",
        ]
        for i, r in enumerate(top_max, 1):
            thumb = extreme_dir / f"{short}_max{i}_{r['tid']}.png"
            _save_rep_crop_thumb(merge, r["tid"], thumb)
            rel = f"w_extremes/{thumb.name}"
            lines.append(
                f"| {i} | `{r['tid']}` | {r['n_crop']} | {r['w']:.6f} | "
                f"{r['dur']:.1f} | ![]({rel}) |"
            )
        lines += [
            "",
            "### w 最小 3 條（人工看：是否短直軌）",
            "",
            "| # | tid | n_crop | w | 時長(s) | 代表 crop |",
            "|--:|-----|-------:|--:|--------:|-----------|",
        ]
        for i, r in enumerate(top_min, 1):
            thumb = extreme_dir / f"{short}_min{i}_{r['tid']}.png"
            _save_rep_crop_thumb(merge, r["tid"], thumb)
            rel = f"w_extremes/{thumb.name}"
            lines.append(
                f"| {i} | `{r['tid']}` | {r['n_crop']} | {r['w']:.6f} | "
                f"{r['dur']:.1f} | ![]({rel}) |"
            )
        lines.append("")

        # 每軌一覽
        lines += [
            "### 每軌一覽",
            "",
            "| tid | n_crop | w | t_start | t_end | dur(s) | sim |",
            "|-----|-------:|--:|--------:|------:|------:|----:|",
        ]
        for r in rows:
            sim_s = f"{r['sim']:.3f}" if r["sim"] is not None else "—"
            lines.append(
                f"| `{r['tid']}` | {r['n_crop']} | {r['w']:.6f} | "
                f"{r['t_start']:.1f} | {r['t_end']:.1f} | {r['dur']:.1f} | {sim_s} |"
            )
        lines.append("")

        # 超節點（多成員）：合併 crops 的 w
        multi = [sn for sn in nodes if len(sn.members) > 1]
        lines += [
            "### 超節點（多成員）合併 crops 的 w",
            "",
            "定義同軌：成員 kept crops 全部合併後兩兩平均。",
            "",
        ]
        if not multi:
            lines.append("（無多成員超節點）")
            lines.append("")
        else:
            lines.append("| label | members | n_crop | w |")
            lines.append("|-------|---------|-------:|--:|")
            sn_rows = []
            for sn in multi:
                _, w_sn, n_c = _sn_rep_from_crops(sn.members)
                sn_rows.append((sn.label, sn.tids, n_c, w_sn))
            sn_rows.sort(key=lambda x: -x[3])
            for lab, mems, n_c, w_sn in sn_rows:
                lines.append(
                    f"| `{lab}` | `{','.join(mems)}` | {n_c} | {w_sn:.6f} |"
                )
            lines.append("")

    lines += [
        "---",
        "",
        "純輸出；未改 M6／既有模式；未調參。",
        "",
    ]
    out_md = out_dir / "w_distribution.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"報告：{out_md}")
    return out_md


def _analyze_m7_edges(short, merge, gt_set, result_m7, result_m6=None):
    nodes = result_m7["nodes"]
    tracks = result_m7["tracks"]
    by_tid = {t.tid: t for t in tracks}
    attach_crop_embs(tracks, merge)
    succ, _, n_legal, m7_meta = _build_succ_m7(nodes)
    edges = []
    for i, items in enumerate(succ):
        for j, e in items:
            e2 = dict(e)
            e2["_i"] = i
            e2["_j"] = j
            e2["LCS"] = e.get("score")  # logit+C+S
            edges.append(e2)

    gt_keys, same_sn, no_edge, missing, ordered = _m5_gt_edge_keys(
        nodes, edges, sorted(gt_set), by_tid
    )
    for e in edges:
        e["is_gt"] = (e["_i"], e["_j"]) in gt_keys

    gt_e = [e for e in edges if e["is_gt"]]
    ng_e = [e for e in edges if not e["is_gt"]]

    def pack_score(key):
        st_gt = _score_stats([e.get(key) for e in gt_e])
        st_ng = _score_stats([e.get(key) for e in ng_e])
        md, d = _m5_effect(st_gt, st_ng)
        return {"st_gt": st_gt, "st_ng": st_ng, "mean_diff": md, "effect_d": d}

    scores = {
        "logit": pack_score("logit"),
        "C": pack_score("C"),
        "S": pack_score("S"),
        "logit+C+S": pack_score("LCS"),
    }

    m6_cmp = None
    if result_m6 is not None:
        attach_crop_embs(result_m6["tracks"], merge)
        nodes6 = result_m6["nodes"]
        succ6, _, _, _ = _build_succ_m6(nodes6)
        edges6 = []
        for i, items in enumerate(succ6):
            for j, e in items:
                e2 = dict(e)
                e2["_i"] = i
                e2["_j"] = j
                edges6.append(e2)
        by6 = {t.tid: t for t in result_m6["tracks"]}
        gk6, _, _, _, _ = _m5_gt_edge_keys(nodes6, edges6, sorted(gt_set), by6)
        for e in edges6:
            e["is_gt"] = (e["_i"], e["_j"]) in gk6
        gt6 = [e for e in edges6 if e["is_gt"]]
        ng6 = [e for e in edges6 if not e["is_gt"]]
        st_gt = _score_stats([e.get("score") for e in gt6])
        st_ng = _score_stats([e.get("score") for e in ng6])
        md, d = _m5_effect(st_gt, st_ng)
        m6_cmp = {
            "n_edges": len(edges6),
            "n_gt": len(gt6),
            "n_nongt": len(ng6),
            "st_gt": st_gt,
            "st_ng": st_ng,
            "mean_diff": md,
            "effect_d": d,
        }

    return {
        "short": short,
        "n_nodes": len(nodes),
        "n_edges": n_legal,
        "n_degenerate": m7_meta.get("n_degenerate"),
        "n_gt": len(gt_e),
        "n_nongt": len(ng_e),
        "same_sn": same_sn,
        "no_edge": no_edge,
        "missing": missing,
        "scores": scores,
        "m6_ACS": m6_cmp,
        "gt_edges": gt_e,
        "edges": edges,
        "m7_meta": m7_meta,
    }


def _write_m7_pair_stats(out_path: Path, *, results_by_short: dict) -> None:
    """Top-1～3 涉及節點的 GT–GT / GT–非GT / 非GT–非GT 配對統計（M7）。"""
    lines = [
        "# M7 配對分數統計表",
        "",
        "> **GT 僅用於分組評估，不參與計分。**",
        "",
        "範圍：0507/0528 的 Top-1～Top-3 所涉全部超節點之間，"
        "在 M7 時間序圖上的合法邊。",
        "",
        "邊分 = logit + C + S；logit = ln(emb/(1−emb))",
        "",
    ]
    for short, data in results_by_short.items():
        result = data["result"]
        gt_set = data["gt_set"]
        merge = data["merge"]
        nodes = result["nodes"]
        tracks = result["tracks"]
        attach_crop_embs(tracks, merge)
        tid_to_sn = {}
        for i, sn in enumerate(nodes):
            for tid in sn.tids:
                tid_to_sn[tid] = i

        involved = set()
        for hyp in result["ranked"][:3]:
            for tid in hyp.get("tids") or []:
                if tid in tid_to_sn:
                    involved.add(tid_to_sn[tid])
            for e in _hyp_flat_edges(hyp):
                for tid in (e.get("from_members") or []) + (e.get("to_members") or []):
                    if tid in tid_to_sn:
                        involved.add(tid_to_sn[tid])

        succ, _, _, _ = _build_succ_m7(nodes)
        pairs = {"GT-GT": [], "GT-非GT": [], "非GT-非GT": []}
        for i, items in enumerate(succ):
            if i not in involved:
                continue
            for j, e in items:
                if j not in involved:
                    continue
                sa_gt = all(t in gt_set for t in nodes[i].tids)
                sb_gt = all(t in gt_set for t in nodes[j].tids)
                if sa_gt and sb_gt:
                    key = "GT-GT"
                elif sa_gt or sb_gt:
                    key = "GT-非GT"
                else:
                    key = "非GT-非GT"
                pairs[key].append(
                    {
                        "from": e.get("from_super"),
                        "to": e.get("to_super"),
                        "via": f"{e.get('from')}→{e.get('to')}",
                        "score": e.get("score"),
                        "logit": e.get("logit"),
                        "C": e.get("C"),
                        "S": e.get("S"),
                        "emb": e.get("emb"),
                    }
                )

        lines.append(f"## {short}")
        lines.append("")
        lines.append(
            f"- 涉及超節點數：{len(involved)} / 全圖 {len(nodes)}"
        )
        lines.append("")

        for key in ("GT-GT", "GT-非GT", "非GT-非GT"):
            rows = pairs[key]
            lines.append(f"### {key}（n={len(rows)}）")
            lines.append("")
            if not rows:
                lines.append("（無合法邊）")
                lines.append("")
                continue
            st_sc = _score_stats([r["score"] for r in rows])
            st_l = _score_stats([r["logit"] for r in rows])
            embs = [float(r["emb"]) for r in rows if r.get("emb") is not None]
            lines.append(
                f"- **logit+C+S**：mean={_fmt_stat(st_sc.get('mean'))}  "
                f"min={_fmt_stat(st_sc.get('min'))}  max={_fmt_stat(st_sc.get('max'))}  "
                f"median={_fmt_stat(st_sc.get('median'))}"
            )
            lines.append(
                f"- **logit 單獨**：mean={_fmt_stat(st_l.get('mean'))}  "
                f"min={_fmt_stat(st_l.get('min'))}  max={_fmt_stat(st_l.get('max'))}  "
                f"median={_fmt_stat(st_l.get('median'))}"
            )
            if embs:
                lines.append(
                    f"- **emb 原始值範圍**：[{min(embs):.4f}, {max(embs):.4f}]  "
                    f"mean={sum(embs)/len(embs):.4f}"
                )
            else:
                lines.append("- emb：—")
            lines.append("")
            lines.append(
                "| from → to | via | emb | logit | C | S | logit+C+S |"
            )
            lines.append(
                "|-----------|-----|----:|------:|--:|--:|----------:|"
            )
            rows_sorted = sorted(
                rows,
                key=lambda r: -(
                    _finite(r["score"])
                    if _finite(r["score"]) is not None
                    else -1e300
                ),
            )
            for r in rows_sorted:
                emb_s = (
                    f"{float(r['emb']):.4f}" if r.get("emb") is not None else "—"
                )
                lines.append(
                    f"| `{r['from']}`→`{r['to']}` | `{r['via']}` | "
                    f"{emb_s} | "
                    f"{_fmt_stat(r['logit'])} | {_fmt_stat(r['C'])} | "
                    f"{_fmt_stat(r['S'])} | {_fmt_stat(r['score'])} |"
                )
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("GT 僅評估；純輸出、未改 M7／M6 邏輯。")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"配對統計：{out_path}")


def _render_m7_report(account: dict) -> str:
    lines = [
        "# M7（logit + C + S）對照實驗",
        "",
        f"生成時間：{account.get('generated_at')}",
        "",
        "> **GT 僅用於評估與著色，不參與計分。**",
        "",
        "邊分 = ln(emb/(1−emb)) + C + S；C、S 公式同 M6；"
        "建邊僅時間順序；hop 不計分；Σ≥1 → degenerate；附 min-logit。",
        "",
        "## 1. 系統層級：M0 / M6 / M7",
        "",
        "| 資料集 | 版 | precision | recall | P | n_seg | n_path | n_hit | min-logit |",
        "|--------|----|-----------:|-------:|--:|------:|-------:|------:|----------:|",
    ]
    for short, pack in account["datasets"].items():
        for key in ("M0", "M6", "M7"):
            d = pack.get(key) or {}
            ml = "—"
            if key == "M7":
                ml = _fmt_stat(pack.get("M7_top1_min_logit"))
            if not d:
                lines.append(f"| {short} | {key} | — | — | — | — | — | — | {ml} |")
                continue
            lines.append(
                f"| {short} | {key} | {_fmt_pct(d.get('precision', 0))} | "
                f"{_fmt_pct(d.get('recall', 0))} | {float(d.get('P') or 0):.6f} | "
                f"{d.get('n_segments')} | {d.get('n_path')} | {d.get('n_hit')} | {ml} |"
            )
    lines.append("")
    for short, pack in account["datasets"].items():
        lines.append(
            f"- **{short} Top-1**：`{pack.get('M7_top1_path')}`；"
            f"score={_fmt_stat(pack.get('M7_top1_score'))}；"
            f"min-logit={_fmt_stat(pack.get('M7_top1_min_logit'))}"
        )
    lines.append("")

    lines += ["## 2. 邊層級：logit / logit+C+S vs M6(A+C+S)", ""]
    for short, pack in account["datasets"].items():
        ep = pack.get("edge") or {}
        lines.append(f"### {short}")
        lines.append("")
        lines.append(
            f"- 超節點：{ep.get('n_nodes')}；時間序邊：{ep.get('n_edges')}；"
            f"degenerate：{ep.get('n_degenerate')}；"
            f"GT 邊：{ep.get('n_gt')}；非 GT：{ep.get('n_nongt')}"
        )
        lines.append("")
        lines.append(
            "| 分數 | GT mean | 非GT mean | mean差 | 效應量 d |"
        )
        lines.append("|------|--------:|----------:|-------:|--------:|")
        scores = ep.get("scores") or {}
        for key in ("logit", "C", "S", "logit+C+S"):
            s = scores.get(key) or {}
            stg, stn = s.get("st_gt") or {}, s.get("st_ng") or {}
            lines.append(
                f"| {key} | {_fmt_stat(stg.get('mean'))} | {_fmt_stat(stn.get('mean'))} | "
                f"{_fmt_stat(s.get('mean_diff'))} | {_fmt_stat(s.get('effect_d'))} |"
            )
        m6 = ep.get("m6_ACS")
        if m6:
            lines.append(
                f"| M6 A+C+S | {_fmt_stat((m6.get('st_gt') or {}).get('mean'))} | "
                f"{_fmt_stat((m6.get('st_ng') or {}).get('mean'))} | "
                f"{_fmt_stat(m6.get('mean_diff'))} | {_fmt_stat(m6.get('effect_d'))} |"
            )
        lines.append("")
        d_l = (scores.get("logit") or {}).get("effect_d")
        d_lcs = (scores.get("logit+C+S") or {}).get("effect_d")
        d_m6 = m6.get("effect_d") if m6 else None
        lines.append(
            f"- 效應量對照：logit d={_fmt_stat(d_l)}；"
            f"logit+C+S d={_fmt_stat(d_lcs)}；"
            f"M6 A+C+S d={_fmt_stat(d_m6)}"
        )
        lines.append("")

    lines += ["## 3. 0528 假橋三邊分解", ""]
    fb = (account["datasets"].get("0528") or {}).get("fake_bridge") or {}
    if not fb:
        lines.append("（無）")
    else:
        lines.append(
            f"- Top-1 含 09_96/07_139：**"
            f"{'是 ★' if fb.get('top1_has_09_96_or_07_139') else '否'}**"
        )
        lines.append(f"- Top-1：`{fb.get('top1_path')}`")
        lines.append("")
        lines.append(
            "| from→to | emb | logit | C | S | Σ | n_skip | ΣP_skip |"
        )
        lines.append(
            "|---------|----:|------:|--:|--:|--:|-------:|--------:|"
        )
        for e in fb.get("focus") or []:
            lines.append(
                f"| `{e.get('from')}`→`{e.get('to')}` | "
                f"{_fmt_stat(e.get('emb'), 4)} | {_fmt_stat(e.get('logit'))} | "
                f"{_fmt_stat(e.get('C'))} | {_fmt_stat(e.get('S'))} | "
                f"{_fmt_stat(e.get('score'))} | {e.get('n_skipped')} | "
                f"{_fmt_stat(e.get('sum_P_skipped'), 4)} |"
            )
        lines.append("")
        for e in fb.get("focus") or []:
            lines.append(
                f"### `{e.get('from')}` → `{e.get('to')}`"
            )
            lines.append("")
            lines.append(
                f"- logit={_fmt_stat(e.get('logit'))}  C={_fmt_stat(e.get('C'))}  "
                f"S={_fmt_stat(e.get('S'))}  Σ={_fmt_stat(e.get('score'))}"
            )
            lines.append(
                f"- emb={_fmt_stat(e.get('emb'), 4)}  "
                f"P_fwd={_fmt_stat(e.get('P_fwd'), 4)}"
            )
            lines.append("")

    lines += [
        "## 4. 常數清單",
        "",
        "**`[]`（空）**",
        "",
        "## 5. 配對統計",
        "",
        "- `output/v1.0/m7_comparison/pair_score_stats_m7.md`",
        "",
        "---",
        "",
        "實驗約束：GT 僅評估；純增量；未改 M6；未改 track_path.py；未調參。",
        "",
    ]
    return "\n".join(lines)


def cmd_compare_m7(argv=None):
    """M7（logit+C+S）對照驗證。"""
    p = argparse.ArgumentParser(description="M7 logit+C+S 對照實驗")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m7_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0507_top1.json",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0528_top1.json",
        },
    ]

    account = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "constants": [],
        "datasets": {},
    }
    results_by_short = {}

    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])
        pack: dict = {"gt_set": sorted(gt_set)}

        m0 = json.loads(ds["m0_json"].read_text(encoding="utf-8"))
        pack["M0"] = _top_pack(m0, gt_set, {"source": str(ds["m0_json"])})

        print(f"\n===== {short} M6（對照） =====")
        result6 = run_with_config(
            ds["merge"],
            RunConfig(
                scoring="m6",
                node_score=False,
                dt_max=None,
                sim_min=float(args.sim_min),
                variant_tag="M6",
            ),
        )
        summary6 = _save_summary(result6, ds["merge"], out_dir, f"{short}_M6_top1")
        pack["M6"] = _top_pack(summary6, gt_set, {"constants": []})

        print(f"\n===== {short} M7 =====")
        result7 = run_with_config(
            ds["merge"],
            RunConfig(
                scoring="m7",
                node_score=False,
                dt_max=None,
                sim_min=float(args.sim_min),
                variant_tag="M7",
            ),
        )
        summary7 = _save_summary(result7, ds["merge"], out_dir, f"{short}_M7_top1")
        pack["M7"] = _top_pack(summary7, gt_set, {"constants": []})

        top7 = result7["ranked"][0] if result7["ranked"] else None
        pack["M7_top1_min_logit"] = top7.get("min_logit") if top7 else None
        pack["M7_top1_path"] = (
            top7.get("path")
            or " -> ".join(top7.get("super_labels") or top7.get("tids") or [])
            if top7
            else None
        )
        pack["M7_top1_score"] = top7.get("score") if top7 else None
        pack["M7_top1_P"] = top7.get("path_probability") if top7 else None

        edge_pack = _analyze_m7_edges(
            short, ds["merge"], gt_set, result7, result_m6=result6
        )
        pack["edge"] = {
            k: v
            for k, v in edge_pack.items()
            if k not in ("edges", "gt_edges")
        }

        def _eb(e):
            return {
                "from_super": e.get("from_super"),
                "to_super": e.get("to_super"),
                "from": e.get("from"),
                "to": e.get("to"),
                "dt": e.get("dt"),
                "emb": e.get("emb"),
                "logit": e.get("logit"),
                "A": e.get("A"),
                "C": e.get("C"),
                "S": e.get("S"),
                "score": e.get("score"),
                "n_skipped": e.get("n_skipped"),
                "sum_P_skipped": e.get("sum_P_skipped"),
                "P_fwd": e.get("P_fwd"),
                "is_gt": e.get("is_gt"),
            }

        if short == "0528":
            focus_pairs = {
                ("K8-07_93", "K8-09_96"),
                ("K8-09_96", "K8-07_139"),
                ("K8-07_139", "K8-09_142"),
            }
            # 也接受超節點標籤對到成員
            focus = []
            for e in edge_pack["edges"]:
                pair = (e.get("from"), e.get("to"))
                if pair in focus_pairs or (
                    "K8-09_96" in (e.get("from_members") or [])
                    and "K8-07_139" in (e.get("to_members") or [])
                ):
                    focus.append(_eb(e))
            # 補齊三邊：若 via 不同但 from/to 成員命中也收
            want = list(focus_pairs)
            have = {(e.get("from"), e.get("to")) for e in focus}
            for e in edge_pack["edges"]:
                for a, b in want:
                    if (a, b) in have:
                        continue
                    fm = e.get("from_members") or []
                    tm = e.get("to_members") or []
                    if a in fm and b in tm:
                        focus.append(_eb(e))
                        have.add((a, b))
            tids = set(top7.get("tids") or []) if top7 else set()
            pack["fake_bridge"] = {
                "focus": focus,
                "top1_has_09_96_or_07_139": ("K8-09_96" in tids)
                or ("K8-07_139" in tids),
                "top1_path": pack["M7_top1_path"],
            }

        results_by_short[short] = {
            "result": result7,
            "result_m6": result6,
            "gt_set": gt_set,
            "merge": ds["merge"],
            "edge_pack": edge_pack,
        }
        account["datasets"][short] = pack

    _write_m7_pair_stats(out_dir / "pair_score_stats_m7.md", results_by_short=results_by_short)
    report_path = args.report.resolve() if args.report else (out_dir / "comparison_m7.md")
    root_report = tp.REPO_ROOT.parent / "comparison_m7.md"
    text = _render_m7_report(account)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    account_path = out_dir / "comparison_m7_account.json"
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n報告：{report_path}")
    print(f"副本：{root_report}")
    print(f"帳本：{account_path}")
    return account


def _hyp_structure_stats(hyp: dict | None) -> dict:
    """巨路徑／碎片化檢查用：節點數、段數、段長。"""
    if not hyp:
        return {
            "n_tids": 0,
            "n_super": 0,
            "n_segments": 0,
            "n_edges": 0,
            "seg_lens": [],
            "max_seg_len": 0,
            "mean_seg_len": None,
            "is_fragmented": False,
            "is_giant": False,
            "note": "無假設",
        }
    segs = list(hyp.get("segments") or [])
    if not segs:
        labels = hyp.get("super_labels") or []
        tids = hyp.get("tids") or []
        edges = hyp.get("edges") or []
        n_super = len(labels) if labels else len(tids)
        return {
            "n_tids": len(tids),
            "n_super": n_super,
            "n_segments": int(hyp.get("n_segments") or 1),
            "n_edges": len(edges),
            "seg_lens": [n_super] if n_super else [],
            "max_seg_len": n_super,
            "mean_seg_len": float(n_super) if n_super else None,
            "is_fragmented": False,
            "is_giant": n_super >= 15,
            "note": (
                f"單段 n_super={n_super}"
                + ("（巨路徑）" if n_super >= 15 else "")
            ),
        }
    seg_lens = []
    n_edges = 0
    for seg in segs:
        labs = seg.get("super_labels") or []
        tids = seg.get("tids") or []
        seg_lens.append(len(labs) if labs else len(tids))
        n_edges += len(seg.get("edges") or [])
    n_super = sum(seg_lens)
    n_seg = len(segs)
    mean_len = (sum(seg_lens) / n_seg) if n_seg else None
    # 碎片化：多段且平均段長 ≤ 3（短段堆疊）
    is_frag = n_seg >= 2 and mean_len is not None and mean_len <= 3.0
    is_giant = n_super >= 15
    note_parts = [f"n_super={n_super}", f"n_seg={n_seg}", f"max_seg={max(seg_lens) if seg_lens else 0}"]
    if is_giant:
        note_parts.append("巨路徑★")
    if is_frag:
        note_parts.append("碎片化★（多段且平均段長≤3）")
    elif n_seg >= 2:
        note_parts.append(f"分段但未碎片化（mean_seg={mean_len:.1f}）")
    return {
        "n_tids": len(hyp.get("tids") or []),
        "n_super": n_super,
        "n_segments": n_seg,
        "n_edges": n_edges,
        "seg_lens": seg_lens,
        "max_seg_len": max(seg_lens) if seg_lens else 0,
        "mean_seg_len": mean_len,
        "is_fragmented": is_frag,
        "is_giant": is_giant,
        "note": "；".join(note_parts),
    }


def _render_m8_report(account: dict) -> str:
    lines = [
        "# M8（C + S，softmax 對裸 emb）對照實驗",
        "",
        f"生成時間：{account.get('generated_at')}",
        "",
        "> **GT 僅用於評估與著色，不參與計分。**",
        "",
        "邊分 = C + S（無 A／無 logit）；C、S 公式同 M6；"
        "競爭 softmax 改對裸 emb；建邊僅時間順序；hop 不計分。",
        "",
        "## 1. 系統層級：M0 / M6 / M7 / M8",
        "",
        "| 資料集 | 版 | precision | recall | P | n_seg | n_path | n_hit |",
        "|--------|----|-----------:|-------:|--:|------:|-------:|------:|",
    ]
    for short, pack in account["datasets"].items():
        for key in ("M0", "M6", "M7", "M8"):
            d = pack.get(key) or {}
            if not d:
                lines.append(f"| {short} | {key} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {short} | {key} | {_fmt_pct(d.get('precision', 0))} | "
                f"{_fmt_pct(d.get('recall', 0))} | {float(d.get('P') or 0):.6f} | "
                f"{d.get('n_segments')} | {d.get('n_path')} | {d.get('n_hit')} |"
            )
    lines.append("")
    for short, pack in account["datasets"].items():
        lines.append(
            f"- **{short} M8 Top-1**：`{pack.get('M8_top1_path')}`；"
            f"score={_fmt_stat(pack.get('M8_top1_score'))}"
        )
    lines.append("")

    lines += ["## 2. 巨路徑／碎片化檢查（Top-1）", ""]
    lines.append(
        "| 資料集 | 版 | n_super | n_tids | n_seg | max_seg | mean_seg | 巨路徑? | 碎片化? | 備註 |"
    )
    lines.append(
        "|--------|----|--------:|-------:|------:|--------:|---------:|:------:|:------:|------|"
    )
    for short, pack in account["datasets"].items():
        for key in ("M6", "M7", "M8"):
            st = pack.get(f"{key}_structure") or {}
            ms = st.get("mean_seg_len")
            ms_s = f"{ms:.1f}" if isinstance(ms, (int, float)) else "—"
            lines.append(
                f"| {short} | {key} | {st.get('n_super', '—')} | "
                f"{st.get('n_tids', '—')} | {st.get('n_segments', '—')} | "
                f"{st.get('max_seg_len', '—')} | {ms_s} | "
                f"{'是 ★' if st.get('is_giant') else '否'} | "
                f"{'是 ★' if st.get('is_fragmented') else '否'} | "
                f"{st.get('note', '')} |"
            )
    lines.append("")
    lines.append(
        "> 巨路徑：Top-1 超節點數 ≥ 15；碎片化：n_seg≥2 且平均段長 ≤ 3。"
    )
    lines.append("")

    lines += ["## 3. 0528 假橋三邊分解", ""]
    fb = (account["datasets"].get("0528") or {}).get("fake_bridge") or {}
    if not fb:
        lines.append("（無）")
    else:
        lines.append(
            f"- Top-1 含 09_96/07_139：**"
            f"{'是 ★' if fb.get('top1_has_09_96_or_07_139') else '否'}**"
        )
        lines.append(f"- Top-1：`{fb.get('top1_path')}`")
        lines.append("")
        lines.append(
            "| from→to | emb | C | S | Σ | n_skip | ΣP_skip | P_fwd |"
        )
        lines.append(
            "|---------|----:|--:|--:|--:|-------:|--------:|------:|"
        )
        for e in fb.get("focus") or []:
            lines.append(
                f"| `{e.get('from')}`→`{e.get('to')}` | "
                f"{_fmt_stat(e.get('emb'), 4)} | {_fmt_stat(e.get('C'))} | "
                f"{_fmt_stat(e.get('S'))} | {_fmt_stat(e.get('score'))} | "
                f"{e.get('n_skipped')} | {_fmt_stat(e.get('sum_P_skipped'), 4)} | "
                f"{_fmt_stat(e.get('P_fwd'), 4)} |"
            )
        lines.append("")
        for e in fb.get("focus") or []:
            lines.append(f"### `{e.get('from')}` → `{e.get('to')}`")
            lines.append("")
            lines.append(
                f"- emb={_fmt_stat(e.get('emb'), 4)}  "
                f"C={_fmt_stat(e.get('C'))}  S={_fmt_stat(e.get('S'))}  "
                f"Σ={_fmt_stat(e.get('score'))}"
            )
            lines.append(
                f"- n_skipped={e.get('n_skipped')}  "
                f"ΣP={_fmt_stat(e.get('sum_P_skipped'), 4)}  "
                f"P_fwd={_fmt_stat(e.get('P_fwd'), 4)}"
            )
            lines.append("")

    lines += [
        "## 4. 常數清單",
        "",
        "**`[]`（空）**",
        "",
        "---",
        "",
        "實驗約束：GT 僅評估；純增量；未改 M6/M7；未改 track_path.py；未調參。",
        "",
    ]
    return "\n".join(lines)


def cmd_compare_m8(argv=None):
    """M8（C+S，softmax 對裸 emb）對照驗證。"""
    p = argparse.ArgumentParser(description="M8 C+S 對照實驗")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m8_comparison",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0507_top1.json",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0528_top1.json",
        },
    ]

    account = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "constants": [],
        "datasets": {},
    }

    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])
        pack: dict = {"gt_set": sorted(gt_set)}

        m0 = json.loads(ds["m0_json"].read_text(encoding="utf-8"))
        pack["M0"] = _top_pack(m0, gt_set, {"source": str(ds["m0_json"])})

        results = {}
        for tag, scoring in (("M6", "m6"), ("M7", "m7"), ("M8", "m8")):
            print(f"\n===== {short} {tag} =====")
            result = run_with_config(
                ds["merge"],
                RunConfig(
                    scoring=scoring,
                    node_score=False,
                    dt_max=None,
                    sim_min=float(args.sim_min),
                    variant_tag=tag,
                ),
            )
            summary = _save_summary(
                result, ds["merge"], out_dir, f"{short}_{tag}_top1"
            )
            pack[tag] = _top_pack(summary, gt_set, {"constants": []})
            top = result["ranked"][0] if result["ranked"] else None
            pack[f"{tag}_structure"] = _hyp_structure_stats(top)
            pack[f"{tag}_top1_path"] = (
                top.get("path")
                or " -> ".join(top.get("super_labels") or top.get("tids") or [])
                if top
                else None
            )
            pack[f"{tag}_top1_score"] = top.get("score") if top else None
            results[tag] = result

        result8 = results["M8"]
        top8 = result8["ranked"][0] if result8["ranked"] else None

        # 假橋：現場建 M8 圖
        if short == "0528":
            nodes = result8["nodes"]
            tracks = result8["tracks"]
            attach_crop_embs(tracks, ds["merge"])
            succ, _, _, _ = _build_succ_m8(nodes)
            edges = []
            for items in succ:
                for _, e in items:
                    edges.append(e)
            focus_pairs = {
                ("K8-07_93", "K8-09_96"),
                ("K8-09_96", "K8-07_139"),
                ("K8-07_139", "K8-09_142"),
            }

            def _eb(e):
                return {
                    "from_super": e.get("from_super"),
                    "to_super": e.get("to_super"),
                    "from": e.get("from"),
                    "to": e.get("to"),
                    "emb": e.get("emb"),
                    "C": e.get("C"),
                    "S": e.get("S"),
                    "score": e.get("score"),
                    "n_skipped": e.get("n_skipped"),
                    "sum_P_skipped": e.get("sum_P_skipped"),
                    "P_fwd": e.get("P_fwd"),
                    "from_members": e.get("from_members"),
                    "to_members": e.get("to_members"),
                }

            focus = []
            have = set()
            for e in edges:
                pair = (e.get("from"), e.get("to"))
                if pair in focus_pairs:
                    focus.append(_eb(e))
                    have.add(pair)
            for e in edges:
                for a, b in focus_pairs:
                    if (a, b) in have:
                        continue
                    fm = e.get("from_members") or []
                    tm = e.get("to_members") or []
                    if a in fm and b in tm:
                        focus.append(_eb(e))
                        have.add((a, b))
            tids = set(top8.get("tids") or []) if top8 else set()
            pack["fake_bridge"] = {
                "focus": focus,
                "top1_has_09_96_or_07_139": ("K8-09_96" in tids)
                or ("K8-07_139" in tids),
                "top1_path": pack.get("M8_top1_path"),
            }

        account["datasets"][short] = pack

    report_path = args.report.resolve() if args.report else (out_dir / "comparison_m8.md")
    root_report = tp.REPO_ROOT.parent / "comparison_m8.md"
    text = _render_m8_report(account)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    account_path = out_dir / "comparison_m8_account.json"
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n報告：{report_path}")
    print(f"副本：{root_report}")
    print(f"帳本：{account_path}")
    return account


def _edge_effect_pack_from_succ(nodes, tracks, gt_set, succ, score_key="score"):
    edges = []
    for i, items in enumerate(succ):
        for j, e in items:
            e2 = dict(e)
            e2["_i"] = i
            e2["_j"] = j
            edges.append(e2)
    by_tid = {t.tid: t for t in tracks}
    gt_keys, same_sn, no_edge, missing, ordered = _m5_gt_edge_keys(
        nodes, edges, sorted(gt_set), by_tid
    )
    for e in edges:
        e["is_gt"] = (e["_i"], e["_j"]) in gt_keys
    gt_e = [e for e in edges if e["is_gt"]]
    ng_e = [e for e in edges if not e["is_gt"]]

    def pack_score(key):
        st_gt = _score_stats([e.get(key) for e in gt_e])
        st_ng = _score_stats([e.get(key) for e in ng_e])
        md, d = _m5_effect(st_gt, st_ng)
        return {"st_gt": st_gt, "st_ng": st_ng, "mean_diff": md, "effect_d": d}

    return {
        "n_edges": len(edges),
        "n_gt": len(gt_e),
        "n_nongt": len(ng_e),
        "same_sn": same_sn,
        "no_edge": no_edge,
        "missing": missing,
        "edges": edges,
        "gt_edges": gt_e,
        "score": pack_score(score_key),
        "components": None,
    }


def _write_m9_pair_stats(out_path: Path, *, results_by_short: dict, calib: dict) -> None:
    lines = [
        "# M9 配對分數統計表",
        "",
        "> **GT 僅用於分組評估，不參與計分。**",
        "",
        "範圍：0507/0528 Top-1～Top-3 所涉超節點之間，M9 時間序合法邊。",
        "",
        "邊分 = LLR + C + S；LLR 不乘 shrink_w。",
        "",
    ]
    for short, data in results_by_short.items():
        result = data["result"]
        gt_set = data["gt_set"]
        merge = data["merge"]
        nodes = result["nodes"]
        tracks = result["tracks"]
        attach_crop_embs(tracks, merge)
        tid_to_sn = {}
        for i, sn in enumerate(nodes):
            for tid in sn.tids:
                tid_to_sn[tid] = i
        involved = set()
        for hyp in result["ranked"][:3]:
            for tid in hyp.get("tids") or []:
                if tid in tid_to_sn:
                    involved.add(tid_to_sn[tid])
            for e in _hyp_flat_edges(hyp):
                for tid in (e.get("from_members") or []) + (e.get("to_members") or []):
                    if tid in tid_to_sn:
                        involved.add(tid_to_sn[tid])

        succ, _, _, _ = _build_succ_m9(nodes, calib)
        pairs = {"GT-GT": [], "GT-非GT": [], "非GT-非GT": []}
        for i, items in enumerate(succ):
            if i not in involved:
                continue
            for j, e in items:
                if j not in involved:
                    continue
                sa_gt = all(t in gt_set for t in nodes[i].tids)
                sb_gt = all(t in gt_set for t in nodes[j].tids)
                if sa_gt and sb_gt:
                    key = "GT-GT"
                elif sa_gt or sb_gt:
                    key = "GT-非GT"
                else:
                    key = "非GT-非GT"
                pairs[key].append(
                    {
                        "from": e.get("from_super"),
                        "to": e.get("to_super"),
                        "via": f"{e.get('from')}→{e.get('to')}",
                        "score": e.get("score"),
                        "LLR": e.get("LLR"),
                        "C": e.get("C"),
                        "S": e.get("S"),
                        "emb": e.get("emb"),
                    }
                )

        lines.append(f"## {short}")
        lines.append("")
        lines.append(f"- 涉及超節點數：{len(involved)} / 全圖 {len(nodes)}")
        lines.append("")
        for key in ("GT-GT", "GT-非GT", "非GT-非GT"):
            rows = pairs[key]
            lines.append(f"### {key}（n={len(rows)}）")
            lines.append("")
            if not rows:
                lines.append("（無合法邊）")
                lines.append("")
                continue
            st_sc = _score_stats([r["score"] for r in rows])
            st_l = _score_stats([r["LLR"] for r in rows])
            embs = [float(r["emb"]) for r in rows if r.get("emb") is not None]
            lines.append(
                f"- **LLR+C+S**：mean={_fmt_stat(st_sc.get('mean'))}  "
                f"min={_fmt_stat(st_sc.get('min'))}  max={_fmt_stat(st_sc.get('max'))}  "
                f"median={_fmt_stat(st_sc.get('median'))}"
            )
            lines.append(
                f"- **LLR 單獨**：mean={_fmt_stat(st_l.get('mean'))}  "
                f"min={_fmt_stat(st_l.get('min'))}  max={_fmt_stat(st_l.get('max'))}  "
                f"median={_fmt_stat(st_l.get('median'))}"
            )
            if embs:
                lines.append(
                    f"- **emb 原始值範圍**：[{min(embs):.4f}, {max(embs):.4f}]  "
                    f"mean={sum(embs)/len(embs):.4f}"
                )
            else:
                lines.append("- emb：—")
            lines.append("")
            lines.append(
                "| from → to | via | emb | LLR | C | S | LLR+C+S |"
            )
            lines.append(
                "|-----------|-----|----:|----:|--:|--:|--------:|"
            )
            rows_sorted = sorted(
                rows,
                key=lambda r: -(
                    _finite(r["score"])
                    if _finite(r["score"]) is not None
                    else -1e300
                ),
            )
            for r in rows_sorted:
                emb_s = (
                    f"{float(r['emb']):.4f}" if r.get("emb") is not None else "—"
                )
                lines.append(
                    f"| `{r['from']}`→`{r['to']}` | `{r['via']}` | "
                    f"{emb_s} | {_fmt_stat(r['LLR'])} | {_fmt_stat(r['C'])} | "
                    f"{_fmt_stat(r['S'])} | {_fmt_stat(r['score'])} |"
                )
            lines.append("")

    lines += ["---", "", "GT 僅評估；純輸出、未改邏輯。", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"配對統計：{out_path}")


def _render_m9_report(account: dict) -> str:
    lines = [
        "# M9（LLR + C + S）對照實驗",
        "",
        f"生成時間：{account.get('generated_at')}",
        "",
        "> **GT 僅用於評估與著色，不參與計分。**",
        "",
        "邊分 = LLR + C + S；LLR = ln(f_same(emb)/f_diff(emb))，"
        "密度取自 `calibration_gt0507.pkl`；**不乘 shrink_w**"
        "（單尺度下 w 為共同倍率）。",
        "C、S 公式同 M6；softmax 對 LLR；建邊僅時間順序；hop 不計分。",
        "",
    ]
    cal = account.get("calibration") or {}
    if cal:
        lines.append(
            f"- 校準：`{cal.get('path')}`；"
            f"emb_same=N({cal.get('same_mu')},{cal.get('same_sigma')})；"
            f"emb_diff=N({cal.get('diff_mu')},{cal.get('diff_sigma')})；"
            f"shrink_w_same={cal.get('same_shrink_w')}（未乘入）"
        )
        lines.append("")

    lines += [
        "## 1. 系統層級：M0 / M6 / M7 / M8 / M9",
        "",
        "| 資料集 | 版 | precision | recall | P | n_seg | n_path | n_hit |",
        "|--------|----|-----------:|-------:|--:|------:|-------:|------:|",
    ]
    for short, pack in account["datasets"].items():
        for key in ("M0", "M6", "M7", "M8", "M9"):
            d = pack.get(key) or {}
            if not d:
                lines.append(f"| {short} | {key} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {short} | {key} | {_fmt_pct(d.get('precision', 0))} | "
                f"{_fmt_pct(d.get('recall', 0))} | {float(d.get('P') or 0):.6f} | "
                f"{d.get('n_segments')} | {d.get('n_path')} | {d.get('n_hit')} |"
            )
    lines.append("")
    for short, pack in account["datasets"].items():
        lines.append(
            f"- **{short} M9 Top-1**：`{pack.get('M9_top1_path')}`；"
            f"score={_fmt_stat(pack.get('M9_top1_score'))}"
        )
    lines.append("")

    lines += ["### 巨路徑／碎片化檢查（Top-1）", ""]
    lines.append(
        "| 資料集 | 版 | n_super | n_tids | n_seg | max_seg | mean_seg | 巨路徑? | 碎片化? | 備註 |"
    )
    lines.append(
        "|--------|----|--------:|-------:|------:|--------:|---------:|:------:|:------:|------|"
    )
    for short, pack in account["datasets"].items():
        for key in ("M6", "M7", "M8", "M9"):
            st = pack.get(f"{key}_structure") or {}
            ms = st.get("mean_seg_len")
            ms_s = f"{ms:.1f}" if isinstance(ms, (int, float)) else "—"
            lines.append(
                f"| {short} | {key} | {st.get('n_super', '—')} | "
                f"{st.get('n_tids', '—')} | {st.get('n_segments', '—')} | "
                f"{st.get('max_seg_len', '—')} | {ms_s} | "
                f"{'是 ★' if st.get('is_giant') else '否'} | "
                f"{'是 ★' if st.get('is_fragmented') else '否'} | "
                f"{st.get('note', '')} |"
            )
    lines.append("")
    lines.append(
        "> 巨路徑：n_super≥15；碎片化：n_seg≥2 且平均段長≤3。"
    )
    lines.append("")

    lines += ["## 2. 邊層級效應量：LLR+C+S vs 前版", ""]
    for short, pack in account["datasets"].items():
        ep = pack.get("edge") or {}
        lines.append(f"### {short}")
        lines.append("")
        lines.append(
            f"- M9 邊：{ep.get('n_edges')}；GT：{ep.get('n_gt')}；"
            f"非GT：{ep.get('n_nongt')}；degenerate：{ep.get('n_degenerate')}"
        )
        lines.append("")
        lines.append(
            "| 分數 | GT mean | 非GT mean | mean差 | 效應量 d |"
        )
        lines.append("|------|--------:|----------:|-------:|--------:|")
        scores = ep.get("scores") or {}
        for key in ("LLR", "C", "S", "LLR+C+S"):
            s = scores.get(key) or {}
            stg, stn = s.get("st_gt") or {}, s.get("st_ng") or {}
            lines.append(
                f"| {key} | {_fmt_stat(stg.get('mean'))} | {_fmt_stat(stn.get('mean'))} | "
                f"{_fmt_stat(s.get('mean_diff'))} | {_fmt_stat(s.get('effect_d'))} |"
            )
        for label, key in (
            ("M6 A+C+S", "m6_ACS"),
            ("M7 logit+C+S", "m7_LCS"),
            ("M8 C+S", "m8_CS"),
        ):
            s = ep.get(key) or {}
            if not s:
                continue
            lines.append(
                f"| {label} | {_fmt_stat((s.get('st_gt') or {}).get('mean'))} | "
                f"{_fmt_stat((s.get('st_ng') or {}).get('mean'))} | "
                f"{_fmt_stat(s.get('mean_diff'))} | {_fmt_stat(s.get('effect_d'))} |"
            )
        lines.append("")

    lines += ["## 3. 0528 假橋三邊分解", ""]
    fb = (account["datasets"].get("0528") or {}).get("fake_bridge") or {}
    if not fb:
        lines.append("（無）")
    else:
        lines.append(
            f"- Top-1 含 09_96/07_139：**"
            f"{'是 ★' if fb.get('top1_has_09_96_or_07_139') else '否'}**"
        )
        lines.append(f"- Top-1：`{fb.get('top1_path')}`")
        lines.append("")
        lines.append(
            "| from→to | emb | LLR | C | S | Σ=LLR+C+S | C+S | n_skip |"
        )
        lines.append(
            "|---------|----:|----:|--:|--:|----------:|----:|-------:|"
        )
        for e in fb.get("focus") or []:
            llr = _finite(e.get("LLR"))
            c = _finite(e.get("C"))
            s = _finite(e.get("S"))
            cs = (c + s) if c is not None and s is not None else None
            lines.append(
                f"| `{e.get('from')}`→`{e.get('to')}` | "
                f"{_fmt_stat(e.get('emb'), 4)} | {_fmt_stat(e.get('LLR'))} | "
                f"{_fmt_stat(e.get('C'))} | {_fmt_stat(e.get('S'))} | "
                f"{_fmt_stat(e.get('score'))} | {_fmt_stat(cs)} | "
                f"{e.get('n_skipped')} |"
            )
        lines.append("")
        # 特別註記 07_93→09_96
        bridge = None
        for e in fb.get("focus") or []:
            if e.get("from") == "K8-07_93" and e.get("to") == "K8-09_96":
                bridge = e
                break
        if bridge:
            llr = _finite(bridge.get("LLR"))
            c = _finite(bridge.get("C"))
            s = _finite(bridge.get("S"))
            cs = (c + s) if c is not None and s is not None else None
            tot = _finite(bridge.get("score"))
            lines.append(
                "#### 焦點：`K8-07_93 → K8-09_96`（emb≈0.976，LLR 預期偏正）"
            )
            lines.append("")
            lines.append(
                f"- emb={_fmt_stat(bridge.get('emb'), 4)}；"
                f"LLR={_fmt_stat(llr)}（"
                f"{'正分 ★' if llr is not None and llr > 0 else '非正'}）"
            )
            lines.append(
                f"- C+S={_fmt_stat(cs)}；Σ={_fmt_stat(tot)}"
            )
            if llr is not None and llr > 0 and tot is not None:
                if tot <= 0:
                    verdict = "C+S **壓得住**（總分≤0）"
                elif cs is not None and cs < 0 and abs(cs) < llr:
                    verdict = "C+S 有罰但 **壓不住** LLR（總分仍正）"
                elif cs is not None and cs >= 0:
                    verdict = "C+S 未形成淨罰（甚至同號），壓不住"
                else:
                    verdict = "C+S 有罰但總分仍正——壓不住"
            else:
                verdict = "見上列數值"
            lines.append(f"- 如實判定：**{verdict}**")
            lines.append("")
        for e in fb.get("focus") or []:
            lines.append(f"### `{e.get('from')}` → `{e.get('to')}`")
            lines.append("")
            lines.append(
                f"- LLR={_fmt_stat(e.get('LLR'))}  C={_fmt_stat(e.get('C'))}  "
                f"S={_fmt_stat(e.get('S'))}  Σ={_fmt_stat(e.get('score'))}"
            )
            lines.append(
                f"- emb={_fmt_stat(e.get('emb'), 4)}  "
                f"P_fwd={_fmt_stat(e.get('P_fwd'), 4)}  "
                f"n_skip={e.get('n_skipped')}"
            )
            lines.append("")

    lines += [
        "## 4. 配對統計",
        "",
        "- `output/v1.0/m9_comparison/pair_score_stats_m9.md`",
        "",
        "## 5. 常數清單",
        "",
        "**`[]`（空）** — 校準密度為既有檔；不乘 shrink；無新手調常數。",
        "",
        "---",
        "",
        "實驗約束：GT 僅評估；純增量；未改 M6–M8；未改 track_path.py；未調參。",
        "",
    ]
    return "\n".join(lines)


def cmd_compare_m9(argv=None):
    """M9（LLR+C+S）對照驗證。"""
    p = argparse.ArgumentParser(description="M9 LLR+C+S 對照實驗")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "m9_comparison",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=tp.OUTPUT_ROOT / "v1.0" / "calibration_gt0507.pkl",
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--report", type=Path, default=None)
    args = p.parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    calib, calib_path = _load_m9_calib(args.calibration)

    datasets = [
        {
            "short": "0507",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260507.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0507_top1.json",
        },
        {
            "short": "0528",
            "merge": tp.QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528",
            "gt": tp.OUTPUT_ROOT / "v1.0" / "ground_truth_20260528.json",
            "m0_json": tp.OUTPUT_ROOT / "v1.0" / "0528_top1.json",
        },
    ]

    account = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "constants": [],
        "calibration": {
            "path": str(calib_path),
            "same_mu": float(calib["emb_same"]["mu"]),
            "same_sigma": float(calib["emb_same"]["sigma"]),
            "same_shrink_w": calib["emb_same"].get("shrink_w"),
            "diff_mu": float(calib["emb_diff"]["mu"]),
            "diff_sigma": float(calib["emb_diff"]["sigma"]),
            "note_no_shrink": (
                "LLR 不乘 shrink_w：單尺度下 w 為共同倍率，不影響排序。"
            ),
        },
        "datasets": {},
    }
    results_by_short = {}

    for ds in datasets:
        short = ds["short"]
        gt_set = set(json.loads(ds["gt"].read_text(encoding="utf-8"))["person_tids"])
        pack: dict = {"gt_set": sorted(gt_set)}
        m0 = json.loads(ds["m0_json"].read_text(encoding="utf-8"))
        pack["M0"] = _top_pack(m0, gt_set, {"source": str(ds["m0_json"])})

        results = {}
        for tag, scoring in (
            ("M6", "m6"),
            ("M7", "m7"),
            ("M8", "m8"),
            ("M9", "m9"),
        ):
            print(f"\n===== {short} {tag} =====")
            result = run_with_config(
                ds["merge"],
                RunConfig(
                    scoring=scoring,
                    node_score=False,
                    dt_max=None,
                    sim_min=float(args.sim_min),
                    variant_tag=tag,
                    calibration_path=str(calib_path) if scoring == "m9" else None,
                ),
            )
            summary = _save_summary(
                result, ds["merge"], out_dir, f"{short}_{tag}_top1"
            )
            pack[tag] = _top_pack(summary, gt_set, {"constants": []})
            top = result["ranked"][0] if result["ranked"] else None
            pack[f"{tag}_structure"] = _hyp_structure_stats(top)
            pack[f"{tag}_top1_path"] = (
                top.get("path")
                or " -> ".join(top.get("super_labels") or top.get("tids") or [])
                if top
                else None
            )
            pack[f"{tag}_top1_score"] = top.get("score") if top else None
            results[tag] = result

        result9 = results["M9"]
        nodes = result9["nodes"]
        tracks = result9["tracks"]
        attach_crop_embs(tracks, ds["merge"])
        succ9, _, n_legal, m9_meta = _build_succ_m9(nodes, calib)
        edges9 = []
        for i, items in enumerate(succ9):
            for j, e in items:
                e2 = dict(e)
                e2["_i"] = i
                e2["_j"] = j
                e2["LCS"] = e.get("score")
                edges9.append(e2)
        by_tid = {t.tid: t for t in tracks}
        gt_keys, same_sn, no_edge, missing, _ = _m5_gt_edge_keys(
            nodes, edges9, sorted(gt_set), by_tid
        )
        for e in edges9:
            e["is_gt"] = (e["_i"], e["_j"]) in gt_keys
        gt_e = [e for e in edges9 if e["is_gt"]]
        ng_e = [e for e in edges9 if not e["is_gt"]]

        def pack_score(key, edges=None):
            src_gt = gt_e if edges is None else [e for e in edges if e["is_gt"]]
            src_ng = ng_e if edges is None else [e for e in edges if not e["is_gt"]]
            st_gt = _score_stats([e.get(key) for e in src_gt])
            st_ng = _score_stats([e.get(key) for e in src_ng])
            md, d = _m5_effect(st_gt, st_ng)
            return {"st_gt": st_gt, "st_ng": st_ng, "mean_diff": md, "effect_d": d}

        edge_pack = {
            "n_nodes": len(nodes),
            "n_edges": n_legal,
            "n_degenerate": m9_meta.get("n_degenerate"),
            "n_gt": len(gt_e),
            "n_nongt": len(ng_e),
            "same_sn": same_sn,
            "no_edge": no_edge,
            "missing": missing,
            "scores": {
                "LLR": pack_score("LLR"),
                "C": pack_score("C"),
                "S": pack_score("S"),
                "LLR+C+S": pack_score("LCS"),
            },
            "edges": edges9,
        }

        # 前版效應量（各用自己的 succ）
        for tag, builder, skey, out_key in (
            ("M6", lambda n: _build_succ_m6(n)[0], "score", "m6_ACS"),
            ("M7", lambda n: _build_succ_m7(n)[0], "score", "m7_LCS"),
            ("M8", lambda n: _build_succ_m8(n)[0], "score", "m8_CS"),
        ):
            res = results[tag]
            attach_crop_embs(res["tracks"], ds["merge"])
            nd = res["nodes"]
            succ = builder(nd)
            ep = _edge_effect_pack_from_succ(
                nd, res["tracks"], gt_set, succ, score_key=skey
            )
            edge_pack[out_key] = {
                "n_edges": ep["n_edges"],
                "n_gt": ep["n_gt"],
                "n_nongt": ep["n_nongt"],
                **ep["score"],
            }

        pack["edge"] = {
            k: v for k, v in edge_pack.items() if k not in ("edges",)
        }

        if short == "0528":
            focus_pairs = {
                ("K8-07_93", "K8-09_96"),
                ("K8-09_96", "K8-07_139"),
                ("K8-07_139", "K8-09_142"),
            }

            def _eb(e):
                return {
                    "from": e.get("from"),
                    "to": e.get("to"),
                    "from_super": e.get("from_super"),
                    "to_super": e.get("to_super"),
                    "emb": e.get("emb"),
                    "LLR": e.get("LLR"),
                    "C": e.get("C"),
                    "S": e.get("S"),
                    "score": e.get("score"),
                    "n_skipped": e.get("n_skipped"),
                    "sum_P_skipped": e.get("sum_P_skipped"),
                    "P_fwd": e.get("P_fwd"),
                    "from_members": e.get("from_members"),
                    "to_members": e.get("to_members"),
                }

            focus = []
            have = set()
            for e in edges9:
                pair = (e.get("from"), e.get("to"))
                if pair in focus_pairs:
                    focus.append(_eb(e))
                    have.add(pair)
            for e in edges9:
                for a, b in focus_pairs:
                    if (a, b) in have:
                        continue
                    if a in (e.get("from_members") or []) and b in (
                        e.get("to_members") or []
                    ):
                        focus.append(_eb(e))
                        have.add((a, b))
            top9 = result9["ranked"][0] if result9["ranked"] else None
            tids = set(top9.get("tids") or []) if top9 else set()
            pack["fake_bridge"] = {
                "focus": focus,
                "top1_has_09_96_or_07_139": ("K8-09_96" in tids)
                or ("K8-07_139" in tids),
                "top1_path": pack.get("M9_top1_path"),
            }

        results_by_short[short] = {
            "result": result9,
            "gt_set": gt_set,
            "merge": ds["merge"],
        }
        account["datasets"][short] = pack

    _write_m9_pair_stats(
        out_dir / "pair_score_stats_m9.md",
        results_by_short=results_by_short,
        calib=calib,
    )
    report_path = args.report.resolve() if args.report else (out_dir / "comparison_m9.md")
    root_report = tp.REPO_ROOT.parent / "comparison_m9.md"
    text = _render_m9_report(account)
    report_path.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    account_path = out_dir / "comparison_m9_account.json"
    account_path.write_text(
        json.dumps(account, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n報告：{report_path}")
    print(f"副本：{root_report}")
    print(f"帳本：{account_path}")
    return account


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return cmd_compare_m2([])
    cmd = argv[0]
    rest = argv[1:]
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "compare":
        return cmd_compare(rest)
    if cmd == "compare_m2":
        return cmd_compare_m2(rest)
    if cmd == "compare_m3":
        return cmd_compare_m3(rest)
    if cmd == "compare_m4":
        return cmd_compare_m4(rest)
    if cmd == "compare_m4b":
        return cmd_compare_m4b(rest)
    if cmd == "compare_m5":
        return cmd_compare_m5(rest)
    if cmd == "compare_m6":
        return cmd_compare_m6(rest)
    if cmd == "compare_m7":
        return cmd_compare_m7(rest)
    if cmd == "compare_m8":
        return cmd_compare_m8(rest)
    if cmd == "compare_m9":
        return cmd_compare_m9(rest)
    if cmd == "inspect_m6":
        return cmd_inspect_m6(rest)
    if cmd == "w_distribution":
        return cmd_w_distribution(rest)
    if cmd == "validate_emb_edge":
        return cmd_validate_emb_edge(rest)
    if cmd.startswith("-"):
        return cmd_compare_m2(argv)
    raise SystemExit(
        f"未知子命令：{cmd}（run | compare | compare_m2 | compare_m3 | "
        f"compare_m4 | compare_m4b | compare_m5 | compare_m6 | compare_m7 | "
        f"compare_m8 | compare_m9 | inspect_m6 | w_distribution | "
        f"validate_emb_edge）"
    )


def _render_m6_report(account: dict) -> str:
    lines = [
        "# M6（A + C + S）對照實驗",
        "",
        f"生成時間：{account.get('generated_at')}",
        "",
        "> **GT 僅用於評估與著色，不參與計分。**",
        "",
        "邊分 = A + C + S；S = ln(1 − Σ P(w|u))（t_start(w) < t_start(v)）；"
        "建邊僅時間順序；hop 不計分；Σ≥1 → degenerate 不採用。",
        "",
        "## 1. 系統層級：M0 / M4 / M6",
        "",
        "| 資料集 | 版 | precision | recall | P | n_seg | n_path | n_hit | min-A |",
        "|--------|----|-----------:|-------:|--:|------:|-------:|------:|------:|",
    ]
    for short, pack in account["datasets"].items():
        for key in ("M0", "M4", "M6"):
            d = pack.get(key) or {}
            ma = "—"
            if key == "M6":
                ma = _fmt_stat(pack.get("M6_top1_min_A"))
            if not d:
                lines.append(f"| {short} | {key} | — | — | — | — | — | — | {ma} |")
                continue
            lines.append(
                f"| {short} | {key} | {_fmt_pct(d.get('precision', 0))} | "
                f"{_fmt_pct(d.get('recall', 0))} | {float(d.get('P') or 0):.6f} | "
                f"{d.get('n_segments')} | {d.get('n_path')} | {d.get('n_hit')} | {ma} |"
            )
        be = pack.get("M6_best_edged") or {}
        if be and pack.get("M6_top1_is_singleton"):
            lines.append(
                f"| {short} | M6≥1邊（Top-1 單節點） | "
                f"{_fmt_pct(be.get('precision', 0))} | {_fmt_pct(be.get('recall', 0))} | "
                f"{float(be.get('P') or 0):.6f} | — | {be.get('n_path')} | "
                f"{be.get('n_hit')} | {_fmt_stat(be.get('min_A'))} |"
            )
    lines.append("")
    lines.append("### 單節點壓制？")
    lines.append("")
    for short, pack in account["datasets"].items():
        lines.append(
            f"- **{short}**：Top-1 邊數={pack.get('M6_top1_n_edges')}；"
            f"單節點={'是 ★' if pack.get('M6_top1_is_singleton') else '否'}；"
            f"路徑=`{pack.get('M6_top1_path')}`"
        )
    lines.append("")

    lines += ["## 2. 邊層級：S / A+C+S vs M4(A+C−M)", ""]
    for short, pack in account["datasets"].items():
        ep = pack.get("edge") or {}
        lines.append(f"### {short}")
        lines.append("")
        lines.append(
            f"- 超節點：{ep.get('n_nodes')}；時間序邊：{ep.get('n_edges')}；"
            f"degenerate：{ep.get('n_degenerate')}；"
            f"GT 邊：{ep.get('n_gt')}；非 GT：{ep.get('n_nongt')}"
        )
        lines.append("")
        lines.append(
            "| 分數 | GT mean | 非GT mean | mean差 | 效應量 d |"
        )
        lines.append("|------|--------:|----------:|-------:|--------:|")
        scores = ep.get("scores") or {}
        for key in ("S", "A", "C", "A+C+S"):
            s = scores.get(key) or {}
            stg, stn = s.get("st_gt") or {}, s.get("st_ng") or {}
            lines.append(
                f"| {key} | {_fmt_stat(stg.get('mean'))} | {_fmt_stat(stn.get('mean'))} | "
                f"{_fmt_stat(s.get('mean_diff'))} | {_fmt_stat(s.get('effect_d'))} |"
            )
        m4 = ep.get("m4_ACM")
        if m4:
            lines.append(
                f"| M4 A+C−M | {_fmt_stat((m4.get('st_gt') or {}).get('mean'))} | "
                f"{_fmt_stat((m4.get('st_ng') or {}).get('mean'))} | "
                f"{_fmt_stat(m4.get('mean_diff'))} | {_fmt_stat(m4.get('effect_d'))} |"
            )
        lines.append("")
        lines.append(
            "#### 教授情境：跳過 ≥2 個但 S 罰款 < 0.2 的 GT 邊"
            "（不像者被跳過不重罰）"
        )
        lines.append("")
        soft = pack.get("soft_skip_gt") or []
        if not soft:
            lines.append("（無）")
        else:
            lines.append(
                "| from→to | n_skip | ΣP_skip | S | −S | A | C | Σ |"
            )
            lines.append(
                "|---------|-------:|--------:|--:|---:|--:|--:|--:|"
            )
            for e in soft:
                S = e.get("S")
                pen = (-float(S)) if S is not None and _finite(S) is not None else None
                lines.append(
                    f"| `{e.get('from_super')}`→`{e.get('to_super')}` | "
                    f"{e.get('n_skipped')} | {_fmt_stat(e.get('sum_P_skipped'), 4)} | "
                    f"{_fmt_stat(S)} | {_fmt_stat(pen)} | {_fmt_stat(e.get('A'))} | "
                    f"{_fmt_stat(e.get('C'))} | {_fmt_stat(e.get('score'))} |"
                )
        lines.append("")

    lines += ["## 3. 0528 假橋：S 殺不殺得死", ""]
    fb = (account["datasets"].get("0528") or {}).get("fake_bridge") or {}
    if not fb:
        lines.append("（無）")
    else:
        lines.append(
            f"- Top-1 含 09_96/07_139：**"
            f"{'是 ★' if fb.get('top1_has_09_96_or_07_139') else '否'}**"
        )
        lines.append(f"- Top-1：`{fb.get('top1_path')}`")
        lines.append("")
        focus = fb.get("focus") or []
        if not focus:
            lines.append("（焦點邊未出現在 M6 合法圖——可能 degenerate 或時間序不符）")
        for e in focus:
            lines.append(
                f"### `{e.get('from')}` → `{e.get('to')}`"
            )
            lines.append("")
            lines.append(
                f"- A={_fmt_stat(e.get('A'))}  C={_fmt_stat(e.get('C'))}  "
                f"S={_fmt_stat(e.get('S'))}  Σ={_fmt_stat(e.get('score'))}"
            )
            lines.append(
                f"- n_skipped={e.get('n_skipped')}  "
                f"ΣP_skipped={_fmt_stat(e.get('sum_P_skipped'), 4)}  "
                f"P_fwd={_fmt_stat(e.get('P_fwd'), 4)}"
            )
            S = _finite(e.get("S"))
            if S is not None and S <= -1.0:
                verdict = "S 重罰（≤−1）——有殺傷力"
            elif S is not None and S < -0.2:
                verdict = "S 中度罰款"
            elif S is not None:
                verdict = "S 幾乎不罰——**殺不死**（如實）"
            else:
                verdict = "S 不可用"
            lines.append(f"- 判定：**{verdict}**")
            lines.append("")
            lines.append("被跳過者 P 明細（按 P 降序，最多 8）：")
            lines.append("")
            lines.append("| to_super | t_start | P | A |")
            lines.append("|----------|--------:|--:|--:|")
            for sk in e.get("skipped_top") or []:
                lines.append(
                    f"| `{sk.get('to_super')}` | {sk.get('t_start', 0):.1f} | "
                    f"{_fmt_stat(sk.get('P'), 4)} | {_fmt_stat(sk.get('A'))} |"
                )
            lines.append("")

    lines += ["## 4. Top-1 vs 次名：min-A 對照", ""]
    lines.append("| 資料集 | 名次 | path | score | P | min-A |")
    lines.append("|--------|------|------|------:|--:|------:|")
    for short, pack in account["datasets"].items():
        lines.append(
            f"| {short} | Top-1 | `{pack.get('M6_top1_path')}` | "
            f"{_fmt_stat(pack.get('M6_top1_score'))} | "
            f"{float(pack.get('M6_top1_P') or 0):.6f} | "
            f"{_fmt_stat(pack.get('M6_top1_min_A'))} |"
        )
        r = pack.get("M6_runner") or {}
        lines.append(
            f"| {short} | 次名 | `{r.get('path')}` | "
            f"{_fmt_stat(r.get('score'))} | "
            f"{float(r.get('P') or 0):.6f} | {_fmt_stat(r.get('min_A'))} |"
        )
    lines.append("")
    lines.append("> min-A 僅附欄，**不參與排名**。")
    lines.append("")

    lines += [
        "## 5. 常數清單",
        "",
        "**`[]`（空）**",
        "",
        "## 6. 拼圖",
        "",
        "- `output/v1.0/m6_comparison/人員追蹤_20260507_m6_top1_collage.png`",
        "- `output/v1.0/m6_comparison/人員追蹤_20260528_m6_top1_collage.png`",
        "",
        "---",
        "",
        "實驗約束：GT 僅評估；未改既有模式；未改 track_path.py；未調參。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
