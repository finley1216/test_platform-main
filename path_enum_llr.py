# -*- coding: utf-8 -*-
"""
跨鏡頭路徑枚舉 + 對數似然比（LLR）計分
======================================
硬規則複用 path_enum_scoring.py 的 load_tracks / edge_check（經 runtime 覆寫門檻），
不修改 path_enum_scoring.py 原文。

結構修正（2026-07-15，附依據）：
  1. emb 硬門檻 → 0.80（llr_gate_config；鑑別交給 LLR_emb）
  2. 共存超節點：時間重疊≥0.5s 且（OVERLAP 或 H×dH<80）→ union-find
  3. 節點證據 = w·ln(P(sim|GT)/P(sim|nonGT))；transit dt 先驗 σ=1.0（PRIOR-WEAK）
  4. MIN_TRANSIT hop1→0（相鄰視野邊界相接，無辯護下界）；hop2 維持 6s
  5. 分段軌跡／排名：單路徑與多段假設進同一排名池（計分公式不變，只改誰跟誰比）
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import path_enum_scoring as pes  # noqa: E402
import llr_gate_config as gates  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

PDF_FLOOR = 1e-12
SHRINK_K = 10.0
HANDOFF_DT_MAX = 2.0
SUPER_OVERLAP_MIN = 0.5
SUPER_DH_MAX = 80.0
PRIOR_DT_SIGMA = 1.0  # PRIOR-WEAK（原 0.5）

NODE_EVIDENCE_NOTE = (
    "節點證據 = w·ln(P(sim|GT)/P(sim|nonGT))，w=n/(n+10)；"
    "需 calibration 含 sim_gt / sim_nongt"
)
PRIOR_WEAK_NOTE = (
    f"PRIOR-WEAK：dt 退先驗時 LogNormal(σ={PRIOR_DT_SIGMA})，tau 仍佔位；收縮 w=1"
)
SEGMENT_RANK_NOTE = (
    "2026-07-15：分段假設參與排名（總分=各段分數和）；"
    "同一 Softmax 池；假設內部共存矛盾（非 OVERLAP/非 ADJACENT 同時出現）作廢；"
    "計分公式不動"
)
# 自高分極大路徑取前 K 條當作多段假設的 seed（其餘仍以單段假設進排名池）
DEFAULT_SEGMENT_SEED_TOP_K = 400
DEFAULT_MAX_HYP_SEGMENTS = 8


# ---------------------------------------------------------------------------
# Calibration / LLR densities
# ---------------------------------------------------------------------------

def load_calibration(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _pdf(dist: dict, x: float) -> float:
    fam = dist["family"]
    if fam == "norm":
        p = float(stats.norm.pdf(x, loc=dist["mu"], scale=dist["sigma"]))
    elif fam == "lognorm":
        xx = max(float(x), 1e-6)
        p = float(stats.lognorm.pdf(xx, dist["sigma"], loc=0.0, scale=math.exp(dist["mu"])))
    elif fam == "halfnorm":
        p = float(stats.halfnorm.pdf(max(float(x), 0.0), loc=0.0, scale=dist["sigma"]))
    elif fam == "uniform":
        lo, hi = float(dist["low"]), float(dist["high"])
        if hi <= lo:
            p = 0.0
        elif lo <= x <= hi:
            p = 1.0 / (hi - lo)
        else:
            p = 0.0
    else:
        raise ValueError(f"unknown family: {fam}")
    return max(p, PDF_FLOOR)


def llr(dist_same: dict, dist_diff: dict, x: float) -> float:
    return math.log(_pdf(dist_same, x)) - math.log(_pdf(dist_diff, x))


def shrink_weight(n: int | None, *, force_full: bool = False) -> float:
    if force_full:
        return 1.0
    nn = max(int(n or 0), 0)
    return float(nn) / float(nn + SHRINK_K)


def node_evidence(sim: float, calib: dict | None) -> dict:
    """回傳 {raw, w, score, enabled}。無 sim_gt/sim_nongt 時 score=0。"""
    if not calib or "sim_gt" not in calib or "sim_nongt" not in calib:
        return {"raw": 0.0, "w": 0.0, "score": 0.0, "enabled": False, "sim": float(sim)}
    raw = llr(calib["sim_gt"], calib["sim_nongt"], float(sim))
    # 收縮用 GT 樣本數
    w = shrink_weight(calib["sim_gt"].get("n"))
    return {"raw": raw, "w": w, "score": w * raw, "enabled": True, "sim": float(sim)}


def _dt_same_dist(calib: dict, cam_u: str, cam_v: str) -> dict:
    key = "|".join(sorted((cam_u, cam_v)))
    d = calib["dt_same_by_pair"].get(key)
    if d is not None:
        # 先驗檔若仍寫 sigma=0.5，執行時改為 PRIOR-WEAK sigma
        if d.get("prior") and float(d.get("sigma", PRIOR_DT_SIGMA)) < PRIOR_DT_SIGMA - 1e-9:
            d = dict(d)
            d["sigma"] = PRIOR_DT_SIGMA
            d["prior_weak"] = True
        return d
    hop = pes.hop_count(cam_u, cam_v)
    if hop is None:
        hop = 1
    tau0 = float(pes.tau(cam_u, cam_v, hop))
    return {
        "family": "lognorm",
        "mu": math.log(max(tau0, 1e-3)),
        "sigma": PRIOR_DT_SIGMA,
        "prior": True,
        "prior_weak": True,
        "n": 0,
        "tau": tau0,
    }


def _is_tau_placeholder(tau0: float | None) -> bool:
    if tau0 is None:
        return True
    return abs(float(tau0) - float(pes.DEFAULT_TAU_HOP1)) < 1e-9


def _dt_weight_and_note(dist_same: dict) -> tuple[float, bool, str | None]:
    n = int(dist_same.get("n") or 0)
    is_prior = bool(dist_same.get("prior"))
    tau0 = dist_same.get("tau")
    if not is_prior and n >= 20:
        return 1.0, False, None
    note = "PRIOR-WEAK"
    if _is_tau_placeholder(tau0 if tau0 is not None else math.exp(float(dist_same.get("mu", 0.0)))):
        note = f"PRIOR-WEAK (tau placeholder=8.0, sigma={PRIOR_DT_SIGMA})"
    return 1.0, True, note


def is_handoff_edge(u: pes.Track, v: pes.Track, dt: float, h_dist: float | None) -> bool:
    if float(dt) > HANDOFF_DT_MAX:
        return False
    key = tuple(sorted((u.cam, v.cam)))
    if key in pes.OVERLAP_PAIRS:
        return True
    return h_dist is not None


def edge_llr(
    u: pes.Track,
    v: pes.Track,
    dt: float,
    emb: float,
    h_dist: float | None,
    calib: dict,
    *,
    dt_scoring: bool = True,
    transition_prior: bool = False,
) -> dict:
    """
    dt_scoring=False：transit 邊 LLR_dt 記 0、顯示 "removed"（硬規則不受影響）。
    transition_prior=True：每邊加 ln(p_edge)（見 calib['transition_prior']）。
    """
    dt_dist = _dt_same_dist(calib, u.cam, v.cam)
    raw_dt = llr(dt_dist, calib["dt_diff"], dt)
    raw_emb = llr(calib["emb_same"], calib["emb_diff"], emb)
    raw_dh = None
    if h_dist is not None:
        raw_dh = llr(calib["dh_same"], calib["dh_diff"], h_dist)

    handoff = is_handoff_edge(u, v, dt, h_dist)
    dt_model = "handoff" if handoff else "transit"
    w_emb = shrink_weight(calib["emb_same"].get("n"))
    w_dh = shrink_weight(calib["dh_same"].get("n")) if raw_dh is not None else None
    w_dt, dt_prior_physical, dt_note = _dt_weight_and_note(dt_dist)

    trans_llr = 0.0
    if transition_prior:
        tp = calib.get("transition_prior") or {}
        p_edge = tp.get("p_edge")
        if p_edge is not None and float(p_edge) > 0.0:
            trans_llr = math.log(float(p_edge))

    if handoff:
        eff_dt = 0.0
        llr_dt_display = None
        raw_total_for_edge = raw_emb + (raw_dh if raw_dh is not None else 0.0)
        eff_emb = w_emb * raw_emb
        eff_dh = (w_dh * raw_dh) if (raw_dh is not None and w_dh is not None) else None
        eff_total = eff_emb + (eff_dh if eff_dh is not None else 0.0) + trans_llr
        dt_note_out = None
    else:
        if dt_scoring:
            eff_dt = w_dt * raw_dt
            llr_dt_display = eff_dt
            dt_note_out = dt_note
            raw_total_for_edge = raw_dt + raw_emb + (raw_dh if raw_dh is not None else 0.0)
        else:
            # 軟證據停用；仍保留 raw 供對照。依據：tau 無實測來源（2026-07-15）
            eff_dt = 0.0
            llr_dt_display = "removed"
            dt_note_out = "dt-scoring=off (2026-07-15: tau 無實測，軟 dt 停用)"
            raw_total_for_edge = raw_emb + (raw_dh if raw_dh is not None else 0.0)
        eff_emb = w_emb * raw_emb
        eff_dh = (w_dh * raw_dh) if (raw_dh is not None and w_dh is not None) else None
        eff_total = eff_dt + eff_emb + (eff_dh if eff_dh is not None else 0.0) + trans_llr

    denom = raw_total_for_edge if abs(raw_total_for_edge) > 1e-12 else 1.0
    # w 不含 transition 常數項，避免除法扭曲
    core = eff_total - trans_llr
    w_edge = core / denom if abs(denom) > 1e-12 else 1.0
    return {
        "from": u.tid,
        "to": v.tid,
        "dt": dt,
        "emb": emb,
        "h_dist": h_dist,
        "dt_model": dt_model,
        "LLR_dt": llr_dt_display,
        "LLR_dt_raw": raw_dt,
        "w_dt": None if handoff or not dt_scoring else w_dt,
        "LLR_emb": eff_emb,
        "LLR_emb_raw": raw_emb,
        "w_emb": w_emb,
        "LLR_dH": eff_dh,
        "LLR_dH_raw": raw_dh,
        "w_dH": w_dh,
        "LLR_transition": trans_llr if transition_prior else None,
        "raw_LLR": raw_total_for_edge,
        "w": w_edge,
        "effective_LLR": eff_total,
        "score": eff_total,
        "dt_prior_physical": bool(dt_prior_physical) and not handoff and dt_scoring,
        "dt_note": dt_note_out,
        "dt_scoring": bool(dt_scoring),
        "transition_prior": bool(transition_prior),
    }


# ---------------------------------------------------------------------------
# 修正二：共存超節點
# ---------------------------------------------------------------------------

@dataclass
class SuperNode:
    sid: str
    members: list  # pes.Track, sorted by t_start
    emb: np.ndarray = field(default=None)
    sim: float = 0.0
    t_start: float = 0.0
    t_end: float = 0.0
    cams: list = field(default_factory=list)

    @property
    def tids(self) -> list[str]:
        return [t.tid for t in self.members]

    @property
    def label(self) -> str:
        if len(self.members) == 1:
            return self.members[0].tid
        return "{" + ",".join(self.tids) + "}"


def _time_overlap_sec(a: pes.Track, b: pes.Track) -> float:
    return min(a.t_end, b.t_end) - max(a.t_start, b.t_start)


def _coexistence_time_ok(u: pes.Track, v: pes.Track) -> tuple[bool, float, str]:
    """
    一般：重疊 ≥ 0.5s。
    退化：任一方時長 ≈0 且時間點落在對方 [t_start,t_end] 內
    （如 K8-22_19 單幀）→ 視為時間共存，否則 OVERLAP/H 對永遠併不進去。
    """
    ov = _time_overlap_sec(u, v)
    if ov >= SUPER_OVERLAP_MIN:
        return True, ov, f"overlap={ov:.2f}s"
    if ov < -1e-9:
        return False, ov, f"overlap={ov:.2f}s"
    # ov ∈ [0, 0.5)：檢查單幀／點時刻落在對方區間
    for short, long in ((u, v), (v, u)):
        dur = float(short.t_end - short.t_start)
        if dur <= 1e-6:
            t0 = float(short.t_start)
            if long.t_start - 1e-9 <= t0 <= long.t_end + 1e-9:
                return True, max(ov, 0.0), f"point_in_span t={t0:.2f} host={long.tid}"
    return False, ov, f"overlap={ov:.2f}s"


def coexistence_merge(u: pes.Track, v: pes.Track) -> tuple[bool, str]:
    """同鏡跳過；時間共存且（OVERLAP 或 H×dH<80）。"""
    if u.cam == v.cam:
        return False, "same_cam_skip"
    tok, ov, tnote = _coexistence_time_ok(u, v)
    if not tok:
        return False, f"no_coexist_time ({tnote})"
    key = tuple(sorted((u.cam, v.cam)))
    if key in pes.OVERLAP_PAIRS:
        return True, f"OVERLAP_PAIRS {tnote}"
    ok_h, d = pes.same_object_h(u, v)
    if ok_h and d is not None and float(d) < SUPER_DH_MAX:
        return True, f"H dH={d:.1f}px {tnote}"
    return False, f"no_overlap_or_H ({tnote})"


def build_supernodes(tracks: list) -> tuple[list[SuperNode], dict]:
    n = len(tracks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    merge_log = []
    for i, j in itertools.combinations(range(n), 2):
        ok, reason = coexistence_merge(tracks[i], tracks[j])
        if ok:
            union(i, j)
            merge_log.append(
                {
                    "a": tracks[i].tid,
                    "b": tracks[j].tid,
                    "reason": reason,
                }
            )

    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(tracks[i])

    supers: list[SuperNode] = []
    for k, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: min(t.t_start for t in kv[1]))):
        members = sorted(members, key=lambda t: (t.t_start, t.t_end, t.tid))
        embs = np.stack([t.emb for t in members], axis=0)
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
        sn = SuperNode(
            sid=f"SN{k}",
            members=members,
            emb=mean_emb,
            sim=float(np.mean([t.sim for t in members])),
            t_start=float(min(t.t_start for t in members)),
            t_end=float(max(t.t_end for t in members)),
            cams=sorted({t.cam for t in members}),
        )
        supers.append(sn)

    report = {
        "n_tracks": n,
        "n_supernodes": len(supers),
        "n_merged_pairs": len(merge_log),
        "merge_log": merge_log,
        "supernodes": [
            {
                "sid": s.sid,
                "members": s.tids,
                "cams": s.cams,
                "t_start": s.t_start,
                "t_end": s.t_end,
                "sim_mean": s.sim,
                "multi": len(s.members) > 1,
            }
            for s in supers
        ],
        "multi_only": [
            s.tids for s in supers if len(s.members) > 1
        ],
    }
    return supers, report


def verify_mislabel_not_in_gt_super(
    super_report: dict, gt_tids: set[str], mislabel: str = "K8-08_43"
) -> dict:
    bad = []
    for sn in super_report["supernodes"]:
        mem = set(sn["members"])
        if mislabel in mem and (mem & gt_tids):
            bad.append(sn)
    return {
        "mislabel": mislabel,
        "merged_with_gt": bool(bad),
        "offending_supernodes": bad,
        "note": (
            "23↔08 無 OVERLAP、無 H，規則上不應與 23_8 合併；"
            "若併入其他 GT 超節點亦屬失敗。"
        ),
    }


def _best_member_edge(sa: SuperNode, sb: SuperNode):
    """
    超節點間合法邊：
      - dt 一律用聯集端點：dt = max(sb.t_start − sa.t_end, 0)
        （MIN_TRANSIT / DT_MAX / 時間順序容許 / handoff 皆用此 dt）
      - emb 取成員對最大值；hop／H 依該成員對鏡頭
    回傳 (u, v, dt_union, hop, emb, h_dist) 或 (None, rejects)。
    """
    dt_raw = float(sb.t_start - sa.t_end)
    dt = max(dt_raw, 0.0)
    best = None
    rejects = []
    for u in sa.members:
        for v in sb.members:
            if v.t_end < u.t_start - pes.DT_MAX and sb.t_end < sa.t_start - pes.DT_MAX:
                continue
            key = tuple(sorted((u.cam, v.cam)))
            tol = pes.OVERLAP_PAIRS.get(key, pes.TOL)
            h_ok, h_dist = pes.same_object_h(u, v)

            # 時間順序：以聯集 dt_raw 為準
            if dt_raw < -tol:
                if not (h_ok or pes.corridor_prefers(u, v)):
                    rejects.append(
                        (
                            u.tid,
                            v.tid,
                            f"時間順序（聯集重疊 {-dt_raw:.1f}s 超過容許 {tol:.1f}s）",
                        )
                    )
                    continue

            hop = pes.hop_count(u.cam, v.cam)
            if hop is None:
                if h_ok and tuple(sorted((u.cam, v.cam))) in pes.ADJACENT:
                    hop = 1
                else:
                    rejects.append((u.tid, v.tid, "拓撲不可達"))
                    continue

            mt = pes.min_transit(u.cam, v.cam, hop, h_ok=h_ok)
            if dt < mt:
                rejects.append(
                    (u.tid, v.tid, f"瞬移（聯集dt={dt:.1f}s < 最短通行 {mt:.1f}s）")
                )
                continue
            if dt > pes.DT_MAX:
                rejects.append(
                    (u.tid, v.tid, f"斷太久（聯集dt={dt:.1f}s > DT_MAX）")
                )
                continue

            emb = pes.emb_sim(u, v)
            emb_need = pes.EMB_EDGE_MIN - 0.02 if h_ok else pes.EMB_EDGE_MIN
            if emb < emb_need:
                rejects.append(
                    (u.tid, v.tid, f"外觀不像（emb={emb:.3f} < {emb_need}）")
                )
                continue

            cand = (u, v, dt, hop, emb, h_dist)
            if best is None or emb > best[4]:
                best = cand
    return best, rejects


# emb=0.80 時合法邊可 >200，全量 DFS 不可行；改 beam。
# 邊數 ≤ 此值時仍全枚舉（含前綴，與舊行為一致）。
FULL_ENUM_EDGE_CAP = 80
DEFAULT_BEAM_WIDTH = 64
DEFAULT_BEAM_MAX_LEAVES = 5000


def _build_nodes(tracks: list, use_supernode: bool):
    if use_supernode:
        return build_supernodes(tracks)
    nodes = [
        SuperNode(
            sid=t.tid,
            members=[t],
            emb=t.emb,
            sim=t.sim,
            t_start=t.t_start,
            t_end=t.t_end,
            cams=[t.cam],
        )
        for t in tracks
    ]
    super_report = {
        "n_tracks": len(tracks),
        "n_supernodes": len(nodes),
        "n_merged_pairs": 0,
        "merge_log": [],
        "supernodes": [
            {
                "sid": s.sid,
                "members": s.tids,
                "cams": s.cams,
                "t_start": s.t_start,
                "t_end": s.t_end,
                "sim_mean": s.sim,
                "multi": False,
            }
            for s in nodes
        ],
        "multi_only": [],
        "disabled": True,
    }
    return nodes, super_report


def _build_succ(nodes: list) -> tuple[list, list, int]:
    n = len(nodes)
    succ = [[] for _ in range(n)]
    rejected_edges = []
    n_legal_edges = 0
    for i, j in itertools.permutations(range(n), 2):
        sa, sb = nodes[i], nodes[j]
        if sb.t_end < sa.t_start - pes.DT_MAX:
            continue
        best, rejects = _best_member_edge(sa, sb)
        if best is not None:
            u, v, dt, hop, emb, h_dist = best
            succ[i].append((j, u, v, dt, hop, emb, h_dist))
            n_legal_edges += 1
        else:
            for r in rejects[:3]:
                rejected_edges.append(r)
    return succ, rejected_edges, n_legal_edges


def _hist_ok(hist_embs, nodes, j, h_dist) -> tuple[bool, float, float]:
    class _EmbProxy:
        pass

    proxy = _EmbProxy()
    proxy.emb = nodes[j].emb
    hsim = pes.hist_emb_sim(hist_embs, proxy)
    emb_need = (
        pes.EMB_HIST_MIN - 0.02
        if (h_dist is not None and h_dist < pes.H_DIST_GATE)
        else pes.EMB_HIST_MIN
    )
    return hsim >= emb_need, hsim, emb_need


def _make_edge_rec(
    nodes,
    idx,
    j,
    u,
    v,
    dt,
    hop,
    emb,
    h_dist,
    hsim,
    calib,
    *,
    dt_scoring: bool = True,
    transition_prior: bool = False,
) -> dict:
    e = edge_llr(
        u,
        v,
        dt,
        emb,
        h_dist,
        calib,
        dt_scoring=dt_scoring,
        transition_prior=transition_prior,
    )
    e["hop"] = hop
    e["hist_emb"] = hsim
    e["from_super"] = nodes[idx].label
    e["to_super"] = nodes[j].label
    e["from_members"] = nodes[idx].tids
    e["to_members"] = nodes[j].tids
    e["dt_source"] = "super_union"
    return e


def enumerate_paths_llr(
    tracks: list,
    calib: dict,
    *,
    use_supernode: bool = True,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = DEFAULT_BEAM_MAX_LEAVES,
    force_full: bool = False,
    dt_scoring: bool = True,
    transition_prior: bool = False,
):
    """可選超節點圖上枚舉；邊多時改 beam（只留極大 leaf）。"""
    nodes, super_report = _build_nodes(tracks, use_supernode)
    succ, rejected_edges, n_legal_edges = _build_succ(nodes)
    n = len(nodes)
    use_beam = (not force_full) and (n_legal_edges > FULL_ENUM_EDGE_CAP)
    super_report["enumeration"] = {
        "n_legal_edges": n_legal_edges,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "beam_max_leaves": beam_max_leaves if use_beam else None,
        "dt_scoring": bool(dt_scoring),
        "transition_prior": bool(transition_prior),
        "note": (
            f"合法邊={n_legal_edges} > {FULL_ENUM_EDGE_CAP}：beam 近似 Softmax／Top-k"
            if use_beam
            else "全量 DFS（含前綴）"
        ),
    }

    def _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim):
        return _make_edge_rec(
            nodes,
            idx,
            j,
            u,
            v,
            dt,
            hop,
            emb,
            h_dist,
            hsim,
            calib,
            dt_scoring=dt_scoring,
            transition_prior=transition_prior,
        )

    all_paths = []

    if not use_beam:
        def dfs(idx, path_idx, edges_info, hist_embs):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, u, v, dt, hop, emb, h_dist in succ[idx]:
                if j in path_idx:
                    continue
                ok_h, hsim, emb_need = _hist_ok(hist_embs, nodes, j, h_dist)
                if not ok_h:
                    rejected_edges.append(
                        (
                            nodes[idx].label,
                            nodes[j].label,
                            f"歷史不像（hist_emb={hsim:.3f} < {emb_need}）",
                        )
                    )
                    continue
                e = _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim)
                edges_info.append(e)
                path_idx.append(j)
                hist_embs.append(nodes[j].emb)
                dfs(j, path_idx, edges_info, hist_embs)
                hist_embs.pop()
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [], [nodes[s].emb])
        return all_paths, rejected_edges, n_legal_edges, nodes, super_report

    # -------- beam：每層保留 score 最高的 beam_width 個前綴；葉節點進結果 --------
    leaves = []
    for s in range(n):
        beam = [(0.0, [s], [], [nodes[s].emb])]
        while beam:
            nxt = []
            for sc, path_idx, edges_info, hist_embs in beam:
                idx = path_idx[-1]
                extended = False
                for j, u, v, dt, hop, emb, h_dist in succ[idx]:
                    if j in path_idx:
                        continue
                    ok_h, hsim, _ = _hist_ok(hist_embs, nodes, j, h_dist)
                    if not ok_h:
                        continue
                    e = _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim)
                    extended = True
                    nxt.append(
                        (
                            sc + e["score"],
                            path_idx + [j],
                            edges_info + [e],
                            hist_embs + [nodes[j].emb],
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
    return all_paths, rejected_edges, n_legal_edges, nodes, super_report


def expand_path_tids(nodes: list[SuperNode], path_idx: list[int]) -> list[str]:
    """輸出展開：各超節點成員依 t_start 排序串接。"""
    out = []
    for i in path_idx:
        out.extend(nodes[i].tids)
    return out


def path_score_llr(nodes_on_path: list[SuperNode], edges_info: list, calib: dict) -> tuple[float, list]:
    node_evs = []
    total = 0.0
    for sn in nodes_on_path:
        ne = node_evidence(sn.sim, calib)
        node_evs.append({"super": sn.label, "members": sn.tids, **ne})
        total += ne["score"]
    total += sum(e["score"] for e in edges_info)
    return total, node_evs


def maximal_paths(scored: list) -> list:
    tid_seqs = {tuple(p["tids"]) for p in scored}
    is_prefix = set()
    for q in tid_seqs:
        for k in range(1, len(q)):
            is_prefix.add(q[:k])
    return [p for p in scored if tuple(p["tids"]) not in is_prefix]


def attach_softmax(maximal: list) -> list:
    if not maximal:
        return maximal
    scores = np.asarray([p["score"] for p in maximal], dtype=np.float64)
    log_z = logsumexp(scores)
    for p, s in zip(maximal, scores):
        p["path_probability"] = float(math.exp(s - log_z))
    return maximal


def _time_overlap_sec(a: pes.Track, b: pes.Track) -> float:
    return max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))


def tracks_physical_coexist_contradiction(a: pes.Track, b: pes.Track) -> bool:
    """
    同假設內部裁決：跨鏡時間共存，且鏡頭對既非 OVERLAP 亦非 ADJACENT → 矛盾。
    （假設外的採用節點與假設內節點共存不構成問題。）
    """
    if a.cam == b.cam:
        return False
    ov = _time_overlap_sec(a, b)
    if ov <= 0.0:
        tok, _, _ = _coexistence_time_ok(a, b)
        if not tok:
            return False
    key = tuple(sorted((a.cam, b.cam)))
    if key in pes.OVERLAP_PAIRS or key in pes.ADJACENT:
        return False
    return True


def hypothesis_internal_contradictions(
    tids: list[str],
    by_tid: dict[str, pes.Track],
) -> list[dict]:
    """回傳假設內部所有矛盾對（空＝通過）。"""
    tracks = [by_tid[t] for t in tids if t in by_tid]
    bad = []
    for u, v in itertools.combinations(tracks, 2):
        if tracks_physical_coexist_contradiction(u, v):
            bad.append(
                {
                    "a": u.tid,
                    "b": v.tid,
                    "cams": f"{u.cam}↔{v.cam}",
                    "overlap_sec": float(_time_overlap_sec(u, v)),
                }
            )
    return bad


def _path_as_segment(path: dict, segment_i: int, gap: float | None) -> dict:
    return {
        "segment": segment_i,
        "path": " -> ".join(path.get("super_labels") or path["tids"]),
        "super_labels": list(path.get("super_labels") or []),
        "tids": list(path["tids"]),
        "super_ids": list(path.get("super_ids") or []),
        "score": float(path["score"]),
        "t_start": float(path["t_start"]),
        "t_end": float(path["t_end"]),
        "gap_after_prev_sec": gap,
        "edges": path.get("edges"),
        "node_evidence": path.get("node_evidence"),
    }


def _hypothesis_from_segments(
    segments: list[dict],
    *,
    source: str,
    seed_rank: int | None = None,
) -> dict:
    tids: list[str] = []
    super_labels: list[str] = []
    super_ids: list[str] = []
    for seg in segments:
        for t in seg["tids"]:
            if t not in tids:
                tids.append(t)
        for lab in seg.get("super_labels") or []:
            if lab not in super_labels:
                super_labels.append(lab)
        for sid in seg.get("super_ids") or []:
            if sid not in super_ids:
                super_ids.append(sid)
    total = float(sum(s["score"] for s in segments))
    n_seg = len(segments)
    return {
        "hypothesis_type": "single" if n_seg == 1 else "segmented",
        "n_segments": n_seg,
        "score": total,
        "tids": tids,
        "super_labels": super_labels,
        "super_ids": super_ids,
        "t_start": float(segments[0]["t_start"]),
        "t_end": float(segments[-1]["t_end"]),
        "segments": segments,
        "path": " || ".join(s["path"] for s in segments),
        "edges": segments[0].get("edges") if n_seg == 1 else None,
        "node_evidence": (
            segments[0].get("node_evidence") if n_seg == 1 else None
        ),
        "source": source,
        "seed_rank": seed_rank,
    }


def grow_segmented_hypothesis(
    seed_path: dict,
    all_nodes: list,
    calib: dict,
    *,
    dt_scoring: bool,
    transition_prior: bool,
    max_segments: int = DEFAULT_MAX_HYP_SEGMENTS,
    pool_cache: dict | None = None,
) -> list[dict]:
    """
    以 seed 為 seg1，在未使用且 t_start>前段結束的超節點中遞迴取最佳延續段。
    回傳 segments（至少含 seed）；若無法延續則僅 1 段。
    """
    cache = pool_cache if pool_cache is not None else {}
    seg1 = _path_as_segment(seed_path, 1, None)
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
            maximal, _n_legal, _meta = _score_paths_on_nodes(
                pool,
                calib,
                dt_scoring=dt_scoring,
                transition_prior=transition_prior,
            )
            cache[key] = maximal
        if not maximal:
            break
        top = maximal[0]
        gap = float(top["t_start"] - prev_end)
        segments.append(_path_as_segment(top, seg_i, gap))
        used_now = set(top.get("super_ids") or [])
        remaining = [n for n in remaining if n.sid not in used_now]
        prev_end = float(top["t_end"])

    return segments


def build_ranked_hypotheses(
    single_maximal: list,
    all_nodes: list,
    tracks: list,
    calib: dict,
    *,
    dt_scoring: bool,
    transition_prior: bool,
    seed_top_k: int = DEFAULT_SEGMENT_SEED_TOP_K,
    max_segments: int = DEFAULT_MAX_HYP_SEGMENTS,
) -> tuple[list, dict]:
    """
    單路徑假設 ∪ 高分路徑延伸的多段假設 → 同一 Softmax 排名池。
    假設內部共存矛盾者作廢。計分公式不變。
    """
    by_tid = {t.tid: t for t in tracks}
    rejected = []
    pool = []
    seen_keys = set()
    pool_cache: dict = {}

    def _try_add(hyp: dict) -> None:
        key = tuple(
            tuple(seg["tids"]) for seg in hyp["segments"]
        )
        if key in seen_keys:
            return
        bad = hypothesis_internal_contradictions(hyp["tids"], by_tid)
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

    # 1) 所有極大單路徑 → 單段假設
    for rank, p in enumerate(single_maximal, 1):
        hyp = _hypothesis_from_segments(
            [_path_as_segment(p, 1, None)],
            source="single_maximal",
            seed_rank=rank,
        )
        _try_add(hyp)

    # 2) 高分 seed → 嘗試多段
    seeds = single_maximal[: max(int(seed_top_k), 0)]
    n_grown = 0
    for rank, seed in enumerate(seeds, 1):
        # 無後續候選則跳過（省枚舉）
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
        segs = grow_segmented_hypothesis(
            seed,
            all_nodes,
            calib,
            dt_scoring=dt_scoring,
            transition_prior=transition_prior,
            max_segments=max_segments,
            pool_cache=pool_cache,
        )
        if len(segs) < 2:
            continue
        hyp = _hypothesis_from_segments(
            segs, source="seed_grown", seed_rank=rank
        )
        before = len(pool)
        _try_add(hyp)
        if len(pool) > before:
            n_grown += 1

    pool.sort(key=lambda h: -h["score"])
    pool = attach_softmax(pool)
    meta = {
        "n_single_maximal": len(single_maximal),
        "n_hypotheses_ranked": len(pool),
        "n_segmented_added": n_grown,
        "n_rejected_contradiction": len(rejected),
        "seed_top_k": seed_top_k,
        "rejected_sample": rejected[:20],
        "note": SEGMENT_RANK_NOTE,
    }
    return pool, meta


def path_time_span(path: dict, nodes: list | None = None) -> tuple[float, float]:
    """路徑觀測區間：成員 t_start 最小、t_end 最大。"""
    if path.get("edges") is not None and path.get("super_ids") and nodes:
        by_sid = {n.sid: n for n in nodes}
        chosen = [by_sid[s] for s in path["super_ids"] if s in by_sid]
        if chosen:
            return (
                float(min(n.t_start for n in chosen)),
                float(max(n.t_end for n in chosen)),
            )
    # fallback：從 tids 無法還原時間；用 edges / 外層附帶
    if "t_start" in path and "t_end" in path:
        return float(path["t_start"]), float(path["t_end"])
    raise ValueError("path_time_span：路徑缺時間資訊")


def _score_paths_on_nodes(
    nodes: list,
    calib: dict,
    *,
    dt_scoring: bool,
    transition_prior: bool,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    beam_max_leaves: int = DEFAULT_BEAM_MAX_LEAVES,
) -> tuple[list, int, dict]:
    """在既有超節點子集上枚舉＋計分（不重建超節點）。"""
    if not nodes:
        return [], 0, {"n_legal_edges": 0, "mode": "empty"}

    # 暫以 tracks=各超節點代表？不行——enumerate 需要再建圖。
    # 直接對 nodes 子列表建 succ / beam。
    n = len(nodes)
    succ = [[] for _ in range(n)]
    n_legal = 0
    for i, j in itertools.permutations(range(n), 2):
        sa, sb = nodes[i], nodes[j]
        if sb.t_end < sa.t_start - pes.DT_MAX:
            continue
        best, _ = _best_member_edge(sa, sb)
        if best is not None:
            u, v, dt, hop, emb, h_dist = best
            succ[i].append((j, u, v, dt, hop, emb, h_dist))
            n_legal += 1

    use_beam = n_legal > FULL_ENUM_EDGE_CAP
    enum_meta = {
        "n_legal_edges": n_legal,
        "mode": "beam" if use_beam else "full",
        "beam_width": beam_width if use_beam else None,
        "n_nodes": n,
    }

    all_paths = []

    def _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim):
        return _make_edge_rec(
            nodes, idx, j, u, v, dt, hop, emb, h_dist, hsim, calib,
            dt_scoring=dt_scoring, transition_prior=transition_prior,
        )

    if not use_beam:
        def dfs(idx, path_idx, edges_info, hist_embs):
            all_paths.append((list(path_idx), list(edges_info)))
            for j, u, v, dt, hop, emb, h_dist in succ[idx]:
                if j in path_idx:
                    continue
                ok_h, hsim, _ = _hist_ok(hist_embs, nodes, j, h_dist)
                if not ok_h:
                    continue
                e = _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim)
                edges_info.append(e)
                path_idx.append(j)
                hist_embs.append(nodes[j].emb)
                dfs(j, path_idx, edges_info, hist_embs)
                hist_embs.pop()
                path_idx.pop()
                edges_info.pop()

        for s in range(n):
            dfs(s, [s], [], [nodes[s].emb])
    else:
        leaves = []
        for s in range(n):
            beam = [(0.0, [s], [], [nodes[s].emb])]
            while beam:
                nxt = []
                for sc, path_idx, edges_info, hist_embs in beam:
                    idx = path_idx[-1]
                    extended = False
                    for j, u, v, dt, hop, emb, h_dist in succ[idx]:
                        if j in path_idx:
                            continue
                        ok_h, hsim, _ = _hist_ok(hist_embs, nodes, j, h_dist)
                        if not ok_h:
                            continue
                        e = _rec(idx, j, u, v, dt, hop, emb, h_dist, hsim)
                        extended = True
                        nxt.append(
                            (
                                sc + e["score"],
                                path_idx + [j],
                                edges_info + [e],
                                hist_embs + [nodes[j].emb],
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
        score, node_evs = path_score_llr(sn_path, edges_info, calib)
        scored.append(
            {
                "tids": expand_path_tids(nodes, path_idx),
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
    maximal = attach_softmax(maximal_paths(scored))
    return maximal, n_legal, enum_meta


def extract_path_segments(
    all_nodes: list,
    calib: dict,
    *,
    first_path: dict | None = None,
    dt_scoring: bool = False,
    transition_prior: bool = False,
    max_segments: int = 20,
) -> list[dict]:
    """
    分段軌跡（2026-07-15）：
      Segment 1 = 全圖 Top-1（若傳 first_path 則直接採用，不再重枚舉）。
      其後：在未使用的超節點中，取 t_start > 前段結束 者，重跑同一套計分，
      Softmax 僅在該段候選極大路徑內；標注與前段的觀測空窗秒數。
      DT_MAX 不變。遞迴至無合法段落。
    """
    remaining = list(all_nodes)
    segments = []
    prev_end = None

    for seg_i in range(1, max_segments + 1):
        if not remaining:
            break

        if seg_i == 1 and first_path is not None:
            top = dict(first_path)
            if "t_start" not in top or "t_end" not in top:
                sids = set(top.get("super_ids") or [])
                chosen = [n for n in all_nodes if n.sid in sids]
                if not chosen and top.get("super_labels"):
                    by_lab = {n.label: n for n in all_nodes}
                    chosen = [by_lab[l] for l in top["super_labels"] if l in by_lab]
                if not chosen:
                    # 最後手段：用 tids 對齊成員
                    tid_set = set(top["tids"])
                    chosen = [n for n in all_nodes if tid_set & set(n.tids)]
                top["t_start"] = float(min(n.t_start for n in chosen))
                top["t_end"] = float(max(n.t_end for n in chosen))
                top["super_ids"] = [n.sid for n in chosen]
            gap = None
            enum_meta = {"mode": "from_global_top1"}
            n_legal = None
            n_cand = None
            top3 = None
        else:
            pool = (
                remaining
                if prev_end is None
                else [n for n in remaining if n.t_start > prev_end]
            )
            if not pool:
                break
            maximal, n_legal, enum_meta = _score_paths_on_nodes(
                pool,
                calib,
                dt_scoring=dt_scoring,
                transition_prior=transition_prior,
            )
            if not maximal:
                break
            top = maximal[0]
            gap = None if prev_end is None else float(top["t_start"] - prev_end)
            n_cand = len(maximal)
            top3 = [
                {
                    "rank": j,
                    "score": p["score"],
                    "path_probability": p.get("path_probability"),
                    "path": " -> ".join(p.get("super_labels") or p["tids"]),
                    "tids": p["tids"],
                }
                for j, p in enumerate(maximal[:3], 1)
            ]

        segments.append(
            {
                "segment": seg_i,
                "path": " -> ".join(top.get("super_labels") or top["tids"]),
                "super_labels": top.get("super_labels"),
                "tids": top["tids"],
                "super_ids": top.get("super_ids"),
                "score": top["score"],
                "path_probability": top.get("path_probability"),
                "t_start": top["t_start"],
                "t_end": top["t_end"],
                "gap_after_prev_sec": gap,
                "n_candidates_maximal": n_cand,
                "n_legal_edges_in_pool": n_legal,
                "enumeration": enum_meta,
                "edges": top.get("edges"),
                "node_evidence": top.get("node_evidence"),
                "top3": top3,
            }
        )
        used_sids = set(top.get("super_ids") or [])
        if not used_sids:
            # 用 tids 去除
            used_tids = set(top["tids"])
            remaining = [n for n in remaining if not (used_tids & set(n.tids))]
        else:
            remaining = [n for n in remaining if n.sid not in used_sids]
        prev_end = float(top["t_end"])

    return segments


def best_disjoint_alternative(maximal: list) -> dict | None:
    if len(maximal) < 2:
        return None
    top = maximal[0]
    top_set = set(top["tids"])
    for p in maximal[1:]:
        if set(p["tids"]).isdisjoint(top_set):
            return p
    return None


def render_collage_if_available(merge_dir: Path, top: dict, out_png: Path) -> Path | None:
    try:
        return pes.render_top1_collage(merge_dir, top, out_png, title_prefix="path_enum LLR Top-1")
    except Exception as exc:
        print(f"警告：拼圖失敗（{exc}），略過 PNG")
        return None


def _fmt_llr_dt(e: dict) -> str:
    if e.get("LLR_dt") == "removed":
        return "removed"
    if e.get("dt_model") == "handoff" or e.get("LLR_dt") is None:
        return "n/a"
    return f"{e['LLR_dt']:+.3f}"


def write_txt_report(
    out_txt: Path,
    merge_dir: Path,
    tracks: list,
    scored: list,
    maximal: list,
    n_legal_edges: int,
    alt: dict | None,
    *,
    super_report: dict | None = None,
    gate_info: dict | None = None,
    segments: list | None = None,
) -> None:
    lines = []
    lines.append(f"輸入：{merge_dir}")
    lines.append(f"SIM_MIN={pes.SIM_MIN}  MODE={pes.MODE}  H矩陣={len(pes.H_MATRICES)}")
    lines.append(
        f"MIN_TRANSIT hop1={pes.DEFAULT_MIN_TRANSIT_HOP1}  hop2={pes.DEFAULT_MIN_TRANSIT_HOP2}"
    )
    if gate_info:
        lines.append(
            f"EMB gates: EDGE={gate_info['EMB_EDGE_MIN']} HIST={gate_info['EMB_HIST_MIN']}  "
            f"({gate_info.get('rationale','')[:80]})"
        )
    lines.append(NODE_EVIDENCE_NOTE)
    lines.append(PRIOR_WEAK_NOTE)
    lines.append(SEGMENT_RANK_NOTE)
    lines.append(f"候選 track={len(tracks)}  合法邊={n_legal_edges}")
    lines.append(f"合法路徑（含前綴）={len(scored)}  排名假設={len(maximal)}")
    if super_report:
        lines.append(
            f"超節點：{super_report['n_supernodes']}  "
            f"合併對數={super_report['n_merged_pairs']}  "
            f"多成員={super_report.get('multi_only')}"
        )
    lines.append("")
    lines.append(f"===== Top {min(3, len(maximal))} 假設（單路徑＋分段，同 Softmax）=====")
    for rank, p in enumerate(maximal[:3], 1):
        n_seg = int(p.get("n_segments") or 1)
        lines.append(
            f"\n#{rank}  score={p['score']:.4f}  P={p.get('path_probability', 0):.6f}  "
            f"段數={n_seg}  type={p.get('hypothesis_type', 'single')}"
        )
        segs = p.get("segments") or []
        if segs:
            for seg in segs:
                gap = seg.get("gap_after_prev_sec")
                gap_s = f"  空窗={gap:.1f}s" if gap is not None else ""
                lines.append(
                    f"  [seg{seg['segment']}] score={seg['score']:.4f}{gap_s}  "
                    f"{seg['path']}"
                )
                lines.append(
                    f"    span=[{seg['t_start']:.1f}, {seg['t_end']:.1f}]"
                )
        else:
            lines.append(
                "  " + " -> ".join(p.get("super_labels") or p["tids"])
            )
        lines.append("  expanded union: " + " -> ".join(p["tids"]))
        if n_seg == 1:
            for ne in p.get("node_evidence") or []:
                if ne.get("enabled"):
                    lines.append(
                        f"      NODE {ne['super']} sim={ne['sim']:.3f}  "
                        f"LLR_raw={ne['raw']:+.3f} w={ne['w']:.3f}  "
                        f"score={ne['score']:+.3f}"
                    )
            for e in p.get("edges") or []:
                hd = e.get("h_dist")
                hd_s = f"{hd:.1f}" if hd is not None else "—"
                ldh = e.get("LLR_dH")
                ldh_s = f"{ldh:+.3f}" if ldh is not None else "—"
                prior_tag = ""
                if e.get("dt_prior_physical"):
                    prior_tag = f"  [{e.get('dt_note') or 'PRIOR-WEAK'}]"
                lines.append(
                    f"      {e.get('from_super', e['from'])} -> "
                    f"{e.get('to_super', e['to'])}   "
                    f"hop={e['hop']}  dt={e['dt']:.1f}s  "
                    f"dt_model={e.get('dt_model')}  "
                    f"emb={e['emb']:.3f}  d_H={hd_s}  "
                    f"LLR_dt={_fmt_llr_dt(e)}  LLR_emb={e['LLR_emb']:+.3f}  "
                    f"LLR_dH={ldh_s}  edge={e['score']:+.3f}{prior_tag}"
                )

    if maximal:
        top = maximal[0]
        lines.append("")
        lines.append(
            f"Top-1 path_probability={top.get('path_probability', 0):.6f}  "
            f"score={top['score']:.4f}  n_segments={top.get('n_segments', 1)}"
        )
        if alt is not None:
            ratio = top["path_probability"] / max(alt["path_probability"], 1e-300)
            lines.append(
                f"最佳不共用 track 替代假設：P={alt['path_probability']:.6f}  "
                f"score={alt['score']:.4f}  ratio(Top1/alt)={ratio:.3f}"
            )
            lines.append("  alt: " + (alt.get("path") or " -> ".join(alt["tids"])))

    if segments:
        lines.append("")
        lines.append(f"===== Top-1 假設各段（共 {len(segments)} 段）=====")
        for seg in segments:
            gap = seg.get("gap_after_prev_sec")
            gap_s = f"  觀測空窗={gap:.1f}s" if gap is not None else ""
            lines.append(
                f"\n[seg{seg['segment']}] score={seg['score']:.4f}{gap_s}"
            )
            lines.append(f"  {seg['path']}")
            lines.append(
                f"  span=[{seg['t_start']:.1f}, {seg['t_end']:.1f}]"
            )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary_json(
    merge_dir: Path,
    scored: list,
    maximal: list,
    collage: Path | None,
    alt: dict | None,
    n_legal_edges: int,
    n_tracks: int,
    *,
    super_report: dict | None = None,
    gate_info: dict | None = None,
    options: dict | None = None,
) -> dict:
    top = maximal[0] if maximal else None
    ratio = None
    if top and alt is not None:
        ratio = top["path_probability"] / max(alt["path_probability"], 1e-300)
    segments = (options or {}).get("segments") or []
    # JSON 精簡：段落不重複塞完整 edges
    segments_out = []
    for seg in segments:
        segments_out.append(
            {
                k: seg.get(k)
                for k in (
                    "segment",
                    "path",
                    "super_labels",
                    "tids",
                    "score",
                    "path_probability",
                    "t_start",
                    "t_end",
                    "gap_after_prev_sec",
                    "n_candidates_maximal",
                    "n_legal_edges_in_pool",
                    "enumeration",
                    "top3",
                )
            }
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
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": pes.MODE,
        "scoring": "LLR",
        "options": {
            k: v
            for k, v in (options or {}).items()
            if k not in ("segments",)
        },
        "gate_info": gate_info,
        "supernodes": super_report,
        "node_evidence_note": NODE_EVIDENCE_NOTE,
        "prior_weak_note": PRIOR_WEAK_NOTE,
        "segment_rank_note": SEGMENT_RANK_NOTE,
        "input_dir": str(merge_dir.resolve()),
        "sim_min": pes.SIM_MIN,
        "n_tracks": n_tracks,
        "n_legal_edges": n_legal_edges,
        "n_paths_all": len(scored),
        "n_hypotheses_ranked": len(maximal),
        "n_paths_maximal": len(maximal),  # 兼容舊欄位＝排名假設數
        "collage": str(collage.resolve()) if collage else None,
        "segments": segments_out,
        "top1": _hyp_brief(top, 1) if top else None,
        "top3_hypotheses": [
            _hyp_brief(p, i) for i, p in enumerate(maximal[:3], 1)
        ],
        "best_disjoint_alternative": {
            "score": alt["score"],
            "path_probability": alt.get("path_probability"),
            "tids": alt["tids"],
            "n_segments": alt.get("n_segments"),
            "path": alt.get("path") or " -> ".join(alt["tids"]),
            "probability_ratio_top1_over_alt": ratio,
        }
        if alt
        else None,
        "top10_paths": [
            _hyp_brief(p, i) for i, p in enumerate(maximal[:10], 1)
        ],
    }


def run_llr(
    merge_dir: Path,
    calib: dict,
    *,
    use_emb_gate_fix: bool = True,
    use_supernode: bool = True,
    use_node_evidence: bool = True,
    dt_scoring: bool = True,
    transition_prior: bool = False,
):
    gate_info = gates.apply_llr_emb_gates(enabled=use_emb_gate_fix)
    tracks = pes.load_tracks(str(merge_dir))
    print(f"讀入 {len(tracks)} 條 track")
    print(
        f"options: emb_gate={use_emb_gate_fix}({gate_info['EMB_EDGE_MIN']})  "
        f"supernode={use_supernode}  node_evidence={use_node_evidence}  "
        f"dt_scoring={dt_scoring}  transition_prior={transition_prior}"
    )

    calib_use = calib
    if not use_node_evidence:
        calib_use = dict(calib)
        calib_use.pop("sim_gt", None)
        calib_use.pop("sim_nongt", None)

    all_paths, rejected, n_legal_edges, nodes, super_report = enumerate_paths_llr(
        tracks,
        calib_use,
        use_supernode=use_supernode,
        dt_scoring=dt_scoring,
        transition_prior=transition_prior,
    )
    print(
        f"超節點={super_report['n_supernodes']}  多成員={super_report.get('multi_only')}"
    )

    scored = []
    for path_idx, edges_info in all_paths:
        sn_path = [nodes[i] for i in path_idx]
        score, node_evs = path_score_llr(sn_path, edges_info, calib_use)
        scored.append(
            {
                "tids": expand_path_tids(nodes, path_idx),
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
    single_maximal = maximal_paths(scored)
    # 單路徑 Softmax 僅供診斷；真正排名改為假設池
    single_maximal_diag = attach_softmax([dict(p) for p in single_maximal])
    print(f"合法邊={n_legal_edges}  路徑={len(scored)}  極大單路徑={len(single_maximal)}")

    ranked, rank_meta = build_ranked_hypotheses(
        single_maximal,
        nodes,
        tracks,
        calib_use,
        dt_scoring=dt_scoring,
        transition_prior=transition_prior,
    )
    print(
        f"排名假設={rank_meta['n_hypotheses_ranked']}  "
        f"多段新增={rank_meta['n_segmented_added']}  "
        f"共存矛盾作廢={rank_meta['n_rejected_contradiction']}"
    )
    if ranked:
        top = ranked[0]
        print(
            f"Top-1 P={top['path_probability']:.6f}  "
            f"score={top['score']:.4f}  "
            f"n_seg={top['n_segments']}  "
            + top.get("path", "")
        )
        for seg in top.get("segments") or []:
            gap = seg.get("gap_after_prev_sec")
            gap_s = f"  gap={gap:.1f}s" if gap is not None else ""
            print(
                f"  seg{seg['segment']}: score={seg['score']:.4f}{gap_s}  {seg['path']}"
            )

    # 兼容：maximal = 排名後的假設（含單／多段）
    maximal = ranked
    # options.segments = Top-1 假設的各段（供拼圖／舊協定）
    segments = list((maximal[0].get("segments") if maximal else None) or [])

    options = {
        "use_emb_gate_fix": use_emb_gate_fix,
        "use_supernode": use_supernode,
        "use_node_evidence": use_node_evidence,
        "dt_scoring": bool(dt_scoring),
        "transition_prior": bool(transition_prior),
        "prior_dt_sigma": PRIOR_DT_SIGMA,
        "dt_scoring_note": (
            None
            if dt_scoring
            else "2026-07-15：tau 無實測來源，transit LLR_dt 軟證據停用"
        ),
        "min_transit_hop1": float(pes.DEFAULT_MIN_TRANSIT_HOP1),
        "min_transit_hop2": float(pes.DEFAULT_MIN_TRANSIT_HOP2),
        "segments": segments,
        "ranking_protocol": SEGMENT_RANK_NOTE,
        "ranking_meta": rank_meta,
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
    return tracks, scored, maximal, n_legal_edges, nodes, super_report, gate_info, options


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="path enum with LLR scoring + structural fixes")
    p.add_argument(
        "input_dir",
        nargs="?",
        default=str(QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260507"),
    )
    p.add_argument("--sim-min", type=float, default=0.85)
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-emb-gate-fix", action="store_true", help="關閉修正一（維持 0.91/0.90）")
    p.add_argument("--no-supernode", action="store_true", help="關閉修正二")
    p.add_argument("--no-node-evidence", action="store_true", help="關閉修正三節點證據")
    p.add_argument(
        "--dt-scoring",
        choices=["on", "off"],
        default="on",
        help="transit LLR_dt：on=計分 / off=0+removed（硬規則不動）",
    )
    p.add_argument(
        "--transition-prior",
        choices=["on", "off"],
        default="off",
        help="每邊加 ln(p_edge)；p_edge 來自 GT 校準",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    merge_dir = Path(args.input_dir).resolve()
    if not merge_dir.is_dir():
        raise SystemExit(f"找不到資料夾：{merge_dir}")

    out_dir = (args.out_dir or (OUTPUT_ROOT / "path_enum_llr")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_path = (args.calibration or (out_dir / "calibration.pkl")).resolve()
    if not calib_path.is_file():
        raise SystemExit(f"找不到 calibration.pkl：{calib_path}")

    pes.SIM_MIN = float(args.sim_min)
    mode = pes.configure_for_input(str(merge_dir))
    print(f"模式：{mode}  SIM_MIN={pes.SIM_MIN}")
    print(f"calibration：{calib_path}")
    calib = load_calibration(calib_path)

    (
        tracks,
        scored,
        maximal,
        n_legal_edges,
        nodes,
        super_report,
        gate_info,
        options,
    ) = run_llr(
        merge_dir,
        calib,
        use_emb_gate_fix=not args.no_emb_gate_fix,
        use_supernode=not args.no_supernode,
        use_node_evidence=not args.no_node_evidence,
        dt_scoring=(args.dt_scoring == "on"),
        transition_prior=(args.transition_prior == "on"),
    )
    alt = best_disjoint_alternative(maximal)

    tag = merge_dir.name
    out_txt = out_dir / f"{tag}_llr_out.txt"
    out_json = out_dir / f"{tag}_llr_top1.json"
    out_png = out_dir / f"{tag}_llr_top1_collage.png"
    out_super = out_dir / f"{tag}_supernodes.json"

    write_txt_report(
        out_txt,
        merge_dir,
        tracks,
        scored,
        maximal,
        n_legal_edges,
        alt,
        super_report=super_report,
        gate_info=gate_info,
        segments=options.get("segments"),
    )
    collage = None
    if maximal:
        collage = render_collage_if_available(merge_dir, maximal[0], out_png)
    summary = build_summary_json(
        merge_dir,
        scored,
        maximal,
        collage,
        alt,
        n_legal_edges,
        len(tracks),
        super_report=super_report,
        gate_info=gate_info,
        options=options,
    )
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    out_super.write_text(json.dumps(super_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"文字報告：{out_txt}")
    print(f"JSON：{out_json}")
    print(f"超節點：{out_super}")
    if collage:
        print(f"拼圖：{collage}")
    return summary


if __name__ == "__main__":
    main()
