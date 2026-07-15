# -*- coding: utf-8 -*-
"""OUT-OF-SAMPLE 0528：拓撲盤點、GT 可行性、09↔10 裸跑診斷（不跑分）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import llr_gate_config as gates  # noqa: E402
import path_enum_llr as llr  # noqa: E402
import path_enum_scoring as pes  # noqa: E402
from evaluate_paths import diagnose_gt_feasibility, write_diagnose_txt  # noqa: E402

try:
    from repo_paths import OUTPUT_ROOT, QUERY_FILTER_OUTPUT_ROOT
except ImportError:
    OUTPUT_ROOT = REPO_ROOT.parent / "output"
    QUERY_FILTER_OUTPUT_ROOT = OUTPUT_ROOT / "query_filter_merge"

MERGE = str(QUERY_FILTER_OUTPUT_ROOT / "人員追蹤_20260528")
GT_PATH = OUTPUT_ROOT / "path_enum_llr" / "ground_truth_20260528.json"
OUT_DIR = OUTPUT_ROOT / "path_enum_llr"
N_GT = 16

TARGET_CAMS = ("K8-10", "K8-12", "K8-30")
GT_TIDS_09_10 = ("K8-09_3", "K8-09_94", "K8-09_142", "K8-10_32")


def cam_relation(cam_a: str, cam_b: str) -> str:
    if cam_a == cam_b:
        return "same_cam"
    key = tuple(sorted((cam_a, cam_b)))
    if key in pes.OVERLAP_PAIRS:
        return f"OVERLAP (tol={pes.OVERLAP_PAIRS[key]}s)"
    hop = pes.hop_count(cam_a, cam_b)
    if hop == 1:
        return "ADJACENT"
    if hop == 2:
        return "hop2"
    return "unreachable"


def topology_inventory() -> dict:
    pes.SIM_MIN = 0.85
    pes.configure_for_input(MERGE)
    pes._load_h_matrices()

    def neighbors(cam: str) -> list[str]:
        nbs = set()
        for a, b in pes.ADJACENT:
            if cam in (a, b):
                nbs.add(b if a == cam else a)
        return sorted(nbs)

    def h_pairs(cam: str) -> list[str]:
        out = []
        for a, b in pes.H_MATRICES:
            if cam in (a, b):
                out.append(f"{a}->{b}")
        return sorted(out)

    def overlap_entries(cam: str) -> list[str]:
        return [
            f"{a}|{b} tol={pes.OVERLAP_PAIRS[k]}s"
            for k in pes.OVERLAP_PAIRS
            for a, b in [k]
            if cam in (a, b)
        ]

    all_h = sorted(f"{a}->{b}" for a, b in pes.H_MATRICES)
    gt_cams = {
        "K8-01", "K8-05", "K8-08", "K8-09", "K8-10",
        "K8-12", "K8-22", "K8-23", "K8-30",
    }

    per_cam = {}
    gaps = []
    for cam in TARGET_CAMS:
        in_adj = any(cam in p for p in pes.ADJACENT)
        nbs = neighbors(cam)
        ovs = overlap_entries(cam)
        hs = h_pairs(cam)
        hops = {}
        unreachable = []
        for oc in sorted(gt_cams):
            if oc == cam:
                continue
            h = pes.hop_count(cam, oc)
            if h is not None:
                hops[oc] = h
            else:
                unreachable.append(oc)
        rec = {
            "in_PERSON_ADJACENT": in_adj,
            "neighbors": nbs,
            "OVERLAP_PAIRS": ovs,
            "H_matrices": hs,
            "hop_to_other_gt_cams": hops,
            "unreachable_gt_cams": unreachable,
            "in_VEHICLE_CORRIDOR": cam in pes.VEHICLE_CORRIDOR,
            "vehicle_corridor_neighbors": [],
        }
        if cam in pes.VEHICLE_CORRIDOR:
            idx = pes.VEHICLE_CORRIDOR.index(cam)
            if idx > 0:
                rec["vehicle_corridor_neighbors"].append(pes.VEHICLE_CORRIDOR[idx - 1])
            if idx < len(pes.VEHICLE_CORRIDOR) - 1:
                rec["vehicle_corridor_neighbors"].append(pes.VEHICLE_CORRIDOR[idx + 1])
        per_cam[cam] = rec
        if not in_adj and not ovs and not hs:
            gaps.append(cam)

    return {
        "mode": pes.MODE,
        "HOMOGRAPHY_DIR": str(pes.HOMOGRAPHY_DIR),
        "PERSON_ADJACENT": sorted([list(p) for p in pes.ADJACENT]),
        "PERSON_OVERLAP_PAIRS": {f"{a}|{b}": v for (a, b), v in pes.OVERLAP_PAIRS.items()},
        "all_H_matrices": all_h,
        "target_cams": per_cam,
        "topology_gaps_blocking_scoring": gaps,
        "can_run_scoring": len(gaps) == 0,
        "note": "K8-09↔K8-10 實際有小重疊，本輪刻意未登記於 PERSON_OVERLAP_PAIRS。",
    }


def overlap_pair_detail(a: pes.Track, b: pes.Track) -> dict:
    ov_start = max(a.t_start, b.t_start)
    ov_end = min(a.t_end, b.t_end)
    overlap_sec = max(0.0, ov_end - ov_start)
    return {
        "a": a.tid,
        "b": b.tid,
        "a_span": [a.t_start, a.t_end],
        "b_span": [b.t_start, b.t_end],
        "overlap_sec": overlap_sec,
        "has_time_overlap": overlap_sec > 0,
        "cam_relation": cam_relation(a.cam, b.cam),
    }


def diagnose_09_10(tracks: list) -> dict:
    by_tid = {t.tid: t for t in tracks}
    pairs = []
    for tid_a in ("K8-09_3", "K8-09_94", "K8-09_142"):
        if tid_a not in by_tid or "K8-10_32" not in by_tid:
            continue
        a, b = by_tid[tid_a], by_tid["K8-10_32"]
        detail = overlap_pair_detail(a, b)
        ok_ab, reason_ab, dt_ab, hop_ab, emb_ab, h_ab = pes.edge_check(a, b)
        ok_ba, reason_ba, dt_ba, hop_ba, emb_ba, h_ba = pes.edge_check(b, a)
        ok_merge, merge_reason = llr.coexistence_merge(a, b)
        detail.update(
            {
                "edge_a_to_b": {
                    "ok": bool(ok_ab),
                    "reason": reason_ab or "",
                    "dt": dt_ab,
                    "hop": hop_ab,
                    "emb": emb_ab,
                    "h_dist": h_ab,
                },
                "edge_b_to_a": {
                    "ok": bool(ok_ba),
                    "reason": reason_ba or "",
                    "dt": dt_ba,
                    "hop": hop_ba,
                    "emb": emb_ba,
                    "h_dist": h_ba,
                },
                "coexistence_merge": {"ok": bool(ok_merge), "reason": merge_reason},
            }
        )
        pairs.append(detail)

  # 超節點：09_* 與 10_32 是否同組
    supers, sn_report = llr.build_supernodes(tracks)
    merge_log = sn_report.get("merge_log", [])
    tid_to_super = {}
    for s in supers:
        for tid in s.tids:
            tid_to_super[tid] = sorted(s.tids)

    super_check = {}
    for tid in GT_TIDS_09_10:
        if tid in tid_to_super:
            mates = [x for x in tid_to_super[tid] if x != tid]
            super_check[tid] = {
                "supernode_tids": tid_to_super[tid],
                "merged_with_10_32": "K8-10_32" in tid_to_super[tid] and tid != "K8-10_32",
                "other_members": mates,
            }
        else:
            super_check[tid] = {"missing_in_candidates": True}

    relevant_merges = [
        m for m in merge_log
        if m["a"] in GT_TIDS_09_10 or m["b"] in GT_TIDS_09_10
        or m["a"] == "K8-10_32" or m["b"] == "K8-10_32"
    ]

    return {
        "pairs": pairs,
        "supernode_membership": super_check,
        "merge_log_involving_gt_09_10": relevant_merges,
        "expected_no_merge": "無 OVERLAP 登記且無 H 時，09↔10 不應合併",
    }


def enrich_overlaps(diag: dict) -> None:
    for ov in diag.get("time_overlaps_among_gt", []):
        ov["cam_relation"] = cam_relation(ov["a_cam"], ov["b_cam"])


def main():
    gates.apply_llr_emb_gates(True)

    topo = topology_inventory()
    tracks = pes.load_tracks(MERGE)
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    gt_tids = gt["person_tids"]
    assert len(gt_tids) == N_GT

    diag = diagnose_gt_feasibility(tracks, gt_tids)
    enrich_overlaps(diag)
    diag_09_10 = diagnose_09_10(tracks)

    out = {
        "dataset": "人員追蹤_20260528",
        "frozen": {
            "calibration": "calibration_gt0507.pkl",
            "settings": "B: dt-scoring off, transition-prior off, EMB 0.80, supernode, node_evidence",
            "no_0528_tuning": True,
            "out_of_sample": True,
        },
        "topology": topo,
        "feasibility": diag,
        "task3_09_10": diag_09_10,
        "scoring_skipped": not topo["can_run_scoring"],
        "scoring_skip_reason": (
            "K8-10/K8-12/K8-30 未登記於人員 ADJACENT/OVERLAP/H，拓撲不可達；"
            "依任務零暫停跑分，待場地圖資補齊。"
            if not topo["can_run_scoring"]
            else None
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "oos_0528_diagnostics.json"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    txt_path = OUT_DIR / "gt_feasibility_20260528.txt"
    write_diagnose_txt(diag, txt_path)

    print(f"拓撲缺口：{topo['topology_gaps_blocking_scoring']}")
    print(f"可跑分：{topo['can_run_scoring']}")
    print(f"GT 候選內：{diag['n_gt_in_candidates']}/{diag['n_gt']}")
    print(f"最長可行路徑：{diag['max_gt_coverable']}/{diag['n_gt']}")
    print(f"JSON：{json_path}")
    print(f"TXT：{txt_path}")
    return out


if __name__ == "__main__":
    main()
