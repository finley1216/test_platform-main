#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版事件測試腳本 - 測試關鍵事件和日期組合
"""

import sys
import requests
import json
from pathlib import Path

API_BASE = "http://localhost:8080"

# 關鍵測試案例
TEST_CASES = [
    # 火災相關
    ("給我火災的影片", "fire", "火災"),
    ("給我1220火災的影片", "fire", "火災+日期"),
    ("給我1221火災的影片", "fire", "火災+日期"),
    ("給我火的影片", "fire", "火"),
    ("給我煙的影片", "fire", "煙"),
    
    # 積水相關
    ("給我積水的影片", "water_flood", "積水"),
    ("給我1220積水的影片", "water_flood", "積水+日期"),
    ("給我水災的影片", "water_flood", "水災"),
    ("給我淹水的影片", "water_flood", "淹水"),
    
    # 倒地相關
    ("給我倒地的影片", "person_fallen_unmoving", "倒地"),
    ("給我倒地不起的影片", "person_fallen_unmoving", "倒地不起"),
    ("給我1220倒地的影片", "person_fallen_unmoving", "倒地+日期"),
    
    # 群聚相關
    ("給我群聚的影片", "crowd_loitering", "群聚"),
    ("給我1220群聚的影片", "crowd_loitering", "群聚+日期"),
    ("給我逗留的影片", "crowd_loitering", "逗留"),
    
    # 吸菸相關
    ("給我吸菸的影片", "smoking_outside_zone", "吸菸"),
    ("給我1220吸菸的影片", "smoking_outside_zone", "吸菸+日期"),
    
    # 併排停車相關
    ("給我併排停車的影片", "double_parking_lane_block", "併排停車"),
    ("給我1220併排停車的影片", "double_parking_lane_block", "併排停車+日期"),
    
    # 遮臉相關
    ("給我遮臉的影片", "abnormal_attire_face_cover_at_entry", "遮臉"),
    ("給我1220遮臉的影片", "abnormal_attire_face_cover_at_entry", "遮臉+日期"),
    
    # 安全門闖入相關
    ("給我闖入的影片", "security_door_tamper", "闖入"),
    ("給我1220闖入的影片", "security_door_tamper", "闖入+日期"),
]

def test_query(query, expected_event, test_name):
    """測試單個查詢"""
    try:
        response = requests.post(
            f"{API_BASE}/rag/search",
            json={
                "query": query,
                "top_k": 5,
                "score_threshold": 0.0
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "query": query,
                "test_name": test_name
            }
        
        data = response.json()
        hits = data.get("hits", [])
        
        # 分析結果
        event_match_count = 0
        keyword_match_count = 0
        total_score = 0
        
        for hit in hits:
            events_true = hit.get("events_true", [])
            summary = hit.get("summary", "").lower()
            summary_original = hit.get("summary", "")
            score = hit.get("score", 0)
            total_score += score
            
            # 檢查事件標記
            if expected_event in events_true:
                event_match_count += 1
            
            # 檢查關鍵字（簡化檢查）
            query_keywords = query.replace("的", "").replace("影片", "").replace("畫面", "").split()
            if any(kw in summary_original or kw.lower() in summary for kw in query_keywords if len(kw) > 1):
                keyword_match_count += 1
        
        avg_score = total_score / len(hits) if hits else 0
        
        return {
            "success": True,
            "query": query,
            "test_name": test_name,
            "count": len(hits),
            "event_match": event_match_count,
            "keyword_match": keyword_match_count,
            "avg_score": avg_score,
            "has_results": len(hits) > 0,
            "has_relevant": event_match_count > 0 or keyword_match_count > 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "test_name": test_name
        }

def main():
    print("=" * 120)
    print("RAG 事件搜索測試報告")
    print("=" * 120)
    print()
    
    results = []
    for query, event_type, test_name in TEST_CASES:
        print(f"測試: {query} ({test_name})")
        result = test_query(query, event_type, test_name)
        results.append(result)
        
        if not result["success"]:
            print(f"  ❌ 失敗: {result.get('error', 'Unknown')}")
        elif not result["has_results"]:
            print(f"  ⚠️  無結果")
        elif result["has_relevant"]:
            print(f"  ✅ 找到 {result['count']} 筆 (事件匹配: {result['event_match']}, 關鍵字匹配: {result['keyword_match']}, 平均分數: {result['avg_score']:.2f})")
        else:
            print(f"  ⚠️  找到 {result['count']} 筆但可能不相關 (平均分數: {result['avg_score']:.2f})")
        print()
    
    # 統計
    print("=" * 120)
    print("測試總結")
    print("=" * 120)
    
    total = len(results)
    success = sum(1 for r in results if r.get("success", False))
    has_results = sum(1 for r in results if r.get("has_results", False))
    has_relevant = sum(1 for r in results if r.get("has_relevant", False))
    
    print(f"總測試數: {total}")
    print(f"成功執行: {success} ({success/total*100:.1f}%)")
    print(f"有結果: {has_results} ({has_results/total*100:.1f}%)")
    print(f"有相關結果: {has_relevant} ({has_relevant/total*100:.1f}%)")
    print()
    
    # 按事件類型分組統計
    event_stats = {}
    for result in results:
        if not result.get("success"):
            continue
        test_name = result.get("test_name", "")
        event_type = test_name.split("+")[0]  # 提取事件類型
        
        if event_type not in event_stats:
            event_stats[event_type] = {"total": 0, "has_results": 0, "has_relevant": 0}
        
        event_stats[event_type]["total"] += 1
        if result.get("has_results"):
            event_stats[event_type]["has_results"] += 1
        if result.get("has_relevant"):
            event_stats[event_type]["has_relevant"] += 1
    
    print("按事件類型統計:")
    print("-" * 120)
    for event_type, stats in sorted(event_stats.items()):
        print(f"{event_type}:")
        print(f"  總測試: {stats['total']}")
        print(f"  有結果: {stats['has_results']} ({stats['has_results']/stats['total']*100:.1f}%)")
        print(f"  有相關: {stats['has_relevant']} ({stats['has_relevant']/stats['total']*100:.1f}%)")
        print()
    
    print("=" * 120)

if __name__ == "__main__":
    main()

