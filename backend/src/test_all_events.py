#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試所有事件類型的 RAG 搜索功能
測試不同的 user prompt 組合（包括日期變化）
"""

import sys
import requests
import json
from pathlib import Path

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent))

# API 端點
API_BASE = "http://localhost:8080"

# 定義所有事件類型和對應的關鍵字
EVENT_TESTS = {
    "fire": {
        "keywords": ["火災", "火", "fire", "煙", "濃煙"],
        "name": "火災"
    },
    "water_flood": {
        "keywords": ["水災", "水", "淹水", "積水", "flood"],
        "name": "水災/積水"
    },
    "person_fallen_unmoving": {
        "keywords": ["倒地", "倒地不起", "fallen", "unmoving"],
        "name": "人員倒地不起"
    },
    "crowd_loitering": {
        "keywords": ["群聚", "聚眾", "逗留", "crowd", "loitering"],
        "name": "群聚逗留"
    },
    "abnormal_attire_face_cover_at_entry": {
        "keywords": ["遮臉", "異常著裝", "face", "cover"],
        "name": "遮臉/異常著裝"
    },
    "double_parking_lane_block": {
        "keywords": ["併排", "停車", "阻塞", "parking", "block"],
        "name": "併排停車/阻塞"
    },
    "smoking_outside_zone": {
        "keywords": ["吸菸", "抽菸", "smoking"],
        "name": "吸菸"
    },
    "security_door_tamper": {
        "keywords": ["闖入", "突破", "安全門", "tamper", "door"],
        "name": "安全門闖入"
    }
}

# 日期變化
DATES = ["1218", "1219", "1220", "1221", "1222", None]  # None 表示不包含日期

# 不同的 prompt 模板
PROMPT_TEMPLATES = [
    "給我{keyword}的影片",
    "我想找出{keyword}的畫面",
    "有{keyword}的影片嗎",
    "幫我找{keyword}",
    "給我{date}{keyword}的影片",
    "我想找出{date}{keyword}的畫面",
    "有{date}{keyword}的影片嗎",
    "幫我找{date}{keyword}",
    "給我{date}的{keyword}影片",
    "我想找出{date}的{keyword}畫面",
]

def test_rag_search(query, event_name):
    """測試 RAG 搜索"""
    try:
        response = requests.post(
            f"{API_BASE}/rag/search",
            json={
                "query": query,
                "top_k": 10,
                "score_threshold": 0.0
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            hits = data.get("hits", [])
            return {
                "success": True,
                "count": len(hits),
                "hits": hits[:3]  # 只取前3個結果
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def format_result(result, query):
    """格式化測試結果"""
    if not result["success"]:
        return f"  ❌ 失敗: {result.get('error', 'Unknown error')}"
    
    count = result["count"]
    if count == 0:
        return f"  ⚠️  找到 0 筆結果"
    
    # 檢查結果中是否有相關的事件標記或關鍵字
    hits = result.get("hits", [])
    relevant_count = 0
    for hit in hits:
        events_true = hit.get("events_true", [])
        summary = hit.get("summary", "").lower()
        # 檢查是否有事件標記或摘要中包含關鍵字
        if events_true or any(kw in summary for kw in query.split()):
            relevant_count += 1
    
    status = "✅" if relevant_count > 0 else "⚠️"
    return f"  {status} 找到 {count} 筆結果 (相關: {relevant_count}/{len(hits)})"

def run_tests():
    """執行所有測試"""
    print("=" * 120)
    print("RAG 事件搜索測試")
    print("=" * 120)
    print()
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for event_type, event_info in EVENT_TESTS.items():
        event_name = event_info["name"]
        keywords = event_info["keywords"]
        
        print(f"\n【事件類型】{event_name} ({event_type})")
        print("-" * 120)
        
        # 測試每個關鍵字
        for keyword in keywords:
            # 測試不包含日期的查詢
            for template in PROMPT_TEMPLATES:
                if "{date}" in template:
                    # 跳過需要日期的模板（稍後測試）
                    continue
                
                query = template.format(keyword=keyword, date="")
                if not query.strip():
                    continue
                
                total_tests += 1
                print(f"\n測試 {total_tests}: {query}")
                result = test_rag_search(query, event_name)
                print(format_result(result, query, event_type))
                
                if result["success"] and result["count"] > 0:
                    passed_tests += 1
                else:
                    failed_tests += 1
        
        # 測試包含日期的查詢（每個關鍵字測試幾個日期）
        for keyword in keywords[:2]:  # 只測試前2個關鍵字，避免測試太多
            for date in DATES[:3]:  # 只測試前3個日期
                if date is None:
                    continue
                
                for template in PROMPT_TEMPLATES:
                    if "{date}" not in template:
                        continue
                    
                    query = template.format(keyword=keyword, date=date)
                    if not query.strip():
                        continue
                    
                    total_tests += 1
                    print(f"\n測試 {total_tests}: {query}")
                    result = test_rag_search(query, event_name)
                    print(format_result(result, query))
                    
                    if result["success"] and result["count"] > 0:
                        passed_tests += 1
                    else:
                        failed_tests += 1
    
    print("\n" + "=" * 120)
    print("測試總結")
    print("=" * 120)
    print(f"總測試數: {total_tests}")
    print(f"✅ 通過: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"❌ 失敗: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    print("=" * 120)

if __name__ == "__main__":
    run_tests()

