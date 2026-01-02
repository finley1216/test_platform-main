# -*- coding: utf-8 -*-
"""
影片管理相關 API
包含：列表、詳情、事件標籤、分類、移動
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any
from fastapi import APIRouter, Request, Depends, HTTPException
from sqlalchemy.orm import Session

# 從 main.py 導入必要的函數和變數
from src.main import (
    get_api_key, ADMIN_TOKEN, HAS_DB, get_db, VIDEO_LIB_DIR
)

router = APIRouter(tags=["影片管理"])

# 影片事件標籤存儲（簡單的 JSON 文件）
VIDEO_EVENTS_FILE = Path("segment") / "_video_events.json"

def _load_video_events() -> Dict[str, Dict[str, Any]]:
    """載入影片事件標籤"""
    if VIDEO_EVENTS_FILE.exists():
        try:
            with open(VIDEO_EVENTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_video_events(events: Dict[str, Dict[str, Any]]):
    """保存影片事件標籤"""
    VIDEO_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VIDEO_EVENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)

def _get_video_lib_categories() -> Dict[str, List[str]]:
    """獲取 video 資料夾中的分類和影片列表"""
    categories = {}
    if VIDEO_LIB_DIR.exists() and VIDEO_LIB_DIR.is_dir():
        for category_dir in VIDEO_LIB_DIR.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                video_files = []
                for video_file in category_dir.iterdir():
                    if video_file.is_file() and video_file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                        video_files.append(video_file.name)
                if video_files:
                    categories[category_name] = sorted(video_files)
    return categories

@router.get("/v1/videos/list", dependencies=[Depends(get_api_key)])
def list_videos():
    """獲取已上傳的影片列表（統一管理 segment 和 video 兩個位置）"""
    seg_dir = Path("segment")
    videos = []
    events = _load_video_events()
    video_lib_categories = _get_video_lib_categories()
    
    # 用於追蹤已處理的 video_lib 影片，避免重複顯示
    processed_video_lib = {}
    
    # 1. 從 segment 資料夾讀取已處理的影片
    if seg_dir.exists():
        for video_dir in seg_dir.iterdir():
            if video_dir.is_dir() and not video_dir.name.startswith("_"):
                segment_id = video_dir.name
                json_files = list(video_dir.glob("*.json"))
                if json_files:
                    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
                    try:
                        with open(latest_json, "r", encoding="utf-8") as f:
                            video_data = json.load(f)
                        
                        original_video_id = None
                        category = None
                        display_name = segment_id
                        
                        if "_" in segment_id:
                            parts = segment_id.split("_", 1)
                            if len(parts) == 2:
                                potential_category = parts[0]
                                potential_video_name = parts[1]
                                if potential_category in video_lib_categories:
                                    for vf in video_lib_categories[potential_category]:
                                        if Path(vf).stem == potential_video_name:
                                            original_video_id = f"{potential_category}/{potential_video_name}"
                                            category = potential_category
                                            display_name = vf
                                            processed_video_lib[original_video_id] = segment_id
                                            break
                        
                        video_info = {
                            "video_id": original_video_id if original_video_id else segment_id,
                            "display_name": display_name,
                            "source": "segment",
                            "json_path": str(latest_json.relative_to(Path("."))),
                            "total_segments": video_data.get("total_segments", 0),
                            "success_segments": video_data.get("success_segments", 0),
                            "model_type": video_data.get("model_type", "unknown"),
                            "last_modified": latest_json.stat().st_mtime,
                            "event_label": events.get(original_video_id or segment_id, {}).get("event_label") or (category if category else None),
                            "event_description": events.get(original_video_id or segment_id, {}).get("event_description", ""),
                            "category": category,
                            "segment_id": segment_id,
                        }
                        videos.append(video_info)
                    except Exception as e:
                        print(f"Warning: Failed to load video info for {segment_id}: {e}")
    
    # 2. 從 video 資料夾讀取歷史影片（按分類），但跳過已經在 segment 中處理過的
    for category_name, video_files in video_lib_categories.items():
        for video_file in video_files:
            video_id = f"{category_name}/{Path(video_file).stem}"
            
            if video_id in processed_video_lib:
                continue
            
            video_path = VIDEO_LIB_DIR / category_name / video_file
            
            event_info = events.get(video_id, {})
            if not event_info.get("event_label"):
                event_info = {"event_label": category_name, "event_description": ""}
            
            video_info = {
                "video_id": video_id,
                "display_name": video_file,
                "source": "video_lib",
                "json_path": None,
                "total_segments": 0,
                "success_segments": 0,
                "model_type": "unknown",
                "last_modified": video_path.stat().st_mtime if video_path.exists() else 0,
                "event_label": event_info.get("event_label", category_name),
                "event_description": event_info.get("event_description", ""),
                "category": category_name,
                "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)) if video_path.exists() else None,
            }
            videos.append(video_info)
    
    videos.sort(key=lambda x: x["last_modified"], reverse=True)
    
    return {
        "videos": videos,
        "total": len(videos),
        "categories": list(video_lib_categories.keys())
    }

@router.get("/v1/videos/{video_id:path}", dependencies=[Depends(get_api_key)])
def get_video_info(video_id: str):
    """獲取特定影片的詳細信息（支持 segment 和 video_lib 兩個來源）"""
    if "/" in video_id:
        category, video_name = video_id.split("/", 1)
        # 先檢查 segment 中是否有對應的處理結果（格式：{category}_{video_name}）
        stem = f"{category}_{video_name}"
        seg_dir = Path("segment") / stem
        
        # 如果 segment 中有結果，優先使用 segment 的結果
        if seg_dir.exists() and list(seg_dir.glob("*.json")):
            events = _load_video_events()
            event_info = events.get(video_id, {})
            
            json_files = list(seg_dir.glob("*.json"))
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                
                # 檢查 video_lib 中是否有原始影片
                video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
                if not video_path.exists():
                    for ext in ['.avi', '.mov', '.mkv', '.flv']:
                        video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                        if video_path.exists():
                            break
                
                return {
                    "video_id": video_id,
                    "display_name": video_path.name if video_path.exists() else f"{video_name}.mp4",
                    "source": "segment" if not video_path.exists() else "video_lib",
                    "json_path": str(latest_json.relative_to(Path("."))),
                    "analysis_data": video_data,
                    "event_label": event_info.get("event_label", category),
                    "event_description": event_info.get("event_description", ""),
                    "event_set_by": event_info.get("set_by", ""),
                    "event_set_at": event_info.get("set_at", ""),
                    "category": category,
                    "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)) if video_path.exists() else None,
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load video data: {e}")
        
        # 如果 segment 中沒有，檢查 video_lib
        video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
        if not video_path.exists():
            for ext in ['.avi', '.mov', '.mkv', '.flv']:
                video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                if video_path.exists():
                    break
        
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found in video library or segment")
        
        events = _load_video_events()
        event_info = events.get(video_id, {})
        
        json_files = list(seg_dir.glob("*.json")) if seg_dir.exists() else []
        if json_files:
            latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_json, "r", encoding="utf-8") as f:
                    video_data = json.load(f)
                return {
                    "video_id": video_id,
                    "display_name": video_path.name,
                    "source": "video_lib",
                    "json_path": str(latest_json.relative_to(Path("."))),
                    "analysis_data": video_data,
                    "event_label": event_info.get("event_label", category),
                    "event_description": event_info.get("event_description", ""),
                    "event_set_by": event_info.get("set_by", ""),
                    "event_set_at": event_info.get("set_at", ""),
                    "category": category,
                    "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)),
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load video data: {e}")
        else:
            return {
                "video_id": video_id,
                "display_name": video_path.name,
                "source": "video_lib",
                "json_path": None,
                "analysis_data": None,
                "event_label": event_info.get("event_label", category),
                "event_description": event_info.get("event_description", ""),
                "event_set_by": event_info.get("set_by", ""),
                "event_set_at": event_info.get("set_at", ""),
                "category": category,
                "video_path": str(video_path.relative_to(VIDEO_LIB_DIR.parent)),
            }
    else:
        seg_dir = Path("segment") / video_id
        if not seg_dir.exists():
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        
        json_files = list(seg_dir.glob("*.json"))
        if not json_files:
            raise HTTPException(status_code=404, detail=f"No analysis result found for {video_id}")
        
        latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest_json, "r", encoding="utf-8") as f:
                video_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load video data: {e}")
        
        events = _load_video_events()
        event_info = events.get(video_id, {})
        
        return {
            "video_id": video_id,
            "display_name": video_id,
            "source": "segment",
            "json_path": str(latest_json.relative_to(Path("."))),
            "analysis_data": video_data,
            "event_label": event_info.get("event_label"),
            "event_description": event_info.get("event_description", ""),
            "event_set_by": event_info.get("set_by", ""),
            "event_set_at": event_info.get("set_at", ""),
            "category": None,
        }

@router.post("/v1/videos/{video_id:path}/event", dependencies=[Depends(get_api_key)])
async def set_video_event(video_id: str, request: Request):
    """設置影片的事件標籤（管理者功能）"""
    video_exists = False
    
    if "/" in video_id:
        category, video_name = video_id.split("/", 1)
        video_path = VIDEO_LIB_DIR / category / f"{video_name}.mp4"
        if not video_path.exists():
            for ext in ['.avi', '.mov', '.mkv', '.flv']:
                video_path = VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                if video_path.exists():
                    break
        video_exists = video_path.exists()
    else:
        seg_dir = Path("segment") / video_id
        video_exists = seg_dir.exists()
    
    if not video_exists:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    
    payload = await request.json()
    event_label = payload.get("event_label", "").strip()
    event_description = payload.get("event_description", "").strip()
    
    if not event_label:
        raise HTTPException(status_code=422, detail="event_label is required")
    
    events = _load_video_events()
    events[video_id] = {
        "event_label": event_label,
        "event_description": event_description,
        "set_by": "admin",
        "set_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    _save_video_events(events)
    
    return {
        "success": True,
        "video_id": video_id,
        "event_label": event_label,
        "event_description": event_description,
        "message": f"影片 {video_id} 已標記為「{event_label}」"
    }

@router.delete("/v1/videos/{video_id:path}/event", dependencies=[Depends(get_api_key)])
def remove_video_event(video_id: str):
    """移除影片的事件標籤"""
    events = _load_video_events()
    if video_id in events:
        del events[video_id]
        _save_video_events(events)
        return {"success": True, "message": f"已移除影片 {video_id} 的事件標籤"}
    return {"success": False, "message": f"影片 {video_id} 沒有事件標籤"}

@router.get("/v1/videos/categories", dependencies=[Depends(get_api_key)])
def get_video_categories():
    """獲取 video 資料夾中的所有分類"""
    categories = _get_video_lib_categories()
    return {
        "categories": list(categories.keys()),
        "category_details": {cat: len(videos) for cat, videos in categories.items()}
    }

@router.post("/v1/videos/{video_id:path}/move", dependencies=[Depends(get_api_key)])
async def move_video_to_category(video_id: str, request: Request):
    """將影片移動到 video 資料夾的指定分類（管理者功能）"""
    api_key = request.headers.get("X-API-Key", "")
    if api_key != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="此功能僅限管理者使用")
    
    payload = await request.json()
    category = payload.get("category", "").strip()
    event_description = payload.get("event_description", "").strip()
    
    if not category:
        raise HTTPException(status_code=422, detail="category is required")
    
    seg_dir = Path("segment") / video_id
    if not seg_dir.exists():
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in segment. Only videos in segment can be moved to categories.")
    
    target_category_dir = VIDEO_LIB_DIR / category
    target_category_dir.mkdir(parents=True, exist_ok=True)
    
    original_video = None
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        potential_file = seg_dir / f"{video_id}{ext}"
        if potential_file.exists():
            original_video = potential_file
            break
    
    if not original_video:
        seg_files = sorted(seg_dir.glob("segment_*.mp4"))
        if seg_files:
            original_video = seg_files[0]
    
    if not original_video:
        raise HTTPException(status_code=404, detail=f"找不到影片文件：{video_id}")
    
    target_video_path = target_category_dir / original_video.name
    import shutil
    shutil.copy2(original_video, target_video_path)
    
    events = _load_video_events()
    new_video_id = f"{category}/{Path(original_video).stem}"
    events[new_video_id] = {
        "event_label": category,
        "event_description": event_description,
        "set_by": "admin",
        "set_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "moved_from": video_id
    }
    _save_video_events(events)
    
    return {
        "success": True,
        "message": f"影片已移動到分類「{category}」",
        "new_video_id": new_video_id,
        "target_path": str(target_video_path)
    }

