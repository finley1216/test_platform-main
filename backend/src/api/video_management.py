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

# 導入資料庫模型
if HAS_DB:
    from src.models import Summary
    from sqlalchemy import func

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

# 注意：具體路由必須放在路徑參數路由之前，否則會被攔截
@router.get("/v1/videos/list", dependencies=[Depends(get_api_key)])
def list_videos():
    """獲取已上傳的影片列表（完全從資料庫讀取，確保與資料庫中的 video 欄位一致）"""
    # 手動獲取 db session（避免條件表達式導致 FastAPI 路由失敗）
    db = None
    if HAS_DB:
        try:
            db = next(get_db())
        except Exception as e:
            print(f"--- [WARNING] 無法獲取資料庫連接: {e} ---")
    
    videos = []
    events = _load_video_events()
    
    # 只從資料庫讀取影片列表
    if HAS_DB and db:
        try:
            # 從資料庫查詢所有不同的 video 值及其統計信息
            video_stats = db.query(
                Summary.video,
                func.count(Summary.id).label('segment_count'),
                func.max(Summary.created_at).label('last_modified')
            ).filter(
                Summary.video.isnot(None),
                Summary.video != ""
            ).group_by(Summary.video).order_by(Summary.video).all()
            
            for video_name, segment_count, last_modified in video_stats:
                if not video_name:
                    continue
                
                # 使用資料庫中的 video 欄位值作為 video_id 和 display_name
                video_id = video_name
                display_name = video_name
                
                # 檢查 segment 資料夾中是否有對應的資料夾（僅用於獲取額外信息）
                seg_dir = Path("segment")
                segment_dir = seg_dir / video_name
                
                # 嘗試從 segment 資料夾獲取詳細信息（可選）
                json_files = list(segment_dir.glob("*.json")) if segment_dir.exists() else []
                video_data = {}
                if json_files:
                    latest_json = max(json_files, key=lambda p: p.stat().st_mtime)
                    try:
                        with open(latest_json, "r", encoding="utf-8") as f:
                            video_data = json.load(f)
                    except Exception:
                        pass
                
                # 計算實際的片段文件數量（可選）
                actual_segment_files = list(segment_dir.glob("segment_*.mp4")) if segment_dir.exists() else []
                actual_total_segments = len(actual_segment_files)
                
                # 使用資料庫中的片段數（優先），如果實際文件數更大則使用實際文件數
                total_segments = max(segment_count, actual_total_segments, video_data.get("total_segments", 0))
                success_segments = video_data.get("success_segments", 0)
                
                # 只使用 event_label，不使用 category 作為預設值
                # 使用 video_name 作為 key 查找事件標籤
                event_info = events.get(video_name, {})
                
                # 計算最後修改時間（優先使用資料庫時間）
                if last_modified:
                    if hasattr(last_modified, 'timestamp'):
                        last_modified_ts = last_modified.timestamp()
                    elif hasattr(last_modified, 'timetuple'):
                        last_modified_ts = time.mktime(last_modified.timetuple())
                    else:
                        last_modified_ts = 0
                else:
                    last_modified_ts = 0
                
                # 如果文件系統有更新時間，使用較新的時間
                if json_files:
                    file_mtime = max(p.stat().st_mtime for p in json_files)
                    last_modified_ts = max(last_modified_ts, file_mtime)
                
                video_info = {
                    "video_id": video_id,  # 完全使用資料庫中的 video 欄位值
                    "display_name": display_name,  # 完全使用資料庫中的 video 欄位值
                    "source": "database",  # 標記為從資料庫讀取
                    "json_path": str(latest_json.relative_to(Path("."))) if json_files else None,
                    "total_segments": total_segments,
                    "success_segments": success_segments,
                    "model_type": video_data.get("model_type", "unknown"),
                    "last_modified": last_modified_ts,
                    "event_label": event_info.get("event_label"),  # 只使用實際設置的 event_label，不使用 category 作為預設值
                    "event_description": event_info.get("event_description", ""),
                    "category": None,  # 不設置 category，所有影片都在"未分類"
                    "segment_id": video_name,  # 使用 video_name 作為 segment_id
                }
                videos.append(video_info)
        except Exception as e:
            print(f"Warning: Failed to load videos from database: {e}")
            import traceback
            traceback.print_exc()
    
    # 按最後修改時間排序
    videos.sort(key=lambda x: x["last_modified"], reverse=True)
    
    # 只返回實際有影片的事件類型（從 videos 列表中提取實際存在的 event_label）
    # 這樣可以過濾掉舊的、已經不在資料庫中的分類
    actual_event_types = set()
    for video in videos:
        if video.get("event_label"):
            actual_event_types.add(video["event_label"])
    
    # 按照預定義的事件類型順序排序
    predefined_event_types = [
        "火災生成",
        "水災生成",
        "人員倒地不起",
        "門禁遮臉入場",
        "車道併排阻塞",
        "離開吸菸區吸菸",
        "聚眾逗留",
        "安全門破壞/撬動",
        "其他",
    ]
    # 只返回預定義列表中且實際存在的事件類型
    filtered_event_types = [et for et in predefined_event_types if et in actual_event_types]
    
    return {
        "videos": videos,
        "total": len(videos),
        "categories": filtered_event_types  # 只返回實際有影片的事件類型
    }

@router.get("/v1/videos/categories", dependencies=[Depends(get_api_key)])
def get_video_categories():
    """獲取預定義的事件類型列表（與前端 EventTagModal 中的 eventTypes 一致）"""
    # 返回預定義的事件類型列表，與前端保持一致
    event_types = [
        "火災生成",
        "水災生成",
        "人員倒地不起",
        "門禁遮臉入場",
        "車道併排阻塞",
        "離開吸菸區吸菸",
        "聚眾逗留",
        "安全門破壞/撬動",
        "其他",
    ]
    return {
        "categories": event_types,  # 返回事件類型列表，而不是 video 資料夾中的分類
        "category_details": {}  # 不再返回分類詳情
    }

# 注意：子路由（/event, /move）必須放在主路由之前，避免被路徑參數路由攔截
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

@router.post("/v1/videos/{video_id:path}/move", dependencies=[Depends(get_api_key)])
async def move_video_to_category(video_id: str, request: Request):
    """將影片移動到 video 資料夾的指定分類（管理者功能）"""
    print(f"--- [DEBUG] move_video_to_category called with video_id: {video_id} ---")
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
    # 移動影片時，不自動設置 event_label，讓用戶手動設置
    # 只保存移動記錄，不設置事件標籤
    if new_video_id not in events:
        events[new_video_id] = {
            "event_label": None,  # 不自動設置，讓用戶手動設置
            "event_description": event_description,
            "set_by": "admin",
            "set_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "moved_from": video_id
        }
    else:
        # 如果已有事件標籤，只更新描述和移動記錄
        events[new_video_id]["moved_from"] = video_id
        if event_description:
            events[new_video_id]["event_description"] = event_description
    _save_video_events(events)
    
    return {
        "success": True,
        "message": f"影片已移動到分類「{category}」",
        "new_video_id": new_video_id,
        "target_path": str(target_video_path)
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
                    "event_label": event_info.get("event_label"),  # 只使用實際設置的 event_label
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
                    "event_label": event_info.get("event_label"),  # 只使用實際設置的 event_label
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
                    "event_label": event_info.get("event_label"),  # 只使用實際設置的 event_label
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

