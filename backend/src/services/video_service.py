# -*- coding: utf-8 -*-
import os
import requests
import tempfile
import shutil
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.models import Summary
from src.config import config
import json

class VideoService:
    @staticmethod
    def list_videos(db: Optional[Session], events: Dict[str, Any], category: Optional[str] = None) -> List[Dict[str, Any]]:
        """獲取所有可用影片列表（綜合版：資料夾 + 資料庫）"""
        seg_root = config.SEGMENT_DIR
        if not seg_root.exists():
            seg_root.mkdir(parents=True, exist_ok=True)
        
        # 1. 取得所有資料夾名稱
        folder_video_ids = [d.name for d in seg_root.iterdir() if d.is_dir() and not d.name.startswith(('.', '_'))]
        print(f"--- [DEBUG] Found {len(folder_video_ids)} video folders in {seg_root} ---")
        
        # 2. 取得資料庫統計資訊
        db_stats = {}
        db_video_ids = set()
        if db:
            try:
                stats_query = db.query(
                    Summary.video,
                    func.count(Summary.id).label('count'),
                    func.max(Summary.created_at).label('mtime')
                ).filter(Summary.video != "").group_by(Summary.video).all()
                
                for s in stats_query:
                    db_stats[s.video] = {"count": s.count, "mtime": s.mtime}
                    db_video_ids.add(s.video)
                print(f"--- [DEBUG] Found {len(db_video_ids)} videos in database ---")
            except Exception as e:
                print(f"--- [ERROR] DB stats fetch failed: {e} ---")

        # 3. 合併所有 Video ID
        all_video_ids = sorted(list(set(folder_video_ids) | db_video_ids))
        
        videos = []
        for video_id in all_video_ids:
            # 獲取事件標籤資訊作為分類參考
            event_info = events.get(video_id, {})
            v_category = event_info.get("event_label") 
            
            # 分類篩選邏輯 (如果前端有傳入 category)
            if category and category != "all" and v_category != category:
                # 額外判斷路徑式分類
                if "/" in video_id:
                    path_category = video_id.split("/")[0]
                    if path_category != category:
                        continue
                else:
                    continue

            stats = db_stats.get(video_id, {"count": 0, "mtime": None})
            
            # 最後修改時間處理
            last_modified_ts = 0
            if stats["mtime"]:
                try:
                    if hasattr(stats["mtime"], 'timestamp'):
                        last_modified_ts = stats["mtime"].timestamp()
                    else:
                        # 處理其他可能的類型
                        last_modified_ts = time.mktime(stats["mtime"].timetuple())
                except: pass
            
            # 如果資料庫沒有時間，嘗試讀取資料夾時間
            if last_modified_ts == 0:
                try:
                    folder_path = seg_root / video_id
                    if folder_path.exists():
                        last_modified_ts = folder_path.stat().st_mtime
                except: pass

            videos.append({
                "video_id": video_id,
                "display_name": video_id,
                "source": "database" if video_id in db_video_ids else "segment",
                "total_segments": stats["count"],
                "success_segments": stats["count"],
                "model_type": "unknown", 
                "last_modified": last_modified_ts,
                "event_label": v_category,
                "event_description": event_info.get("event_description", ""),
                "category": None, # 配合舊版 frontend 邏輯
                "segment_id": video_id,
            })
        
        # 排序：有標籤的優先，然後按時間倒序
        videos.sort(key=lambda x: (x.get("event_label") is None, -x["last_modified"]))
        return videos

    @staticmethod
    def get_categories(db: Optional[Session], events: Dict[str, Any]) -> List[str]:
        """從事件配置與預定義清單中獲取分類"""
        categories = set()
        for v in events.values():
            if v.get("event_label"):
                categories.add(v["event_label"])
        
        predefined = ["火災生成", "水災生成", "人員倒地不起", "門禁遮臉入場", "車道併排阻塞", "離開吸菸區吸菸", "聚眾逗留", "安全門破壞/撬動", "其他"]
        for p in predefined:
            categories.add(p)
            
        return sorted(list(categories))

    @staticmethod
    def download_to_temp(url: str) -> str:
        suffix = Path(url).suffix or ".mp4"
        fd, tmp = tempfile.mkstemp(prefix="download_", suffix=suffix)
        os.close(fd)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return tmp

    @staticmethod
    def prepare_segments(
        video_path: Optional[str],
        video_id: Optional[str],
        target_filename: str,
        segment_duration: float,
        overlap: float,
        target_short: int,
        strict_segmentation: bool
    ):
        from src.utils.video_utils import _probe_duration_seconds, _split_one_video
        from src.config import config
        from fastapi import HTTPException
        import shutil

        # 1. 決定原始影片路徑與 Stem
        source_path = None
        stem = None

        if video_id and video_id.strip():
            video_id_clean = video_id.strip()
            print(f"--- [VideoService] 強制重新處理影片 ID: {video_id_clean} ---")
            
            # 優先嘗試還原路徑格式 (category/video_name)
            if "/" in video_id_clean:
                category, video_name = video_id_clean.split("/", 1)
                stem = f"{category}_{video_name}"
                potential_path = config.VIDEO_LIB_DIR / category / f"{video_name}.mp4"
                if not potential_path.exists():
                    for ext in ['.avi', '.mov', '.mkv', '.flv']:
                        p = config.VIDEO_LIB_DIR / category / f"{video_name}{ext}"
                        if p.exists(): potential_path = p; break
                
                if potential_path.exists():
                    source_path = str(potential_path)
            
            # 如果還是沒找到，嘗試在 segment 目錄找原始檔備份
            if not source_path:
                stem = video_id_clean.replace("/", "_")
                potential_path = config.SEGMENT_DIR / stem / f"{stem}.mp4"
                if not potential_path.exists():
                    for ext in ['.avi', '.mov', '.mkv', '.flv']:
                        p = config.SEGMENT_DIR / stem / f"{stem}{ext}"
                        if p.exists(): potential_path = p; break
                
                if potential_path.exists():
                    source_path = str(potential_path)

            if not source_path:
                # 最後手段：檢查 segment 目錄是否已經有片段了，如果有，則沿用（不建議，但在找不到原始檔時是唯一辦法）
                seg_dir = config.SEGMENT_DIR / stem
                if seg_dir.exists() and list(seg_dir.glob("segment_*.mp4")):
                    print(f"--- [VideoService] 警告：找不到原始檔，沿用現有片段 ---")
                    seg_files = sorted(seg_dir.glob("segment_*.mp4"))
                    total_duration = len(seg_files) * segment_duration
                    return seg_dir, seg_files, stem, total_duration
                raise HTTPException(status_code=404, detail=f"找不到影片 {video_id_clean} 的原始檔，無法重新執行切割")
        else:
            stem = Path(target_filename).stem
            source_path = video_path

        # 2. 強制執行 FFmpeg 切割流程
        seg_dir = config.SEGMENT_DIR / stem
        print(f"--- [VideoService] 開始執行正常流程：FFmpeg 切割 (Stem: {stem}) ---")
        
        # 為了確保「全新處理」，清理舊的片段資料夾
        if seg_dir.exists():
            print(f"--- [VideoService] 清理舊資料夾，確保重新處理: {seg_dir} ---")
            shutil.rmtree(seg_dir)
        seg_dir.mkdir(parents=True, exist_ok=True)

        # 如果是上傳的檔案，保存一份原始檔在 segment 目錄供日後重新處理
        if not video_id:
            backup_ext = Path(source_path).suffix
            backup_path = seg_dir / f"{stem}{backup_ext}"
            shutil.copy2(source_path, backup_path)
            source_path = str(backup_path)

        total_duration = _probe_duration_seconds(source_path)
        seg_files = _split_one_video(
            source_path, seg_dir, segment_duration, overlap,
            resolution=target_short if strict_segmentation else None
        )
        
        print(f"--- [VideoService] 切割完成，共 {len(seg_files)} 個片段 ---")
        return seg_dir, seg_files, stem, total_duration
