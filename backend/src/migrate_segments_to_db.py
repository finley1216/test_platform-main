#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration script: Convert segment JSON files to PostgreSQL database
Reads all JSON files from segment directory and imports summary_independent to Summary table
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path (script is now in src directory)
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from src.database import engine, SessionLocal, Base
from src.models import Summary
from src.config import config


def create_tables():
    """Create database tables if they don't exist"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created")


def parse_time_range(time_range_str: str) -> tuple:
    """
    Parse time range string like "00:00:00 - 00:00:08" to (start_time, end_time) datetime objects
    Returns (start_datetime, end_datetime) or (None, None) if parsing fails
    """
    try:
        if not time_range_str or not isinstance(time_range_str, str):
            return None, None
            
        if " - " in time_range_str:
            start_str, end_str = time_range_str.split(" - ", 1)
            start_str = start_str.strip()
            end_str = end_str.strip()
            
            # Parse HH:MM:SS format
            start_time_obj = datetime.strptime(start_str, "%H:%M:%S").time()
            end_time_obj = datetime.strptime(end_str, "%H:%M:%S").time()
            
            # Create datetime objects (using today's date as base)
            today = datetime.now().date()
            
            start_time = datetime.combine(today, start_time_obj)
            end_time = datetime.combine(today, end_time_obj)
            
            return start_time, end_time
    except ValueError as e:
        print(f"Warning: Could not parse time range '{time_range_str}': {e}")
    except Exception as e:
        print(f"Warning: Unexpected error parsing time range '{time_range_str}': {e}")
    
    return None, None


def process_json_file(json_path: Path) -> list:
    """
    Process a single JSON file and extract summary data
    Returns list of Summary objects to insert
    """
    summaries = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get video name from directory name
        video_dir = json_path.parent.name
        
        # Process each result in the JSON
        results = data.get("results", [])
        for result in results:
            if not result.get("success", False):
                continue
            
            parsed = result.get("parsed", {})
            summary_text = parsed.get("summary_independent", "")
            
            # Skip if no summary
            if not summary_text or not summary_text.strip():
                continue
            
            # Parse time range
            time_range = result.get("time_range", "")
            start_time, end_time = parse_time_range(time_range)
            
            # Get additional fields
            segment = result.get("segment", "")
            duration_sec = result.get("duration_sec")
            time_sec = result.get("time_sec")
            
            # Get event detection data from frame_analysis.events
            frame_analysis = parsed.get("frame_analysis", {})
            events = frame_analysis.get("events", {})
            
            # Extract video name from JSON file path (e.g., segment/fire_1/fire_1.json -> fire_1)
            video_name = json_path.parent.name if json_path.parent.name else None
            
            # Create Summary object
            summary = Summary(
                start_timestamp=start_time if start_time else datetime.now(),
                end_timestamp=end_time,
                location=None,  # Can be filled later
                camera=None,    # Can be filled later
                video=video_dir,  # [新增] 保存影片名稱（從目錄名稱提取），用於區分不同影片的相同 segment
                message=summary_text.strip(),
                segment=segment if segment else None,
                time_range=time_range if time_range else None,
                duration_sec=float(duration_sec) if duration_sec is not None else None,
                time_sec=float(time_sec) if time_sec is not None else None,
                # Event detection fields
                water_flood=bool(events.get("water_flood", False)),
                fire=bool(events.get("fire", False)),
                abnormal_attire_face_cover_at_entry=bool(events.get("abnormal_attire_face_cover_at_entry", False)),
                person_fallen_unmoving=bool(events.get("person_fallen_unmoving", False)),
                double_parking_lane_block=bool(events.get("double_parking_lane_block", False)),
                smoking_outside_zone=bool(events.get("smoking_outside_zone", False)),
                crowd_loitering=bool(events.get("crowd_loitering", False)),
                security_door_tamper=bool(events.get("security_door_tamper", False)),
                event_reason=events.get("reason", "") if events.get("reason") else None,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            summaries.append(summary)
    
    except Exception as e:
        print(f"  ✗ Error processing {json_path}: {e}")
    
    return summaries


def migrate_segments():
    """Main migration function"""
    print("=" * 60)
    print("Segment JSON to PostgreSQL Migration")
    print("=" * 60)
    
    # Create tables
    create_tables()
    
    # Get segment directory
    segment_dir = config.SEGMENT_DIR
    if not segment_dir.exists():
        print(f"Error: Segment directory not found: {segment_dir}")
        return
    
    print(f"\nScanning segment directory: {segment_dir}")
    
    # Find all JSON files (excluding _video_events.json)
    json_files = []
    for json_file in segment_dir.rglob("*.json"):
        if json_file.name != "_video_events.json":
            json_files.append(json_file)
    
    print(f"Found {len(json_files)} JSON files to process\n")
    
    if not json_files:
        print("No JSON files found. Exiting.")
        return
    
    # Process all files
    all_summaries = []
    for json_file in json_files:
        print(f"Processing: {json_file.relative_to(segment_dir)}")
        summaries = process_json_file(json_file)
        all_summaries.extend(summaries)
    
    print(f"\nTotal summaries extracted: {len(all_summaries)}")
    
    if not all_summaries:
        print("No summaries to insert. Exiting.")
        return
    
    # Insert/Update into database (與 RAG 邏輯一致：影片相同則更新，新的則新增)
    print("\nInserting/Updating summaries into database...")
    db = SessionLocal()
    try:
        inserted_count = 0
        updated_count = 0
        skipped_count = 0
        
        for summary in all_summaries:
            try:
                # Check if record already exists (by video, segment and time_range)
                # 與 RAG 的 _save_results_to_postgres 邏輯一致
                # [修改] 加入 video 欄位判斷，避免不同影片的相同 segment 和 time_range 被誤判為同一筆記錄
                existing = db.query(Summary).filter(
                    Summary.video == summary.video,
                    Summary.segment == summary.segment,
                    Summary.time_range == summary.time_range
                ).first()
                
                if existing:
                    # 更新現有記錄（與 RAG 邏輯一致：影片相同則更新）
                    existing.start_timestamp = summary.start_timestamp
                    existing.end_timestamp = summary.end_timestamp
                    existing.video = summary.video  # [新增] 確保 video 欄位也被更新
                    existing.message = summary.message
                    existing.segment = summary.segment
                    existing.time_range = summary.time_range
                    existing.duration_sec = summary.duration_sec
                    existing.time_sec = summary.time_sec
                    # 更新事件檢測欄位
                    existing.water_flood = summary.water_flood
                    existing.fire = summary.fire
                    existing.abnormal_attire_face_cover_at_entry = summary.abnormal_attire_face_cover_at_entry
                    existing.person_fallen_unmoving = summary.person_fallen_unmoving
                    existing.double_parking_lane_block = summary.double_parking_lane_block
                    existing.smoking_outside_zone = summary.smoking_outside_zone
                    existing.crowd_loitering = summary.crowd_loitering
                    existing.security_door_tamper = summary.security_door_tamper
                    existing.event_reason = summary.event_reason
                    # 更新 updated_at 時間戳
                    existing.updated_at = datetime.now()
                    updated_count += 1
                else:
                    # 新增記錄（新的則新增）
                    db.add(summary)
                    inserted_count += 1
            except Exception as e:
                print(f"  ✗ Error processing summary (segment: {summary.segment}, time_range: {summary.time_range}): {e}")
                skipped_count += 1
                continue
        
        db.commit()
        print(f"\n✓ Migration completed!")
        print(f"  - Inserted (new): {inserted_count}")
        print(f"  - Updated (existing): {updated_count}")
        print(f"  - Skipped (errors): {skipped_count}")
        print(f"  - Total processed: {len(all_summaries)}")
        
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    try:
        migrate_segments()
    except KeyboardInterrupt:
        print("\n\nMigration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

