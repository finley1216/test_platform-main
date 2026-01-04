#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同步資料庫中的 video 欄位與 segment 資料夾名稱
將資料庫中的 video 欄位更新為對應的 segment 資料夾名稱
"""

import sys
import os
from pathlib import Path

# 添加 src 目錄到路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 允許通過環境變數覆蓋資料庫連接
if not os.getenv("POSTGRES_HOST") and not os.getenv("DATABASE_URL"):
    os.environ["POSTGRES_HOST"] = "localhost"

from src.database import SessionLocal
from src.models import Summary

def normalize_video_name(video_name: str) -> str:
    """
    標準化影片名稱（去掉副檔名）
    例如：Video_突破安全門3.avi -> Video_突破安全門3
    """
    if not video_name:
        return video_name
    
    # 去掉常見的副檔名
    for ext in ['.avi', '.mp4', '.mov', '.mkv', '.flv', '.AVI', '.MP4', '.MOV', '.MKV', '.FLV']:
        if video_name.endswith(ext):
            return video_name[:-len(ext)]
    
    return video_name

def main():
    """主函數：同步資料庫中的 video 欄位與 segment 資料夾名稱"""
    import sys
    
    # 檢查是否有 --preview 參數
    preview_mode = "--preview" in sys.argv or "-p" in sys.argv
    
    db = SessionLocal()
    seg_dir = Path("segment")
    
    try:
        print("=" * 80)
        if preview_mode:
            print("預覽模式：查看將要更新的記錄（不會實際修改資料庫）")
        else:
            print("開始同步資料庫中的 video 欄位與 segment 資料夾名稱...")
        print("=" * 80)
        
        if not seg_dir.exists():
            print("❌ segment 資料夾不存在")
            return
        
        # 獲取所有 segment 資料夾名稱
        segment_folders = []
        for folder in seg_dir.iterdir():
            if folder.is_dir() and not folder.name.startswith("_"):
                segment_folders.append(folder.name)
        
        print(f"\n找到 {len(segment_folders)} 個 segment 資料夾:")
        for folder in sorted(segment_folders):
            print(f"  - {folder}")
        
        print(f"\n開始檢查資料庫記錄...")
        
        # 查詢所有需要更新的記錄
        all_records = db.query(Summary).filter(Summary.video.isnot(None)).all()
        
        print(f"資料庫中共有 {len(all_records)} 筆有 video 欄位的記錄")
        
        update_map = {}  # 舊名稱 -> 新名稱的映射
        fixed_count = 0
        unchanged_count = 0
        not_found_count = 0
        
        # 建立 segment 資料夾名稱的集合（標準化後）
        segment_names_normalized = {normalize_video_name(name): name for name in segment_folders}
        
        for record in all_records:
            old_video = record.video
            normalized_old = normalize_video_name(old_video)
            
            # 檢查標準化後的名稱是否在 segment 資料夾中
            if normalized_old in segment_names_normalized:
                # 使用 segment 資料夾的實際名稱（標準化後）
                new_video = normalized_old
                
                if old_video != new_video:
                    if old_video not in update_map:
                        update_map[old_video] = new_video
                        print(f"\n發現需要更新:")
                        print(f"  舊名稱: {old_video}")
                        print(f"  新名稱: {new_video}")
                    
                    # 更新記錄
                    if not preview_mode:
                        record.video = new_video
                    fixed_count += 1
                else:
                    unchanged_count += 1
            else:
                # 檢查是否只是副檔名不同
                found_match = False
                for seg_name in segment_folders:
                    if normalize_video_name(seg_name) == normalized_old:
                        new_video = seg_name
                        if old_video not in update_map:
                            update_map[old_video] = new_video
                            print(f"\n發現需要更新（副檔名不同）:")
                            print(f"  舊名稱: {old_video}")
                            print(f"  新名稱: {new_video}")
                        
                        if not preview_mode:
                            record.video = new_video
                        fixed_count += 1
                        found_match = True
                        break
                
                if not found_match:
                    not_found_count += 1
        
        if fixed_count > 0:
            if preview_mode:
                print("\n" + "=" * 80)
                print(f"預覽完成：將更新 {fixed_count} 筆記錄")
                print(f"  未變更: {unchanged_count} 筆記錄")
                print(f"  找不到對應的 segment 資料夾: {not_found_count} 筆記錄")
                print("\n要執行實際更新，請運行：")
                print("  python sync_video_names.py")
                print("=" * 80)
                db.rollback()
            else:
                # 提交更改
                db.commit()
                print("\n" + "=" * 80)
                print(f"✓ 成功更新 {fixed_count} 筆記錄")
                print(f"  未變更: {unchanged_count} 筆記錄")
                print(f"  找不到對應的 segment 資料夾: {not_found_count} 筆記錄")
                print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("沒有需要更新的記錄")
            if not_found_count > 0:
                print(f"  找不到對應的 segment 資料夾: {not_found_count} 筆記錄")
            print("=" * 80)
            db.rollback()
        
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()

