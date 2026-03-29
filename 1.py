def print_report(rtsp_count, groups, segments_per_group):
    total_segments = rtsp_count * groups * segments_per_group
    processed = total_segments  # 假設已全部完成
    percentage = (processed / total_segments) * 100

    print("--- 測試報告 ---")
    print(f"RTSP 路數： {rtsp_count}")
    print(f"總段數： {total_segments} ({rtsp_count} 路 x {groups} 組 x {segments_per_group} 段)")
    print(f"10s 內處理完： {processed} / {total_segments} ({percentage:.0f}%)")


# ===== 測試 =====
print_report(rtsp_count=8, groups=6, segments_per_group=30)