def print_test_report(rtsp_count, groups, segments_per_group, completed=None):
    total = rtsp_count * groups * segments_per_group
    if completed is None:
        completed = total
    percent = int(completed / total * 100)
    
    print("--- 測試報告 ---")
    print(f"RTSP 路數：{rtsp_count}")
    print(f"總段數：{total} ({rtsp_count} 路 x {groups} 每分鐘幾段 x {segments_per_group} 多少分鐘）")
    print(f"10s 內處理完：{completed} / {total} ({percent}%)")

# 範例
print_test_report(rtsp_count=12, groups=6, segments_per_group=30)