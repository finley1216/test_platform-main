# -*- coding: utf-8 -*-
"""
MCP 工具：日期時間解析
從用戶查詢中解析日期時間範圍
"""

import re
from datetime import datetime, date, time, timedelta
from typing import Dict, Any, Optional, Tuple
from zoneinfo import ZoneInfo

try:
    from date_extractor import extract_dates  # type: ignore  # 可選依賴，Docker 容器內可能未安裝
    HAS_DATE_EXTRACTOR = True
except ImportError:
    HAS_DATE_EXTRACTOR = False

TZ = ZoneInfo("Asia/Taipei")


def day_window(d: date) -> Tuple[datetime, datetime]:
    """Return (start_dt, end_dt) tz-aware in Asia/Taipei for whole day."""
    start = datetime.combine(d, time(0, 0, 0), tzinfo=TZ)
    end = start + timedelta(days=1)
    return start, end


def week_window(anchor: date) -> Tuple[datetime, datetime]:
    """Return Monday 00:00 ~ next Monday 00:00 for the week containing anchor."""
    monday = anchor - timedelta(days=anchor.weekday())  # Monday
    start = datetime.combine(monday, time(0, 0, 0), tzinfo=TZ)
    end = start + timedelta(days=7)
    return start, end


def parse_relative_date(text: str, now: datetime) -> Optional[Dict[str, Any]]:
    """
    Parse relative Chinese words into a date or week window.
    Returns dict or None.
    """
    today = now.date()

    # Day-relative
    if "今天" in text or "今日" in text:
        s, e = day_window(today)
        return {"mode": "RELATIVE_TODAY", "picked": today, "start": s, "end": e}

    if "昨天" in text:
        d = today - timedelta(days=1)
        s, e = day_window(d)
        return {"mode": "RELATIVE_YESTERDAY", "picked": d, "start": s, "end": e}

    if "前天" in text:
        d = today - timedelta(days=2)
        s, e = day_window(d)
        return {"mode": "RELATIVE_DAY_BEFORE_YESTERDAY", "picked": d, "start": s, "end": e}

    if "明天" in text:
        d = today + timedelta(days=1)
        s, e = day_window(d)
        return {"mode": "RELATIVE_TOMORROW", "picked": d, "start": s, "end": e}

    # Week-relative (whole week)
    if "本週" in text or "這週" in text or "本周" in text or "这周" in text:
        s, e = week_window(today)
        return {"mode": "RELATIVE_THIS_WEEK", "picked": today, "start": s, "end": e}

    if "上週" in text or "上周" in text:
        s, e = week_window(today - timedelta(days=7))
        return {"mode": "RELATIVE_LAST_WEEK", "picked": today - timedelta(days=7), "start": s, "end": e}

    if "下週" in text or "下周" in text:
        s, e = week_window(today + timedelta(days=7))
        return {"mode": "RELATIVE_NEXT_WEEK", "picked": today + timedelta(days=7), "start": s, "end": e}

    return None


def parse_mmdd_from_text(text: str) -> Optional[Tuple[int, int]]:
    """
    Return (month, day) if text contains:
      - standalone 4 digits MMDD (e.g., 1220)
      - or MM/DD or MM-DD (e.g., 12/20, 12-20)
    Otherwise return None.
    """
    # 1) MM/DD or MM-DD
    m = re.search(r"(?<!\d)(\d{1,2})[/-](\d{1,2})(?!\d)", text)
    if m:
        mm, dd = int(m.group(1)), int(m.group(2))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return mm, dd

    # 2) standalone 4 digits MMDD (排除 8 位數的 YYYYMMDD)
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", text)
    if m:
        mmdd = m.group(1)
        # 檢查是否為有效的 MMDD（前兩位應該是月份 01-12）
        mm, dd = int(mmdd[:2]), int(mmdd[2:])
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return mm, dd

    return None


def parse_yyyymmdd_from_text(text: str) -> Optional[Tuple[int, int, int]]:
    """
    Return (year, month, day) if text contains YYYYMMDD format (e.g., 20251220).
    Otherwise return None.
    """
    # 匹配 8 位數字的 YYYYMMDD
    m = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", text)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # 基本驗證：年份應該在合理範圍內（例如 2000-2100），月份 01-12，日期 01-31
        if 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            try:
                # 進一步驗證：檢查日期是否有效（例如 2 月 30 日會失敗）
                date(year, month, day)
                return year, month, day
            except ValueError:
                return None
    return None


def parse_query_time_window(query: str, now: Optional[datetime] = None, timezone: str = "Asia/Taipei") -> Dict[str, Any]:
    """
    解析查詢中的時間範圍，返回字典包含：
    - mode: 解析模式（RELATIVE_*, MMDD_RULE, YYYYMMDD_RULE, DATE_EXTRACTOR, NONE）
    - picked_date: 選中的日期（ISO 格式字串，或 None）
    - time_start: 開始時間（ISO 格式字串，或 None）
    - time_end: 結束時間（ISO 格式字串，或 None）
    - date_filter: date 對象（用於向後兼容，或 None）
    
    Args:
        query: 用戶查詢文字
        now: 當前時間（用於相對日期計算），如果為 None 則使用當前時間
        timezone: 時區名稱，默認為 "Asia/Taipei"
    """
    if now is None:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
    else:
        tz = now.tzinfo if now.tzinfo else ZoneInfo(timezone)

    # 1) Relative date words (fastest + best for today/yesterday)
    rel = parse_relative_date(query, now)
    if rel:
        return {
            "query": query,
            "mode": rel["mode"],
            "picked_date": rel["picked"].isoformat(),
            "time_start": rel["start"].isoformat(),
            "time_end": rel["end"].isoformat(),
            "date_filter": rel["picked"],  # 向後兼容
        }

    # 2) YYYYMMDD format (20251220) - 優先於 MMDD
    yyyymmdd = parse_yyyymmdd_from_text(query)
    if yyyymmdd:
        year, mm, dd = yyyymmdd
        d = date(year, mm, dd)
        start, end = day_window(d)
        return {
            "query": query,
            "mode": "YYYYMMDD_RULE",
            "picked_date": d.isoformat(),
            "time_start": start.isoformat(),
            "time_end": end.isoformat(),
            "date_filter": d,  # 向後兼容
        }

    # 3) MMDD shorthand (1220 / 12/20)
    mmdd = parse_mmdd_from_text(query)
    if mmdd:
        mm, dd = mmdd
        d = date(now.year, mm, dd)
        start, end = day_window(d)
        return {
            "query": query,
            "mode": "MMDD_RULE",
            "picked_date": d.isoformat(),
            "time_start": start.isoformat(),
            "time_end": end.isoformat(),
            "date_filter": d,  # 向後兼容
        }

    # 4) Fallback: date-extractor (for longer natural language dates)
    if HAS_DATE_EXTRACTOR:
        dates = extract_dates(query)
        if dates:
            # Choose first parsed date; convert tz if needed
            d0 = dates[0]
            if d0.tzinfo is None:
                d0 = d0.replace(tzinfo=tz)
            else:
                d0 = d0.astimezone(tz)

            start, end = day_window(d0.date())
            return {
                "query": query,
                "mode": "DATE_EXTRACTOR",
                "picked_date": d0.date().isoformat(),
                "time_start": start.isoformat(),
                "time_end": end.isoformat(),
                "date_filter": d0.date(),  # 向後兼容
            }

    return {
        "query": query,
        "mode": "NONE",
        "picked_date": None,
        "time_start": None,
        "time_end": None,
        "date_filter": None,
    }

