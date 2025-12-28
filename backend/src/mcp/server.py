#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP (Model Context Protocol) 服務器：日期解析工具
使用 JSON-RPC 2.0 協議，通過標準輸入/輸出進行通信
"""

import sys
import json
import os
import yaml
from pathlib import Path
from typing import Dict, Any

# 設置 Python 路徑
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))

# 導入工具
from tools.parse_time import parse_query_time_window

# 載入配置
CONFIG_PATH = script_dir / "config.yaml"
config = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

# 從配置中讀取設定
TIMEZONE = config.get("timezone", "Asia/Taipei")
LOG_LEVEL = config.get("log_level", "INFO")
SERVER_NAME = config.get("server", {}).get("name", "date-parser-mcp-server")
SERVER_VERSION = config.get("server", {}).get("version", "1.0.0")
PROTOCOL_VERSION = config.get("server", {}).get("protocol_version", "2024-11-05")


def log(level: str, message: str):
    """簡單的日誌函數"""
    if LOG_LEVEL == "DEBUG" or (LOG_LEVEL == "INFO" and level in ["INFO", "WARNING", "ERROR"]):
        print(f"[{level}] {message}", file=sys.stderr, flush=True)


def handle_initialize(params: dict) -> dict:
    """處理 initialize 請求"""
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {}
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION
        }
    }


def handle_tools_list() -> dict:
    """返回可用工具列表"""
    # 嘗試從 schemas/tools.json 讀取工具規格
    schema_path = script_dir / "schemas" / "tools.json"
    if schema_path.exists():
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
                return schema_data
        except Exception as e:
            log("WARNING", f"無法讀取工具規格檔案: {e}")
    
    # 回退到硬編碼的工具列表
    return {
        "tools": [
            {
                "name": "parse_date",
                "description": "解析中文查詢中的日期時間範圍，支援相對日期（今天/昨天/本週）、MMDD格式（1220/12/20）、YYYYMMDD格式（20251220）等",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "用戶查詢文字，例如：'給我今天的影片'、'給我 1220 的影片'"
                        },
                        "timezone": {
                            "type": "string",
                            "description": "時區名稱（可選），默認為 'Asia/Taipei'",
                            "default": "Asia/Taipei"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]
    }


def handle_tools_call(tool_name: str, arguments: dict) -> dict:
    """處理工具調用"""
    if tool_name == "parse_date":
        query = arguments.get("query", "")
        timezone = arguments.get("timezone", TIMEZONE)
        
        if not query:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": "missing query parameter"
                        }, ensure_ascii=False)
                    }
                ]
            }
        
        try:
            # 調用日期解析函數
            result = parse_query_time_window(query, timezone=timezone)
            log("DEBUG", f"解析結果: {result.get('mode')} - {result.get('picked_date')}")
            
            # 返回結果
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False, default=str)
                    }
                ]
            }
        except Exception as e:
            log("ERROR", f"日期解析錯誤: {e}")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({
                            "error": str(e),
                            "mode": "NONE",
                            "picked_date": None,
                            "time_start": None,
                            "time_end": None,
                            "date_filter": None
                        }, ensure_ascii=False)
                    }
                ]
            }
    else:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "error": f"Unknown tool: {tool_name}"
                    }, ensure_ascii=False)
                }
            ]
        }


def handle_request(request: dict) -> dict:
    """處理 JSON-RPC 請求"""
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    try:
        if method == "initialize":
            result = handle_initialize(params)
        elif method == "tools/list":
            result = handle_tools_list()
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = handle_tools_call(tool_name, arguments)
        else:
            result = {"error": f"Unknown method: {method}"}
        
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }
    except Exception as e:
        log("ERROR", f"處理請求時發生錯誤: {e}")
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": "Internal error",
                "data": str(e)
            }
        }
    
    return response


def main():
    """主函數：從標準輸入讀取 JSON-RPC 請求，寫入標準輸出"""
    log("INFO", f"{SERVER_NAME} v{SERVER_VERSION} 啟動")
    
    # 處理請求
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            request = json.loads(line)
            log("DEBUG", f"收到請求: {request.get('method')}")
            response = handle_request(request)
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except json.JSONDecodeError as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if 'request' in locals() else None,
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": str(e)
                }
            }
            print(json.dumps(error_response, ensure_ascii=False), flush=True)
        except Exception as e:
            log("ERROR", f"未處理的錯誤: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if 'request' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e)
                }
            }
            print(json.dumps(error_response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

