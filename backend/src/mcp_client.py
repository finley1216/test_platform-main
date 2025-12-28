# -*- coding: utf-8 -*-
"""
MCP 客戶端：用於調用 MCP 服務器
"""

import json
import subprocess
import os
from typing import Dict, Any, Optional
from pathlib import Path


class MCPDateParserClient:
    """MCP 日期解析客戶端"""
    
    def __init__(self, server_script: Optional[str] = None):
        """
        初始化 MCP 客戶端
        
        Args:
            server_script: MCP 服務器腳本路徑，如果為 None 則使用默認路徑
        """
        if server_script is None:
            # 默認使用 mcp/server.py
            script_dir = Path(__file__).parent
            server_script = str(script_dir / "mcp" / "server.py")
        
        self.server_script = server_script
        self.process = None
        self.request_id = 0
        self.initialized = False
    
    def _get_next_id(self) -> int:
        """獲取下一個請求 ID"""
        self.request_id += 1
        return self.request_id
    
    def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """發送 JSON-RPC 請求並獲取響應"""
        if params is None:
            params = {}
        
        request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": method,
            "params": params
        }
        
        # 如果進程未啟動，啟動它
        if self.process is None:
            # 設置工作目錄為腳本所在目錄
            script_dir = Path(self.server_script).parent
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{script_dir}:{script_dir.parent}:{env.get('PYTHONPATH', '')}"
            
            self.process = subprocess.Popen(
                ["python3", self.server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(script_dir),
                env=env
            )
            # 發送初始化請求
            try:
                init_request = {
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {}
                }
                init_json = json.dumps(init_request, ensure_ascii=False) + "\n"
                self.process.stdin.write(init_json)
                self.process.stdin.flush()
                
                # 讀取初始化響應
                init_line = self.process.stdout.readline()
                if init_line:
                    init_response = json.loads(init_line.strip())
                    if init_response.get("result"):
                        self.initialized = True
            except Exception as e:
                print(f"--- [WARNING] MCP 初始化失敗: {e} ---")
                # 繼續使用，讓後續請求來處理錯誤
        
        # 發送請求
        request_json = json.dumps(request, ensure_ascii=False) + "\n"
        self.process.stdin.write(request_json)
        self.process.stdin.flush()
        
        # 讀取響應
        response_line = self.process.stdout.readline()
        if not response_line:
            raise Exception("No response from MCP server")
        
        response = json.loads(response_line.strip())
        
        # 檢查錯誤
        if "error" in response:
            error = response["error"]
            raise Exception(f"MCP Error: {error.get('message', 'Unknown error')} - {error.get('data', '')}")
        
        return response.get("result", {})
    
    def parse_date(self, query: str) -> Dict[str, Any]:
        """
        解析日期
        
        Args:
            query: 用戶查詢文字
            
        Returns:
            日期解析結果字典
        """
        try:
            result = self._send_request("tools/call", {
                "name": "parse_date",
                "arguments": {
                    "query": query
                }
            })
            
            # 提取內容
            if "content" in result and len(result["content"]) > 0:
                content_text = result["content"][0].get("text", "{}")
                return json.loads(content_text)
            else:
                return {}
        except Exception as e:
            print(f"--- [WARNING] MCP 日期解析失敗: {e} ---")
            # 回退到直接調用（使用新的 MCP 工具）
            try:
                from mcp.tools.parse_time import parse_query_time_window
            except ImportError:
                from src.mcp.tools.parse_time import parse_query_time_window
            return parse_query_time_window(query)
    
    def close(self):
        """關閉 MCP 客戶端"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            self.initialized = False


# 全局客戶端實例（單例模式）
_global_client: Optional[MCPDateParserClient] = None


def get_mcp_client() -> MCPDateParserClient:
    """獲取全局 MCP 客戶端實例"""
    global _global_client
    if _global_client is None:
        _global_client = MCPDateParserClient()
    return _global_client


def parse_date_via_mcp(query: str) -> Dict[str, Any]:
    """
    通過 MCP 解析日期（便捷函數）
    
    Args:
        query: 用戶查詢文字
        
    Returns:
        日期解析結果字典
    """
    client = get_mcp_client()
    return client.parse_date(query)

