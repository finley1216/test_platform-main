# -*- coding: utf-8 -*-
"""
偵測項目管理 API - 動態管理事件類型和更新 frame_prompt.md
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

# 先定義 router，避免循環導入問題
router = APIRouter(tags=["偵測項目管理"])

from src.database import get_db
from src.models import DetectionItem

# 從 main.py 導入必要的函數（在 router 定義之後）
from src.main import get_api_key

# ================== Pydantic Models ==================

class DetectionItemCreate(BaseModel):
    """創建偵測項目的請求模型"""
    name: str  # 唯一識別名稱
    name_en: str  # 英文名稱（用於 prompt）
    name_zh: str  # 中文名稱（用於顯示）
    description: Optional[str] = None  # 偵測標準描述
    is_enabled: bool = True  # 是否啟用

class DetectionItemUpdate(BaseModel):
    """更新偵測項目的請求模型"""
    name: Optional[str] = None
    name_en: Optional[str] = None
    name_zh: Optional[str] = None
    description: Optional[str] = None
    is_enabled: Optional[bool] = None

class DetectionItemResponse(BaseModel):
    """偵測項目的回應模型"""
    id: int
    name: str
    name_en: str
    name_zh: str
    description: Optional[str]
    is_enabled: bool
    alert_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

# ================== Helper Functions ==================

def generate_frame_prompt(db: Session) -> str:
    """
    根據資料庫中啟用的偵測項目生成 frame_prompt.md 內容
    """
    # 獲取所有啟用的偵測項目
    items = db.query(DetectionItem).filter(DetectionItem.is_enabled == True).order_by(DetectionItem.id).all()
    
    if not items:
        return """你是一個**嚴格的災害/人員異常偵測器**。請**僅依據畫面真實內容**，嚴格遵守以下 JSON 格式輸出，**不要猜測**：

{
  "events": {
    "reason": ""
  }
}

### 注意事項
- 目前沒有啟用的偵測項目
- 請在 reason 中描述畫面內容
- **只輸出純 JSON**；不要 Markdown/解釋/```json。
"""
    
    # 構建 JSON 格式的事件欄位
    event_fields = []
    for item in items:
        event_fields.append(f'    "{item.name_en}": false,')
    
    # 構建事件判斷標準
    event_standards = []
    for idx, item in enumerate(items, 1):
        desc = item.description if item.description else "請根據畫面判斷是否發生此事件"
        event_standards.append(f"{idx}) {item.name_en}（{item.name_zh}）：{desc} → **true**。")
    
    # 生成完整 prompt
    prompt = f"""你是一個**嚴格的災害/人員異常偵測器**。請**僅依據畫面真實內容**，嚴格遵守以下 JSON 格式輸出，**不要猜測**：

{{
  "events": {{
{chr(10).join(event_fields)}

    "reason": ""
  }}
}}

### 事件判斷標準（**一旦符合任一「明確徵象」就必須標記 true**；否則為 false）
{chr(10).join(event_standards)}

### 決斷與一致性
- 觀察到明確徵象就設 **true**；否則 **false**。
- 若為 true，`reason` 以「事件鍵：具體畫面證據」撰寫；**多事件以分號分隔**。
- **不得**在 `reason` 描述異常卻把事件設為 false。
- **只輸出純 JSON**；不要 Markdown/解釋/```json。
"""
    
    return prompt

def update_frame_prompt_file(db: Session) -> dict:
    """
    更新 frame_prompt.md 文件
    """
    try:
        # 生成新的 prompt 內容
        prompt_content = generate_frame_prompt(db)
        
        # 獲取 prompts 目錄路徑
        prompts_dir = Path(__file__).parent.parent.parent / "prompts"
        frame_prompt_path = prompts_dir / "frame_prompt.md"
        
        # 寫入文件
        frame_prompt_path.write_text(prompt_content, encoding="utf-8")
        
        return {
            "success": True,
            "message": "frame_prompt.md 已成功更新",
            "path": str(frame_prompt_path)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"更新 frame_prompt.md 失敗：{str(e)}",
            "error": str(e)
        }

# ================== API Endpoints ==================

@router.get("/detection-items", response_model=List[DetectionItemResponse], dependencies=[Depends(get_api_key)])
def list_detection_items(
    enabled_only: bool = False,
    db: Session = Depends(get_db)
):
    """
    列出所有偵測項目
    
    - **enabled_only**: 是否只列出啟用的項目
    """
    query = db.query(DetectionItem)
    if enabled_only:
        query = query.filter(DetectionItem.is_enabled == True)
    
    items = query.order_by(DetectionItem.id).all()
    return items

@router.get("/detection-items/{item_id}", response_model=DetectionItemResponse, dependencies=[Depends(get_api_key)])
def get_detection_item(
    item_id: int,
    db: Session = Depends(get_db)
):
    """獲取單個偵測項目的詳細資訊"""
    item = db.query(DetectionItem).filter(DetectionItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="偵測項目不存在")
    return item

@router.post("/detection-items", response_model=DetectionItemResponse, dependencies=[Depends(get_api_key)])
def create_detection_item(
    item_data: DetectionItemCreate,
    db: Session = Depends(get_db)
):
    """
    創建新的偵測項目
    
    創建後會自動更新 frame_prompt.md
    """
    # 檢查名稱是否已存在
    existing = db.query(DetectionItem).filter(DetectionItem.name == item_data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"偵測項目名稱 '{item_data.name}' 已存在")
    
    # 創建新項目
    new_item = DetectionItem(
        name=item_data.name,
        name_en=item_data.name_en,
        name_zh=item_data.name_zh,
        description=item_data.description,
        is_enabled=item_data.is_enabled
    )
    
    db.add(new_item)
    db.commit()
    db.refresh(new_item)
    
    # 更新 frame_prompt.md
    update_result = update_frame_prompt_file(db)
    if not update_result["success"]:
        print(f"[警告] {update_result['message']}")
    
    return new_item

@router.put("/detection-items/{item_id}", response_model=DetectionItemResponse, dependencies=[Depends(get_api_key)])
def update_detection_item(
    item_id: int,
    item_data: DetectionItemUpdate,
    db: Session = Depends(get_db)
):
    """
    更新偵測項目
    
    更新後會自動更新 frame_prompt.md
    """
    item = db.query(DetectionItem).filter(DetectionItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="偵測項目不存在")
    
    # 更新欄位
    if item_data.name is not None:
        # 檢查新名稱是否與其他項目衝突
        existing = db.query(DetectionItem).filter(
            DetectionItem.name == item_data.name,
            DetectionItem.id != item_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail=f"偵測項目名稱 '{item_data.name}' 已被使用")
        item.name = item_data.name
    
    if item_data.name_en is not None:
        item.name_en = item_data.name_en
    if item_data.name_zh is not None:
        item.name_zh = item_data.name_zh
    if item_data.description is not None:
        item.description = item_data.description
    if item_data.is_enabled is not None:
        item.is_enabled = item_data.is_enabled
    
    item.updated_at = datetime.now()
    
    db.commit()
    db.refresh(item)
    
    # 更新 frame_prompt.md
    update_result = update_frame_prompt_file(db)
    if not update_result["success"]:
        print(f"[警告] {update_result['message']}")
    
    return item

@router.delete("/detection-items/{item_id}", dependencies=[Depends(get_api_key)])
def delete_detection_item(
    item_id: int,
    db: Session = Depends(get_db)
):
    """
    刪除偵測項目
    
    刪除後會自動更新 frame_prompt.md
    """
    item = db.query(DetectionItem).filter(DetectionItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="偵測項目不存在")
    
    item_name = item.name_zh
    db.delete(item)
    db.commit()
    
    # 更新 frame_prompt.md
    update_result = update_frame_prompt_file(db)
    if not update_result["success"]:
        print(f"[警告] {update_result['message']}")
    
    return {
        "success": True,
        "message": f"偵測項目 '{item_name}' 已刪除",
        "prompt_update": update_result
    }

@router.post("/detection-items/regenerate-prompt", dependencies=[Depends(get_api_key)])
def regenerate_prompt(db: Session = Depends(get_db)):
    """
    手動重新生成 frame_prompt.md
    
    根據當前資料庫中啟用的偵測項目重新生成 prompt 文件
    """
    result = update_frame_prompt_file(db)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    
    # 獲取當前啟用的項目數量
    enabled_count = db.query(DetectionItem).filter(DetectionItem.is_enabled == True).count()
    
    return {
        **result,
        "enabled_items_count": enabled_count
    }

@router.get("/detection-items/preview-prompt/content", dependencies=[Depends(get_api_key)])
def preview_prompt(db: Session = Depends(get_db)):
    """
    預覽即將生成的 prompt 內容（不實際寫入文件）
    """
    prompt_content = generate_frame_prompt(db)
    enabled_count = db.query(DetectionItem).filter(DetectionItem.is_enabled == True).count()
    
    return {
        "prompt_content": prompt_content,
        "enabled_items_count": enabled_count
    }
