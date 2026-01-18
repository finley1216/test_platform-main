from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.services.rtsp_service import RTSPStreamManager

router = APIRouter(prefix="/v1/rtsp", tags=["RTSP Control"])

class StartStreamRequest(BaseModel):
    rtsp_url: str
    video_id: str
    segment_duration: int = 10

class StopStreamRequest(BaseModel):
    video_id: str

@router.post("/start")
def start_stream(req: StartStreamRequest):
    try:
        res = RTSPStreamManager.start_stream(req.rtsp_url, req.video_id, req.segment_duration)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
def stop_stream(req: StopStreamRequest):
    return RTSPStreamManager.stop_stream(req.video_id)

@router.get("/status")
def get_status():
    return RTSPStreamManager.get_status()