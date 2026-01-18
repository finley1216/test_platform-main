import React, { useState, useEffect, useRef } from 'react';
import apiService from '../services/api';

const RTSPStatusModal = ({ isOpen, onClose, apiKey }) => {
  const [url, setUrl] = useState("rtsp://rtsp-server:8554/live"); // Docker 內部地址
  const [videoId, setVideoId] = useState("CAM_01");
  const [activeStreams, setActiveStreams] = useState({});
  const [logs, setLogs] = useState([]);

  // 為了讓前端能看到影片，我們需要用 MediaMTX 的 HLS 功能
  // 注意：這裡是瀏覽器存取，所以要用 localhost (如果你是在本機跑)
  // 或者是你的伺服器 IP
  const hlsUrl = "http://localhost:8888/live"; 

  const seenResultsRef = useRef(new Set());

  useEffect(() => {
    // 只有在視窗開啟且有 apiKey 時才啟動
    if (!isOpen || !apiKey) return;

    const pollTask = async () => {
      try {
        // 1. 同步獲取串流狀態
        const status = await apiService.getRTSPStatus(apiKey);
        setActiveStreams(status || {});

        // 2. 獲取分析進度
        const idsToTrack = [videoId, ...Object.keys(status || {})].filter(id => id);
        const uniqueIds = [...new Set(idsToTrack)];

        for (const id of uniqueIds) {
          try {
            const info = await apiService.getVideoInfo(id, apiKey);
            if (info.analysis_data?.results) {
              info.analysis_data.results.forEach(res => {
                const logKey = `${id}-${res.time_range}`;
                if (!seenResultsRef.current.has(logKey)) {
                  const summary = res.parsed?.summary_independent || "處理中...";
                  const events = res.parsed?.frame_analysis?.events || {};
                  const hasAnomaly = Object.values(events).some(v => v === true);
                  const logMsg = `[${new Date().toLocaleTimeString()}] ${id} 分析完成: ${summary.substring(0, 50)}${summary.length > 50 ? "..." : ""}${hasAnomaly ? " ⚠️ 偵測到異常！" : ""}`;
                  
                  setLogs(prev => [logMsg, ...prev].slice(0, 50));
                  seenResultsRef.current.add(logKey);
                }
              });
            }
          } catch (e) { /* 忽略個別影片失敗 */ }
        }
      } catch (e) {
        console.error("Polling failed", e);
      }
    };

    // 降低頻率為 5 秒一次，減少網路負擔
    const interval = setInterval(pollTask, 5000);
    pollTask(); // 立即執行第一次

    return () => {
      clearInterval(interval);
      seenResultsRef.current.clear();
    };
  }, [isOpen, apiKey, videoId]); // 只在視窗、密鑰或主要 ID 變更時啟動一次邏輯

  const handleStart = async () => {
    try {
      await apiService.startRTSP({ rtsp_url: url, video_id: videoId }, apiKey);
      setLogs(prev => [`[${new Date().toLocaleTimeString()}] 啟動串流: ${videoId}`, ...prev]);
    } catch (e) {
      alert("啟動失敗: " + e.message);
    }
  };

  const handleStop = async (id) => {
    try {
      await apiService.stopRTSP({ video_id: id }, apiKey);
      setLogs(prev => [`[${new Date().toLocaleTimeString()}] 停止串流: ${id}`, ...prev]);
    } catch (e) {
      alert("停止失敗: " + e.message);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '600px' }}>
        <div className="modal-header">
          <h3 className="modal-title">🎥 RTSP 監控台</h3>
          <button className="modal-close-btn" onClick={onClose}>×</button>
        </div>

        <div className="modal-body" style={{ background: '#1e1e1e', color: 'white' }}>
          {/* 預覽視窗 (嘗試播放 HLS) */}
          <div style={{ background: 'black', height: '300px', marginBottom: '20px', borderRadius: '8px', overflow: 'hidden', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '1px solid #333' }}>
             <p style={{color: '#aaa', textAlign: 'center', padding: '20px'}}>
               若 MediaMTX HLS (Port 8888) 有通，<br/>可在此預覽: <br/>
               <code style={{background: '#333', padding: '2px 5px', borderRadius: '4px', marginTop: '10px', display: 'inline-block'}}>{hlsUrl}</code>
             </p>
             <div style={{ fontSize: '12px', color: '#666', marginTop: '10px' }}>
               (建議使用支援 HLS 的播放器元件，例如 hls.js)
             </div>
          </div>

          <div style={{ marginBottom: '20px', background: '#2d2d2d', padding: '15px', borderRadius: '8px' }}>
            <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', color: '#aaa' }}>RTSP 串流網址</label>
            <input 
              value={url} onChange={e => setUrl(e.target.value)} 
              placeholder="rtsp://..." 
              style={{ width: '100%', marginBottom: '12px', padding: '10px', background: '#1a1a1a', border: '1px solid #444', borderRadius: '4px', color: 'white' }}
            />
            <div style={{ display: 'flex', gap: '10px' }}>
              <div style={{ flex: 1 }}>
                <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px', color: '#aaa' }}>影片 ID</label>
                <input 
                  value={videoId} onChange={e => setVideoId(e.target.value)} 
                  placeholder="e.g. CAM_01" 
                  style={{ width: '100%', padding: '10px', background: '#1a1a1a', border: '1px solid #444', borderRadius: '4px', color: 'white' }}
                />
              </div>
              <div style={{ display: 'flex', alignItems: 'flex-end' }}>
                <button onClick={handleStart} className="btn btn-primary" style={{ height: '42px', padding: '0 30px' }}>
                  啟動分析
                </button>
              </div>
            </div>
          </div>

          <div style={{ borderTop: '1px solid #444', paddingTop: '15px' }}>
            <h4 style={{ marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: Object.keys(activeStreams).length > 0 ? '#4CAF50' : '#666' }}></span>
              運行中串流:
            </h4>
            <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
              {Object.entries(activeStreams).map(([id, info]) => (
                <div key={id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', padding: '10px', background: '#2a2a2a', borderRadius: '6px' }}>
                  <div>
                    <span style={{ fontWeight: 'bold', color: '#4CAF50' }}>{id}</span>
                    <span style={{ marginLeft: '10px', fontSize: '12px', color: '#888' }}>PID: {info.pid} | Uptime: {info.uptime}s</span>
                  </div>
                  <button onClick={() => handleStop(id)} className="btn btn-danger" style={{ padding: '4px 12px', fontSize: '12px' }}>
                    停止
                  </button>
                </div>
              ))}
              {Object.keys(activeStreams).length === 0 && <p style={{ color: '#666', textAlign: 'center', padding: '10px' }}>無運行中串流</p>}
            </div>
          </div>

          <div style={{ marginTop: '20px', maxHeight: '120px', overflowY: 'auto', fontSize: '12px', background: '#000', padding: '10px', borderRadius: '4px', fontFamily: 'monospace' }}>
            {logs.length === 0 && <div style={{ color: '#444' }}>等待日誌...</div>}
            {logs.map((log, i) => <div key={i} style={{ color: '#888', marginBottom: '2px' }}>{log}</div>)}
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>關閉</button>
        </div>
      </div>
    </div>
  );
};

export default RTSPStatusModal;