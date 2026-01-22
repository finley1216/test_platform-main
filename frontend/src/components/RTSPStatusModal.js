import React, { useState, useEffect, useRef } from 'react';
import apiService from '../services/api';

const RTSPStatusModal = ({ isOpen, onClose, apiKey }) => {
  const [url, setUrl] = useState("rtsp://rtsp-server:8554/live"); // Docker 內部地址
  const [videoId, setVideoId] = useState("CAM_01");
  const [activeStreams, setActiveStreams] = useState({});
  const [logs, setLogs] = useState([]);

  // 為了讓前端能看到影片，我們需要用 MediaMTX 的 HLS 功能
  // 自動根據目前的網域動態生成 HLS URL
  const [hlsUrl, setHlsUrl] = useState(`http://${window.location.hostname}:8888/live`);

  useEffect(() => {
    // 監聽網域變化（通常不會變，但初始化時很重要）
    setHlsUrl(`http://${window.location.hostname}:8888/live`);
  }, []);

  const seenResultsRef = useRef(new Set());
  const isFirstPollRef = useRef(true);
  const logBufferRef = useRef([]); // 緩衝隊列

  // 新增：均速釋放日誌的計時器
  useEffect(() => {
    const releaseInterval = setInterval(() => {
      if (logBufferRef.current.length > 0) {
        // 從緩衝區取出最舊的一筆
        const nextLog = logBufferRef.current.shift();
        setLogs(prev => [nextLog, ...prev].slice(0, 100));
      }
    }, 800); // 每 0.8 秒釋放一筆，讓視覺更平滑

    return () => clearInterval(releaseInterval);
  }, []);

  useEffect(() => {
    // 只有在視窗開啟且有 apiKey 時才啟動
    if (!isOpen || !apiKey) {
      isFirstPollRef.current = true;
      logBufferRef.current = [];
      return;
    }

    const pollTask = async () => {
      try {
        // 1. 同步獲取串流狀態
        const status = await apiService.getRTSPStatus(apiKey);
        setActiveStreams(status || {});

        // 2. 獲取分析進度 (改為併行請求以減少卡頓)
        const idsToTrack = [videoId, ...Object.keys(status || {})].filter(id => id);
        const uniqueIds = [...new Set(idsToTrack)];

        const infoResults = await Promise.all(
          uniqueIds.map(id => apiService.getVideoInfo(id, apiKey).catch(() => null))
        );

        let newLogItems = [];
        
        infoResults.forEach((info, index) => {
          if (!info || !info.analysis_data?.results) return;
          const id = uniqueIds[index];
          
          // 確保結果按時間順序排序
          const sortedResults = [...info.analysis_data.results].sort((a, b) => a.time_range.localeCompare(b.time_range));

          sortedResults.forEach(res => {
            const logKey = `${id}-${res.time_range}`;
            if (!seenResultsRef.current.has(logKey)) {
              seenResultsRef.current.add(logKey);
              
              // 如果是進入後的第一次輪詢，只紀錄 key 不顯示
              if (isFirstPollRef.current) return;

              // [修改] 只提取 Event Detection Prompt 的結果
              const eventObj = res.parsed?.frame_analysis?.events || {};
              const reason = eventObj.reason || "";
              
              const detectedEvents = Object.entries(eventObj)
                .filter(([key, value]) => key !== "reason" && value === true)
                .map(([key, _]) => {
                  const names = {
                    fire: "火災",
                    water_flood: "水災",
                    person_fallen: "倒地",
                    double_parking: "併排",
                    smoking: "吸菸",
                    crowd: "聚眾",
                    security_door: "門禁異常",
                    abnormal_attire: "遮臉"
                  };
                  return names[key] || key;
                });

              // 優先顯示偵測到的事件名稱，若無則顯示無異常
              let eventStr = detectedEvents.length > 0 
                ? `偵測到：${detectedEvents.join(", ")}` 
                : "無異常";
              
              // 如果有理由，也併入顯示
              if (detectedEvents.length > 0 && reason) {
                eventStr += ` (${reason})`;
              }
              
              logBufferRef.current.push({
                time: new Date().toLocaleTimeString(),
                id: id,
                range: res.time_range,
                eventStr: eventStr,
                isCritical: detectedEvents.length > 0
              });
            }
          });
        });
        
        isFirstPollRef.current = false;
      } catch (e) {
        console.error("Polling failed", e);
      }
    };

    // 恢復較快的頻率以符合「極速模式」需求
    const interval = setInterval(pollTask, 2000);
    pollTask(); // 立即執行第一次

    return () => {
      clearInterval(interval);
      seenResultsRef.current.clear();
      setLogs([]); // 離開時清空
    };
  }, [isOpen, apiKey, videoId]); // 只在視窗、密鑰或主要 ID 變更時啟動一次邏輯

  const handleStart = async () => {
    try {
      await apiService.startRTSP({ rtsp_url: url, video_id: videoId }, apiKey);
      setLogs(prev => [{
        time: new Date().toLocaleTimeString(),
        id: "系統",
        eventStr: `啟動串流: ${videoId}`,
        isCritical: false
      }, ...prev]);
    } catch (e) {
      alert("啟動失敗: " + e.message);
    }
  };

  const handleStop = async (id) => {
    try {
      await apiService.stopRTSP({ video_id: id }, apiKey);
      setLogs(prev => [{
        time: new Date().toLocaleTimeString(),
        id: "系統",
        eventStr: `停止串流: ${id}`,
        isCritical: false
      }, ...prev]);
    } catch (e) {
      alert("停止失敗: " + e.message);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '1100px', width: '95%' }}>
        <div className="modal-header">
          <h3 className="modal-title">🎥 RTSP 監控台</h3>
          <button className="modal-close-btn" onClick={onClose}>×</button>
        </div>

        <div className="modal-body" style={{ background: '#1e1e1e', color: 'white', padding: '20px' }}>
          <div style={{ display: 'flex', gap: '25px', height: '650px' }}>
            
            {/* 左側：影片與控制 */}
            <div style={{ flex: '1.4', display: 'flex', flexDirection: 'column' }}>
              <div style={{ background: 'black', flex: 1, minHeight: '360px', marginBottom: '20px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #333', position: 'relative' }}>
                 <iframe
                   src={`http://${window.location.hostname}:8888/live/`}
                   style={{ width: '100%', height: '100%', border: 'none' }}
                   title="RTSP Preview"
                   allow="autoplay; fullscreen"
                 />
                 <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.5)', padding: '2px 8px', borderRadius: '4px', fontSize: '10px', color: '#ff9800' }}>
                   LIVE (HLS)
                 </div>
              </div>

              <div style={{ background: '#2d2d2d', padding: '15px', borderRadius: '8px', marginBottom: '15px' }}>
                <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                  <div style={{ flex: 1 }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px', color: '#aaa' }}>RTSP 網址</label>
                    <input 
                      value={url} onChange={e => setUrl(e.target.value)} 
                      style={{ width: '100%', padding: '8px', background: '#1a1a1a', border: '1px solid #444', borderRadius: '4px', color: 'white', fontSize: '13px' }}
                    />
                  </div>
                  <div style={{ width: '120px' }}>
                    <label style={{ display: 'block', marginBottom: '5px', fontSize: '12px', color: '#aaa' }}>影片 ID</label>
                    <input 
                      value={videoId} onChange={e => setVideoId(e.target.value)} 
                      style={{ width: '100%', padding: '8px', background: '#1a1a1a', border: '1px solid #444', borderRadius: '4px', color: 'white', fontSize: '13px' }}
                    />
                  </div>
                </div>
                <button onClick={handleStart} className="btn btn-primary" style={{ width: '100%', height: '38px' }}>
                  啟動 AI 分析
                </button>
              </div>

              <div style={{ background: '#252525', padding: '15px', borderRadius: '8px', flex: '0 0 auto' }}>
                <h4 style={{ fontSize: '14px', marginBottom: '10px', color: '#4CAF50' }}>● 運行中串流</h4>
                <div style={{ maxHeight: '100px', overflowY: 'auto' }}>
                  {Object.entries(activeStreams).map(([id, info]) => (
                    <div key={id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px', padding: '8px', background: '#333', borderRadius: '4px', fontSize: '13px' }}>
                      <span>{id} <small style={{ color: '#888', marginLeft: '5px' }}>({info.uptime}s)</small></span>
                      <button onClick={() => handleStop(id)} className="btn btn-danger" style={{ padding: '2px 10px', fontSize: '11px' }}>停止</button>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* 右側：顯目的事件日誌 */}
            <div style={{ flex: '1', display: 'flex', flexDirection: 'column', background: '#000', borderRadius: '8px', border: '1px solid #333' }}>
              <div style={{ padding: '12px 15px', borderBottom: '1px solid #333', background: '#111', borderTopLeftRadius: '8px', borderTopRightRadius: '8px' }}>
                <h4 style={{ margin: 0, fontSize: '15px', color: '#ff9800' }}>即時分析日誌</h4>
              </div>
              
              <div style={{ flex: 1, overflowY: 'auto', padding: '10px' }}>
                {logs.length === 0 && <div style={{ color: '#444', textAlign: 'center', marginTop: '20px' }}>等待偵測資料...</div>}
                {logs.map((log, i) => (
                    <div key={i} style={{ 
                      padding: '10px', 
                      marginBottom: '8px', 
                      background: log.isCritical ? '#451a1a' : '#1a1a1a', 
                      borderRadius: '6px',
                      borderLeft: `4px solid ${log.isCritical ? '#ff4444' : '#444'}`,
                      animation: i === 0 ? 'fadeIn 0.3s ease-out' : 'none'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <span style={{ color: '#888', fontSize: '11px' }}>{log.time}</span>
                        <span style={{ color: '#4CAF50', fontSize: '11px', fontWeight: 'bold' }}>{log.id}</span>
                      </div>
                      <div style={{ 
                        fontSize: '14px', 
                        color: log.isCritical ? '#ff6666' : '#eee',
                        fontWeight: log.isCritical ? 'bold' : 'normal'
                      }}>
                        {log.eventStr}
                      </div>
                      {log.range && <div style={{ fontSize: '10px', color: '#555', marginTop: '4px' }}>片段: {log.range}</div>}
                    </div>
                ))}
              </div>
            </div>

          </div>
        </div>

        <div className="modal-footer">
          <style>{`
            @keyframes fadeIn {
              from { opacity: 0; transform: translateY(-10px); }
              to { opacity: 1; transform: translateY(0); }
            }
          `}</style>
          <button className="btn btn-secondary" onClick={onClose}>關閉</button>
        </div>
      </div>
    </div>
  );
};

export default RTSPStatusModal;