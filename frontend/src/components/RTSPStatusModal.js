import React, { useState, useEffect, useRef } from 'react';
import apiService from '../services/api';

const RTSPStatusModal = ({ isOpen, onClose, apiKey }) => {
  const [url, setUrl] = useState("rtsp://rtsp-server:8554/live"); // Docker 內部地址
  const [videoId, setVideoId] = useState("CAM_01");
  const [activeStreams, setActiveStreams] = useState({});
  const [logs, setLogs] = useState([]);
  const [hlsStatus, setHlsStatus] = useState({ 
    canPlay: false, 
    message: "等待串流啟動...",
    streamEnded: false 
  });
  const [systemStatus, setSystemStatus] = useState({
    cpu: 0,
    ram: 0,
    gpu: null,
    models: { yolo_world: false, reid_model: false },
    modelsStr: "YOLO: ❌ Not Loaded | ReID: ❌ Not Loaded",
    disk: 0
  });

  // 為了讓前端能看到影片，我們需要用 MediaMTX 的 HLS 功能
  // 通過 nginx 代理訪問 HLS 流（避免防火牆問題）
  // 使用 /live/ 路徑，mediamtx 會自動提供 HLS 播放器頁面
  const [hlsUrl, setHlsUrl] = useState(`http://${window.location.hostname}:${window.location.port || 3000}/hls/live/`);

  useEffect(() => {
    // 使用 nginx 代理的 HLS 路徑，自動使用當前 hostname 和 port
    const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
    setHlsUrl(`http://${window.location.hostname}:${port}/hls/live/`);
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

  // 移除 HLS manifest 檢查，直接顯示畫面
  // 因為 MediaMTX 的 HLS 播放器頁面在 /live/，不需要檢查 manifest

  useEffect(() => {
    // 視窗打開時立即允許顯示 HLS（使用 stream-simulator 的推流）
    if (isOpen) {
      const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
      const hlsPath = `http://${window.location.hostname}:${port}/hls/live/`;
      setHlsStatus({ canPlay: true, message: "串流運行中", streamEnded: false });
      setHlsUrl(hlsPath);
      console.log("🚀 [HLS] Modal opened, using stream-simulator feed at", hlsPath);
    }
    
    // 只有在視窗開啟且有 apiKey 時才啟動輪詢
    if (!isOpen || !apiKey) {
      isFirstPollRef.current = true;
      logBufferRef.current = [];
      return;
    }

    const pollTask = async () => {
      try {
        // 0. 獲取系統狀態（CPU、GPU、RAM 等）
        try {
          const sysStatus = await apiService.getSystemStatus(apiKey);
          if (sysStatus) {
            // 格式化 GPU 狀態（與後端格式一致）
            let gpuStatusStr = null;
            if (sysStatus.gpu?.devices && sysStatus.gpu.devices.length > 0) {
              const gpu = sysStatus.gpu.devices[0];
              // 計算記憶體使用百分比
              const memPercent = gpu.mem_util_percent !== null && gpu.mem_util_percent !== undefined
                ? gpu.mem_util_percent
                : ((gpu.mem_used_mb / gpu.mem_total_mb) * 100);
              gpuStatusStr = `${gpu.name}: Mem ${memPercent.toFixed(2)}%`;
            }
            
            // 格式化模型狀態（與後端格式一致）
            const modelsStr = `YOLO: ${sysStatus.models?.yolo_world ? '✅ Loaded' : '❌ Not Loaded'} | ReID: ${sysStatus.models?.reid_model ? '✅ Loaded' : '❌ Not Loaded'}`;
            
            setSystemStatus({
              cpu: sysStatus.cpu?.percent || 0,
              ram: sysStatus.memory?.percent || 0,
              gpu: gpuStatusStr,
              models: sysStatus.models || { yolo_world: false, reid_model: false },
              modelsStr: modelsStr,
              disk: sysStatus.disk?.free_gb || 0
            });
          }
        } catch (e) {
          console.warn("Failed to fetch system status:", e);
        }

        // 1. 同步獲取串流狀態
        const status = await apiService.getRTSPStatus(apiKey);
        setActiveStreams(status || {});

        // 1.5. 直接允許播放 HLS（使用 stream-simulator 的推流）
        // 不需要檢查後端串流狀態，直接顯示 /live 的 HLS
        setHlsStatus(prev => {
          if (!prev.canPlay) {
            console.log("🚀 [HLS] Using stream-simulator feed at /live");
            return { 
              canPlay: true, 
              message: "串流運行中",
              streamEnded: false 
            };
          }
          return prev;
        });
        
        // 確保 HLS URL 正確設置（使用 nginx 代理）
        const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
        setHlsUrl(`http://${window.location.hostname}:${port}/hls/live/`);

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
              <div style={{ background: 'black', flex: 1, minHeight: '360px', marginBottom: '20px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #333', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {hlsStatus.canPlay ? (
                  <>
                    <iframe
                      src={hlsUrl.endsWith('/') ? hlsUrl : `${hlsUrl}/`}
                      style={{ width: '100%', height: '100%', border: 'none', position: 'absolute', top: 0, left: 0 }}
                      title="RTSP Preview"
                      allow="autoplay; fullscreen"
                      onError={(e) => {
                        // 處理 iframe 載入錯誤（雖然 iframe 的 onError 不總是觸發）
                        console.warn("⚠️ [HLS] Iframe load error detected");
                        // 檢查是否是因為串流已結束
                        const currentStream = activeStreams?.[videoId];
                        if (currentStream?.ended || currentStream?.status === "ended") {
                          setHlsStatus({ 
                            canPlay: false, 
                            message: "分析完成，串流已關閉",
                            streamEnded: true 
                          });
                        } else {
                          setHlsStatus({ 
                            canPlay: false, 
                            message: "HLS 串流載入失敗，請檢查 MediaMTX 服務",
                            streamEnded: false 
                          });
                        }
                      }}
                    />
                    <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.7)', padding: '4px 10px', borderRadius: '4px', fontSize: '11px', color: '#4CAF50', fontWeight: 'bold', zIndex: 10 }}>
                      ● LIVE (HLS) - GPU加速
                    </div>
                  </>
                ) : (
                  <div style={{ 
                    textAlign: 'center', 
                    color: hlsStatus.streamEnded ? '#4CAF50' : '#888',
                    padding: '40px',
                    width: '100%'
                  }}>
                    <div style={{ fontSize: '48px', marginBottom: '20px' }}>
                      {hlsStatus.streamEnded ? '✅' : '⏳'}
                    </div>
                    <div style={{ fontSize: '18px', marginBottom: '10px', fontWeight: 'bold' }}>
                      {hlsStatus.message}
                    </div>
                    {hlsStatus.streamEnded && (
                      <div style={{ fontSize: '14px', color: '#666', marginTop: '10px' }}>
                        串流已正常結束，分析結果已保存
                      </div>
                    )}
                    {!hlsStatus.streamEnded && !hlsStatus.canPlay && (
                      <div style={{ fontSize: '12px', color: '#555', marginTop: '10px' }}>
                        請先啟動 AI 分析以開始串流
                      </div>
                    )}
                  </div>
                )}
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

              {/* 系統監控面板 */}
              <div style={{ background: '#252525', padding: '15px', borderRadius: '8px', marginBottom: '15px', flex: '0 0 auto' }}>
                <h4 style={{ fontSize: '14px', marginBottom: '12px', color: '#60a5fa', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  📊 系統監控
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '12px' }}>
                  <div style={{ background: '#1a1a1a', padding: '10px', borderRadius: '6px', border: '1px solid #333' }}>
                    <div style={{ color: '#888', marginBottom: '4px' }}>CPU (%)</div>
                    <div style={{ color: '#4CAF50', fontSize: '18px', fontWeight: 'bold' }}>
                      {systemStatus.cpu.toFixed(1)}
                    </div>
                  </div>
                  <div style={{ background: '#1a1a1a', padding: '10px', borderRadius: '6px', border: '1px solid #333' }}>
                    <div style={{ color: '#888', marginBottom: '4px' }}>RAM (%)</div>
                    <div style={{ color: '#ff9800', fontSize: '18px', fontWeight: 'bold' }}>
                      {systemStatus.ram.toFixed(1)}
                    </div>
                  </div>
                  <div style={{ background: '#1a1a1a', padding: '10px', borderRadius: '6px', border: '1px solid #333', gridColumn: '1 / -1' }}>
                    <div style={{ color: '#888', marginBottom: '4px' }}>GPU Status</div>
                    <div style={{ color: '#a78bfa', fontSize: '13px', fontWeight: 'bold' }}>
                      {systemStatus.gpu || 'N/A'}
                    </div>
                  </div>
                  <div style={{ background: '#1a1a1a', padding: '10px', borderRadius: '6px', border: '1px solid #333', gridColumn: '1 / -1' }}>
                    <div style={{ color: '#888', marginBottom: '4px' }}>Models Loaded</div>
                    <div style={{ color: '#4CAF50', fontSize: '12px', fontWeight: 'bold' }}>
                      {systemStatus.modelsStr || 'YOLO: ❌ Not Loaded | ReID: ❌ Not Loaded'}
                    </div>
                  </div>
                  <div style={{ background: '#1a1a1a', padding: '10px', borderRadius: '6px', border: '1px solid #333', gridColumn: '1 / -1' }}>
                    <div style={{ color: '#888', marginBottom: '4px' }}>Disk Free</div>
                    <div style={{ color: '#60a5fa', fontSize: '13px', fontWeight: 'bold' }}>
                      {systemStatus.disk.toFixed(2)} GB
                    </div>
                  </div>
                </div>
              </div>

              <div style={{ background: '#252525', padding: '15px', borderRadius: '8px', flex: '0 0 auto' }}>
                <h4 style={{ fontSize: '14px', marginBottom: '10px', color: '#4CAF50' }}>● 運行中串流</h4>
                <div style={{ maxHeight: '120px', overflowY: 'auto' }}>
                  {Object.entries(activeStreams).length === 0 ? (
                    <div style={{ color: '#666', fontSize: '12px', textAlign: 'center', padding: '10px' }}>
                      目前沒有運行中的串流
                    </div>
                  ) : (
                    Object.entries(activeStreams).map(([id, info]) => {
                      const statusColor = info.status === "running" ? "#4CAF50" : 
                                        info.status === "ended" ? "#888" : "#ff4444";
                      const statusText = info.status === "running" ? "運行中" : 
                                        info.status === "ended" ? "已結束" : "錯誤";
                      return (
                        <div key={id} style={{ 
                          display: 'flex', 
                          justifyContent: 'space-between', 
                          alignItems: 'center', 
                          marginBottom: '6px', 
                          padding: '8px', 
                          background: '#333', 
                          borderRadius: '4px', 
                          fontSize: '13px',
                          borderLeft: `3px solid ${statusColor}`
                        }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <span style={{ fontWeight: 'bold' }}>{id}</span>
                              <span style={{ 
                                fontSize: '10px', 
                                color: statusColor,
                                background: `${statusColor}20`,
                                padding: '2px 6px',
                                borderRadius: '3px'
                              }}>
                                {statusText}
                              </span>
                            </div>
                            <small style={{ color: '#888', display: 'block', marginTop: '2px' }}>
                              {info.uptime}s | PID: {info.pid}
                            </small>
                            {info.error_message && (
                              <small style={{ color: '#ff6666', display: 'block', marginTop: '2px', fontSize: '11px' }}>
                                {info.error_message}
                              </small>
                            )}
                          </div>
                          {info.status === "running" && (
                            <button onClick={() => handleStop(id)} className="btn btn-danger" style={{ padding: '2px 10px', fontSize: '11px' }}>停止</button>
                          )}
                        </div>
                      );
                    })
                  )}
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