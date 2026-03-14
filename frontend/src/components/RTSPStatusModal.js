import React, { useState, useEffect, useRef, useMemo } from 'react';
import apiService from '../services/api';

const LOCAL_VIDEO_FILENAME = 'Video_衛哨端出入口.mp4';

// 指向前端的 public 資料夾，讓瀏覽器能直接讀取影片。
const LOCAL_VIDEO_STATIC_URL = `${process.env.PUBLIC_URL || ''}/video/${LOCAL_VIDEO_FILENAME}`;

// 是給後端使用的路徑標記，通常對應伺服器上的某個分析路徑。
const LOCAL_VIDEO_ID = '門禁遮臉入場/Video_衛哨端出入口';

const RTSPStatusModal = ({ isOpen, onClose, apiKey }) => {

  // RTSP 位址
  const [url, setUrl] = useState("rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream");
  // 影片 ID
  const [videoId, setVideoId] = useState("CAM_01");
  const [activeStreams, setActiveStreams] = useState({});
  const [logs, setLogs] = useState([]);

  // 用於判斷串流是否可播放。
  const [hlsStatus, setHlsStatus] = useState({ 
    canPlay: false, 
    message: "等待串流啟動...",
    streamEnded: false 
  });

  // 系統狀態：CPU、RAM、GPU、模型載入狀態、硬碟空間等。
  const [systemStatus, setSystemStatus] = useState({
    cpu: 0,
    ram: 0,
    gpu: null,
    models: { yolo_world: false, reid_model: false },
    modelsStr: "YOLO: ❌ Not Loaded | ReID: ❌ Not Loaded",
    disk: 0
  });

  // RTSP 串流通常無法在瀏覽器直接播放。這裡假設後端使用了 MediaMTX 將 RTSP 轉成 HLS (HTTP Live Streaming)。透過 Nginx 代理轉發到 /hls/live/ 進行播放。
  const [hlsUrl, setHlsUrl] = useState(`http://${window.location.hostname}:${window.location.port || 3000}/hls/live/`);
  
  // 檢視模式：'rtsp' = 即時 HLS 串流（原有），'local' = 本地影片（門禁遮臉入場）
  const [viewMode, setViewMode] = useState('local');

  // 本地影片載入錯誤（無法播放時顯示）
  const [localVideoError, setLocalVideoError] = useState(null);
  const [localVideoReady, setLocalVideoReady] = useState(false);
  const [isLocalAnalyzing, setIsLocalAnalyzing] = useState(false);
  const [localAnalysisResult, setLocalAnalysisResult] = useState(null);
  const localVideoUrl = LOCAL_VIDEO_STATIC_URL;

  useEffect(() => {
    // 使用 nginx 代理的 HLS 路徑，自動使用當前 hostname 和 port
    const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
    setHlsUrl(`http://${window.location.hostname}:${port}/hls/live/`);
  }, []);

  // 使用 Set 紀錄已顯示過的結果，防止重複顯示舊資訊。
  const seenResultsRef = useRef(new Set());
  const isFirstPollRef = useRef(true);
  const pollCountRef = useRef(0); // 打開 modal 以來的輪詢次數，用來區分「一打開就有的舊資料」vs「分析剛跑完的新結果」
  const initialBatchKeysRef = useRef(new Set()); // 一打開就存在的那批 logKey，永不顯示（避免競態下仍跳出 30 筆）
  const logBufferRef = useRef([]); // 緩衝隊列

  // 播放同步單段分析：已分析過的 10 秒區間索引（0, 1, 2...），onTimeUpdate 跨區時觸發 /v1/analyze_single_segment
  const analyzedSegmentsRef = useRef(new Set());
  const lastSegmentIndexRef = useRef(-1);
  const segmentAnalysisInFlightRef = useRef(false); // 同一時間只發一個單段分析請求，避免多個長連線導致 proxy/連線逾時
  const videoRef = useRef(null);

  // 這是一個標記（Flag），確保在同一次打開 Modal 期間，AI 分析只會被自動觸發一次，不會因為 React 的重複渲染（Re-render）導致不斷發送 API 請求。
  const autoAnalysisTriggeredRef = useRef(false);

  // 打開 modal 時清成「這次分析」的乾淨狀態，不混用上次或後端其他任務的結果
  useEffect(() => {
    if (!isOpen) return;
    setLogs([]);
    setLocalAnalysisResult(null);
    logBufferRef.current = [];
    analyzedSegmentsRef.current.clear();
    lastSegmentIndexRef.current = -1;
    segmentAnalysisInFlightRef.current = false;
    seenResultsRef.current.clear();
    initialBatchKeysRef.current.clear();
    isFirstPollRef.current = true;
    pollCountRef.current = 0;
  }, [isOpen]);

  





  // 日誌緩衝區 (Log Buffer Queue)，這邊感覺要刪掉
  useEffect(() => {
    const releaseInterval = setInterval(() => {
      if (logBufferRef.current.length > 0) {
        // 從緩衝區取出最舊的一筆
        const nextLog = logBufferRef.current.shift();
        console.log('[pollTest] 緩衝區釋放一筆到畫面', { range: nextLog?.range, id: nextLog?.id, bufferRemain: logBufferRef.current.length });
        setLogs(prev => [nextLog, ...prev].slice(0, 100));
      }
    }, 800); // 每 0.8 秒釋放一筆，讓視覺更平滑
    return () => clearInterval(releaseInterval);
  }, []);









  // 每次打開 modal 時重置「自動分析」標記，如果不重置，使用者關閉後再打開彈窗，系統會以為「上次已經分析過了」而不自動執行 AI 偵測。
  useEffect(() => {
    if (isOpen) autoAnalysisTriggeredRef.current = false;
  }, [isOpen]);

  // 移除 HLS manifest 檢查，直接顯示畫面
  // 因為 MediaMTX 的 HLS 播放器頁面在 /live/，不需要檢查 manifest

  useEffect(() => {
    if (isOpen) {

      // 利用 window.location 自動判斷當前是在 localhost:3000、80 還是 443 環境，增加部署的靈活性。
      const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');

      // 指向 /hls/live/。在後端架構中，這通常是 Nginx 反向代理後的路徑，實際上連向 MediaMTX 提供的 HLS 切片服務。
      const hlsPath = `http://${window.location.hostname}:${port}/hls/live/`;

      // 設置 HLS 狀態為可播放，並顯示串流運行中。
      setHlsStatus({ canPlay: true, message: "串流運行中", streamEnded: false });

      // 設置 HLS URL，讓前端可以播放。
      setHlsUrl(hlsPath);
      console.log("🚀 [HLS] Modal opened, using stream-simulator feed at", hlsPath);
    }
    
    // 只有在視窗開啟且有 apiKey 時才啟動輪詢
    if (!isOpen || !apiKey) {
      isFirstPollRef.current = true;
      pollCountRef.current = 0;
      initialBatchKeysRef.current.clear();
      analyzedSegmentsRef.current.clear();
      lastSegmentIndexRef.current = -1;
      segmentAnalysisInFlightRef.current = false;
      logBufferRef.current = [];
      return;
    }

    // 輪詢函式 (Polling Function)，負責定期向後端請求資料並更新 UI，包含系統硬體監測、串流狀態與 AI 分析日誌。處理每隔一段時間執行的重複任務。
    const pollTask = async () => {
      try {
        pollCountRef.current += 1;
        const pollCount = pollCountRef.current;
        console.log('[pollTest] pollTask 開始一輪', { pollCount, isFirstPoll: isFirstPollRef.current, seenCount: seenResultsRef.current.size });

        // 0. 獲取系統狀態（CPU、GPU、RAM 等）
        try {

          // 透過 API 服務獲取伺服器的硬體資訊（CPU、GPU、記憶體等）。
          const sysStatus = await apiService.getSystemStatus(apiKey);
          if (sysStatus) {

            // 初始化 GPU 狀態字串為空。
            let gpuStatusStr = null;

            // 檢查後端是否有回傳 GPU 設備資訊且長度大於 0。
            if (sysStatus.gpu?.devices && sysStatus.gpu.devices.length > 0) {

              // 取得第一個 GPU 設備。
              const gpu = sysStatus.gpu.devices[0];

              // 計算 GPU 記憶體使用率：若後端有給百分比就直接用，否則用「已用量 / 總量」自行計算。
              const memPercent = gpu.mem_util_percent !== null && gpu.mem_util_percent !== undefined
                ? gpu.mem_util_percent
                : ((gpu.mem_used_mb / gpu.mem_total_mb) * 100);

              // 格式化 GPU 顯示字串，例如：NVIDIA RTX 4090: Mem 12.50%。
              gpuStatusStr = `${gpu.name}: Mem ${memPercent.toFixed(2)}%`;
            }
            
            // 檢查 AI 模型（YOLO 物件偵測與 ReID 重識別）是否已載入 GPU 記憶體，並轉為可讀文字。
            const modelsStr = `YOLO: ${sysStatus.models?.yolo_world ? '✅ Loaded' : '❌ Not Loaded'} | ReID: ${sysStatus.models?.reid_model ? '✅ Loaded' : '❌ Not Loaded'}`;
            
            // 更新 React 的 State，將 CPU 使用率、RAM、GPU 字串、磁碟剩餘空間等同步到畫面上。
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

        // 1. 獲取目前所有正在執行的 RTSP 串流清單及其狀態。
        const status = await apiService.getRTSPStatus(apiKey);

        // 將活躍的串流列表更新到 UI。
        setActiveStreams(status || {});

        // 1.5. 更新 HLS 播放器狀態。這裡強制設為 canPlay: true，是因為系統使用了模擬器預推流，不需要等待後端確認。
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
        
        // 自動偵測目前的網域名稱與連接埠，組合出 Nginx 轉發的 HLS 影片串流位址。（使用 nginx 代理）
        const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
        setHlsUrl(`http://${window.location.hostname}:${port}/hls/live/`);

        // 2. 整理出一份需要追蹤分析結果的 ID 清單（包含本地測試影片、當前選中影片及所有串流），並去除重複項。
        const idsToTrack = [LOCAL_VIDEO_ID, videoId, ...Object.keys(status || {})].filter(id => id);
        const uniqueIds = [...new Set(idsToTrack)];
        console.log('[pollTest] 本輪追蹤的 video ids', uniqueIds);

        // 併發請求：同時向後端查詢所有 ID 的分析報告。若其中一個請求失敗則回傳 null。
        const infoResults = await Promise.all(
          uniqueIds.map(id => apiService.getVideoInfo(id, apiKey).catch(() => null))
        );

        infoResults.forEach((info, idx) => {
          const id = uniqueIds[idx];
          const n = info?.analysis_data?.results?.length ?? 0;
          if (n > 0) console.log('[pollTest] getVideoInfo 回傳', { id, resultCount: n });
        });

        let newLogItems = [];
        let hadResultsInThisPoll = false;
        
        // 遍歷所有分析結果，如果該 ID 尚未產生分析資料，則跳過。
        infoResults.forEach((info, index) => {
          if (!info || !info.analysis_data?.results) return;
          const id = uniqueIds[index];
          hadResultsInThisPoll = true;
          
          // 根據時間區間（例如 00:00-00:10）對分析結果進行排序，確保日誌順序正確。
          const sortedResults = [...info.analysis_data.results].sort((a, b) =>
            (a.time_range || '').localeCompare(b.time_range || '')
          );

          // 建立一個唯一 Key，組合「影片 ID」與「時間段」，避免重複顯示相同的分析結果。
          sortedResults.forEach(res => {
            const logKey = `${id}-${res.time_range}`;
            if (!seenResultsRef.current.has(logKey)) {
              seenResultsRef.current.add(logKey);
              
              // 本地影片只依 onTimeUpdate + analyze_single_segment 逐段顯示，不從 getVideoInfo 推入，避免一次出現多筆
              if (id === LOCAL_VIDEO_ID) return;
              
              // 僅在「前 2 次輪詢就拿到結果」時視為舊資料不顯示（一打開就有 30 筆 = 上次分析存的）。
              // 並把這批 key 記住，之後即使競態導致 isFirstPoll 被改掉，這批也永不顯示。
              const skipAsOldData = isFirstPollRef.current && pollCountRef.current <= 2;
              if (skipAsOldData) {
                initialBatchKeysRef.current.add(logKey);
                return;
              }
              // 一打開就存在的那批（第一次輪詢時已加入 initialBatchKeysRef）永不顯示
              if (initialBatchKeysRef.current.has(logKey)) return;

              // 從 AI 回傳的 JSON 中提取 events 物件（內含火災、倒地等布林值）與模型給出的 reason（理由）。
              const eventObj = res.parsed?.frame_analysis?.events || {};
              const reason = eventObj.reason || "";
              
              // 找出所有值為 true 的事件（即被偵測到的異常），並排除 reason 欄位。
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

              // 如果有偵測到事件，組合成字串；若無，顯示「無異常」。
              let eventStr = detectedEvents.length > 0 
                ? `偵測到：${detectedEvents.join(", ")}` 
                : "無異常";
              
              // 如果有異常且模型提供了說明文字，則附在括號內。
              if (detectedEvents.length > 0 && reason) {
                eventStr += ` (${reason})`;
              }
              
              // 將處理好的日誌物件推入緩衝區。isCritical 為真時，UI 通常會顯示為紅色高亮。
              logBufferRef.current.push({
                time: new Date().toLocaleTimeString(),
                id: id === LOCAL_VIDEO_ID ? '本地影片' : id,
                range: res.time_range,
                eventStr: eventStr,
                isCritical: detectedEvents.length > 0
              });
              console.log('[pollTest] 新結果推入緩衝區', { id, range: res.time_range, bufferLength: logBufferRef.current.length });
            }
          });
        });
        
        // 只有「本輪有拿到結果」時才標記第一次輪詢結束，避免競態。
        if (hadResultsInThisPoll) isFirstPollRef.current = false;
        console.log('[pollTest] pollTask 本輪結束', { pollCount: pollCountRef.current, hadResultsInThisPoll, isFirstPollNow: isFirstPollRef.current, seenCount: seenResultsRef.current.size });
      } catch (e) {
        console.error("Polling failed", e);
      }
    };

    // 設定一個定時器，每隔 2000 毫秒（2 秒） 就自動呼叫一次上面定義的 pollTask 函式。
    const interval = setInterval(pollTask, 2000);

    // setInterval 會在 2 秒後才第一次觸發。為了讓使用者一打開頁面就能看到數據（而不是空等 2 秒），這裡手動先執行一次。
    pollTask();

    return () => {

      // 停止計時器：告訴瀏覽器停止那個每 2 秒一次的任務。如果不做這步，即使使用者關閉了監控視窗，背景程式仍會持續瘋狂請求 API，浪費伺服器資源。
      clearInterval(interval);

      // 清空 Set，避免重複顯示舊資訊。
      seenResultsRef.current.clear();
      initialBatchKeysRef.current.clear();
      analyzedSegmentsRef.current.clear();
      lastSegmentIndexRef.current = -1;
      segmentAnalysisInFlightRef.current = false;
      // 清空日誌緩衝區。
      setLogs([]);
    };
  }, [isOpen, apiKey, videoId]); // 只在視窗、密鑰或主要 ID 變更時啟動一次邏輯

  // 用於觸發本地影片分析的完整流程。它負責從前端發送分析參數到後端，並將回傳的 AI 摘要即時轉換為使用者可讀的日誌訊息。
  const handleStart = async () => {

    // 將 UI 狀態設為「分析中」，通常會觸發按鈕禁用（Disabled）或顯示 Loading 動畫。
    setIsLocalAnalyzing(true);
    
    // 清空上一次的分析結果，確保畫面資料是最新的。
    setLocalAnalysisResult(null);

    // 在日誌列表的最上方插入一條「系統訊息」，告知使用者分析程序已經啟動。
    setLogs(prev => [{
      time: new Date().toLocaleTimeString(),
      id: '系統',
      eventStr: `開始分析本地影片: ${LOCAL_VIDEO_ID}`,
      isCritical: false
    }, ...prev]);

    try {
      const formData = new FormData();
      formData.append('model_type', 'qwen');
      formData.append('segment_duration', 10);
      formData.append('overlap', 0);
      formData.append('qwen_model', 'qwen2.5vl:latest');
      formData.append('frames_per_segment', 8);
      formData.append('target_short', 720);
      formData.append('video_id', LOCAL_VIDEO_ID);

      // 核心 API 呼叫：等待後端執行完複雜的 AI 切片、物件偵測與 VLM 摘要。
      const data = await apiService.runAnalysis(formData, apiKey);

      // 將後端回傳的完整 JSON 資料（包含各段落的詳細結果）存入 State。
      setLocalAnalysisResult(data);

      // 提取分析統計：總共處理了幾段，其中幾段成功。
      const total = data?.total_segments ?? 0;
      const success = data?.success_segments ?? 0;
      const t = new Date().toLocaleTimeString();
      const newEntries = [{
        time: t,
        id: '系統',
        eventStr: `✓ 分析完成：${success}/${total} 段成功`,
        isCritical: true
      }];

      // // 如果有分析結果，取前 30 段進行日誌展示（避免日誌瞬間爆炸太長）。
      // if (data?.results?.length) {
      //   data.results.slice(0, 30).forEach((r, i) => {

      //     // 關鍵提取：優先尋找 AI 生成的「獨立摘要（Independent Summary）」。
      //     const seg = r?.parsed?.summary_independent || r?.segment || `片段 ${i + 1}`;

      //     // 確保摘要內容轉為純字串格式。
      //     const summary = (typeof seg === 'string' ? seg : seg?.message) || '';
      //     if (summary) {

      //       // 將摘要的前 100 個字加入日誌緩衝區。這讓使用者能直接在日誌面板看到影片發生了什麼事。
      //       newEntries.push({
      //         time: t,
      //         id: r?.segment ?? `seg_${i + 1}`,
      //         eventStr: summary.substring(0, 100) + (summary.length > 100 ? '…' : ''),
      //         isCritical: false
      //       });
      //     }
      //   });
      // }

      // 只顯示「分析完成」系統訊息；片段內容改由 pollTask 從 getVideoInfo 取得後顯示
      setLogs(prev => [...newEntries, ...prev].slice(0, 100));
    } catch (e) {
      setLogs(prev => [{
        time: new Date().toLocaleTimeString(),
        id: '系統',
        eventStr: `✗ 分析失敗: ${e?.message || e}`,
        isCritical: true
      }, ...prev]);
    } 
    
    // 無論成功或失敗，最後都解開按鈕的鎖定狀態。
    finally {
      setIsLocalAnalyzing(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '1100px', width: '95%' }}>
        <div className="modal-header">
          <h3 className="modal-title">📁 本地影片 AI 分析</h3>
          <button className="modal-close-btn" onClick={onClose}>×</button>
        </div>

        <div className="modal-body" style={{ background: '#1e1e1e', color: 'white', padding: '20px' }}>
          <div style={{ display: 'flex', gap: '25px', height: '650px' }}>
            
            {/* 左側：本地影片 + 啟動 AI 分析 */}
            <div style={{ flex: '1.4', display: 'flex', flexDirection: 'column' }}>
              <div style={{ background: 'black', flex: 1, minHeight: '360px', marginBottom: '16px', borderRadius: '8px', overflow: 'hidden', border: '1px solid #333', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {localVideoError ? (
                  <div style={{ padding: '24px', textAlign: 'center', color: '#ff9800' }}>
                    <div style={{ fontSize: '14px', marginBottom: '8px' }}>⚠️ {localVideoError}</div>
                    <div style={{ fontSize: '12px', color: '#888' }}>請將 <strong>{LOCAL_VIDEO_FILENAME}</strong> 放到專案的 <code>frontend/public/video/</code> 資料夾內後重新整理。</div>
                  </div>
                ) : (
                  <>
                    <video
                      ref={videoRef}
                      key="local-video"
                      src={localVideoUrl}
                      controls
                      loop
                      autoPlay
                      muted
                      playsInline
                      preload="auto"
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      title="本地影片"
                      onLoadedData={() => { setLocalVideoReady(true); setLocalVideoError(null); }}
                      onCanPlay={() => {

                        // 隱藏正在轉圈圈的載入動畫（Loading Spinner）。解除「開始分析」按鈕的禁用狀態（Disabled）。顯示影片播放器介面。
                        setLocalVideoReady(true);
                        setLocalVideoError(null);

                        // 改為不自動觸發整支影片分析，避免一次出現 30 筆；只依 onTimeUpdate 隨播放逐段分析。「重新分析」按鈕會清空並從頭播。
                        // if (!autoAnalysisTriggeredRef.current && apiKey) {
                        //   autoAnalysisTriggeredRef.current = true;
                        //   handleStart();
                        // }
                      }}
                      onTimeUpdate={(e) => {
                        const video = e.target;
                        if (!video || !apiKey) return;
                        if (segmentAnalysisInFlightRef.current) return; // 同一時間只發一個請求，避免多個長連線導致 ERR_CONNECTION_TIMED_OUT
                        const t = video.currentTime;
                        const segmentIndex = Math.floor(t / 10);
                        if (analyzedSegmentsRef.current.has(segmentIndex)) {
                          lastSegmentIndexRef.current = segmentIndex;
                          return;
                        }
                        segmentAnalysisInFlightRef.current = true;
                        analyzedSegmentsRef.current.add(segmentIndex);
                        lastSegmentIndexRef.current = segmentIndex;
                        const startTime = segmentIndex * 10;
                        apiService.runInstantSegmentAnalysis(LOCAL_VIDEO_ID, startTime, 10, apiKey)
                          .then((res) => {
                            const eventObj = res?.parsed?.frame_analysis?.events || {};
                            const reason = eventObj.reason || '';
                            const detectedEvents = Object.entries(eventObj)
                              .filter(([key, value]) => key !== 'reason' && value === true)
                              .map(([key]) => {
                                const names = { fire: '火災', water_flood: '水災', person_fallen: '倒地', double_parking: '併排', smoking: '吸菸', crowd: '聚眾', security_door: '門禁異常', abnormal_attire: '遮臉' };
                                return names[key] || key;
                              });
                            let eventStr = detectedEvents.length > 0 ? `偵測到：${detectedEvents.join(', ')}` : '無異常';
                            if (detectedEvents.length > 0 && reason) eventStr += ` (${reason})`;
                            const summary = res?.parsed?.summary_independent || res?.segment || '';
                            const summaryStr = (typeof summary === 'string' ? summary : summary?.message) || '';
                            if (summaryStr) eventStr = summaryStr.substring(0, 120) + (summaryStr.length > 120 ? '…' : '');
                            const newResult = {
                              time: new Date().toLocaleTimeString(),
                              id: '本地影片',
                              range: res?.time_range || `${startTime}-${startTime + 10}s`,
                              eventStr,
                              isCritical: detectedEvents.length > 0
                            };
                            setLogs(prev => [newResult, ...prev].slice(0, 100));
                          })
                          .catch(() => { analyzedSegmentsRef.current.delete(segmentIndex); })
                          .finally(() => { segmentAnalysisInFlightRef.current = false; });
                      }}
                      
                      onError={(e) => {
                        const msg = e?.target?.error?.message || '無法載入影片';
                        setLocalVideoError(msg);
                        setLocalVideoReady(false);
                      }}
                    />
                    <div style={{ position: 'absolute', top: '10px', right: '10px', background: 'rgba(0,0,0,0.7)', padding: '4px 10px', borderRadius: '4px', fontSize: '11px', color: '#2196F3', fontWeight: 'bold', zIndex: 10 }}>
                      📁 {LOCAL_VIDEO_FILENAME}
                    </div>
                  </>
                )}
              </div>

              <div style={{ background: '#2d2d2d', padding: '15px', borderRadius: '8px', marginBottom: '15px' }}>
                <div style={{ fontSize: '12px', color: '#888', marginBottom: '8px' }}>
                  播放時每 10 秒自動分析該段，結果即時顯示於右側；按「重新分析」會清空日誌、影片從頭播並重新逐段判斷。
                </div>
                <button
                  onClick={() => {
                    setLogs([]);
                    setLocalAnalysisResult(null);
                    logBufferRef.current = [];
                    analyzedSegmentsRef.current.clear();
                    lastSegmentIndexRef.current = -1;
                    segmentAnalysisInFlightRef.current = false;
                    if (videoRef.current) {
                      videoRef.current.currentTime = 0;
                    }
                  }}
                  className="btn btn-primary"
                  style={{ width: '100%', height: '44px', fontSize: '15px' }}
                >
                  重新分析
                </button>
                {localAnalysisResult && (
                  <div style={{ marginTop: '10px', padding: '10px', background: 'rgba(76, 175, 80, 0.15)', borderRadius: '6px', border: '1px solid #4CAF50', fontSize: '12px', color: '#a5d6a7' }}>
                    <strong>✓ 分析結果</strong>
                    <div style={{ marginTop: '6px' }}>
                      {localAnalysisResult.total_segments ?? 0} 段 / 成功 {localAnalysisResult.success_segments ?? 0} 段
                      {localAnalysisResult.total_time_sec != null && ` · 耗時 ${localAnalysisResult.total_time_sec} 秒`}
                    </div>
                  </div>
                )}
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

              {/* 運行中串流：顯示進度（讀到哪一段、已處理幾段）、錯誤；本地影片分析時也顯示狀態 */}
              <div style={{ background: '#252525', padding: '15px', borderRadius: '8px', flex: '0 0 auto' }}>
                <h4 style={{ fontSize: '14px', marginBottom: '10px', color: '#4CAF50' }}>● 運行中串流</h4>
                <div style={{ maxHeight: '220px', overflowY: 'auto' }}>
                  {isLocalAnalyzing && (
                    <div style={{
                      marginBottom: '8px', padding: '8px', background: '#333', borderRadius: '4px', fontSize: '12px',
                      borderLeft: '3px solid #ff9800'
                    }}>
                      <div style={{ fontWeight: 'bold', color: '#ff9800' }}>本地影片分析中</div>
                      <div style={{ color: '#aaa', fontSize: '11px', marginTop: '4px' }}>{LOCAL_VIDEO_ID}</div>
                      <div style={{ color: '#888', fontSize: '11px', marginTop: '2px' }}>完成後結果會顯示於下方日誌</div>
                    </div>
                  )}
                  {Object.entries(activeStreams).length === 0 && !isLocalAnalyzing ? (
                    <div style={{ color: '#666', fontSize: '12px', textAlign: 'center', padding: '8px' }}>
                      目前沒有運行中的串流（僅本地影片分析時為正常）
                    </div>
                  ) : null}
                  {Object.entries(activeStreams).length > 0 ? (
                    Object.entries(activeStreams).map(([id, info]) => (
                      <div key={id} style={{
                        marginBottom: '8px', padding: '8px', background: '#333', borderRadius: '4px', fontSize: '12px',
                        borderLeft: `3px solid ${info.status === 'running' ? '#4CAF50' : info.status === 'ended' ? '#888' : '#ff4444'}`
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                          <span style={{ fontWeight: 'bold' }}>{id}</span>
                          <span style={{ color: '#888', fontSize: '11px' }}>{info.status} | 運行 {info.uptime}s</span>
                        </div>
                        {(info.segments_processed != null || info.current_segment || info.last_segment_name) && (
                          <div style={{ color: '#aaa', fontSize: '11px', marginTop: '4px' }}>
                            已處理 <strong style={{ color: '#fff' }}>{info.segments_processed ?? 0}</strong> 段
                            {info.current_segment && (
                              <span> · 正在處理: <span style={{ color: '#ff9800' }}>{info.current_segment}</span></span>
                            )}
                            {info.last_segment_name && !info.current_segment && (
                              <span> · 上一段: {info.last_segment_name}</span>
                            )}
                          </div>
                        )}
                        {info.error_message && (
                          <div style={{ color: '#ff6666', fontSize: '11px', marginTop: '4px' }}>⚠️ {info.error_message}</div>
                        )}
                        {info.recent_errors && info.recent_errors.length > 0 && (
                          <div style={{ marginTop: '6px' }}>
                            <div style={{ color: '#ff9800', fontSize: '11px', marginBottom: '2px' }}>最近分析錯誤:</div>
                            {info.recent_errors.slice(-3).map((err, i) => (
                              <div key={i} style={{ color: '#ff6666', fontSize: '10px', marginLeft: '6px' }}>
                                {err.segment}: {err.error}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))
                  ) : null}
                </div>
              </div>
            </div>

            {/* 右側：分析日誌 */}
            <div style={{ flex: '1', display: 'flex', flexDirection: 'column', background: '#000', borderRadius: '8px', border: '1px solid #333' }}>
              <div style={{ padding: '12px 15px', borderBottom: '1px solid #333', background: '#111', borderTopLeftRadius: '8px', borderTopRightRadius: '8px' }}>
                <h4 style={{ margin: 0, fontSize: '15px', color: '#ff9800' }}>分析日誌</h4>
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