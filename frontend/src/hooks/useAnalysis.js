import { useState, useCallback, useRef, useEffect } from "react";
import apiService from "../services/api";

export const useAnalysis = (apiKey) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisError, setAnalysisError] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  
  const watchdogRef = useRef(null);
  const pollingRef = useRef(null);
  const lastProgressRef = useRef(0);
  const lastProgressTimeRef = useRef(Date.now());
  const loggedMilestonesRef = useRef(new Set());
  const clientSendTimeRef = useRef(null);

  // 清理定時器
  const clearTimers = () => {
    if (watchdogRef.current) clearInterval(watchdogRef.current);
    if (pollingRef.current) clearInterval(pollingRef.current);
  };

  useEffect(() => {
    return () => clearTimers();
  }, []);

  const fetchSystemStatus = async () => {
    const status = await apiService.getSystemStatus(apiKey);
    if (status) {
      console.group('🔍 Server Diagnostics');
      console.log(`Time: ${new Date().toLocaleTimeString()} | Active Requests: ${status.active_requests}`);
      console.table({
        "CPU (%)": status.cpu.percent,
        "RAM (%)": status.memory.percent,
        "Backend CUDA": status.backend_cuda
          ? (status.backend_cuda.available ? `✅ ${status.backend_cuda.device_name || "GPU"}` : "❌ 後端看不到 GPU（YOLO/ReID 用 CPU，較慢）")
          : "N/A",
        "GPU Status": status.gpu?.devices?.map(d => {
          const hasUtil = d.gpu_util_percent != null && typeof d.gpu_util_percent === 'number';
          const hasMem = d.mem_util_percent != null && typeof d.mem_util_percent === 'number';
          let statusStr = d.name;
          if (hasUtil && hasMem) {
            statusStr += `: Util ${d.gpu_util_percent}% | Mem ${d.mem_util_percent}%`;
          } else if (hasUtil) {
            statusStr += `: ${d.gpu_util_percent}%`;
          } else if (hasMem) {
            statusStr += `: Mem ${d.mem_util_percent}%`;
          } else {
            statusStr += `: N/A`;
          }
          return statusStr;
        }).join(", ") || "N/A",
        "Models Loaded": `YOLO: ${status.models.yolo_world ? '✅ Loaded' : '⏳ Will load on demand'} | ReID: ${status.models.reid_model ? '✅ Loaded' : '⏳ Will load on demand'}`,
        "Disk Free": `${status.disk.free_gb} GB`
      });
      console.groupEnd();
    }
  };

  const runAnalysis = useCallback(async (formData, onSuccess) => {
    setIsAnalyzing(true);
    setAnalysisData(null);
    setAnalysisError("");
    setUploadProgress(0);
    lastProgressRef.current = 0;
    lastProgressTimeRef.current = Date.now();
    loggedMilestonesRef.current = new Set();
    clientSendTimeRef.current = Date.now();

    const startTime = Date.now();
    const tsMs = () => new Date().toISOString();
    console.log("%c" + "=".repeat(80), "color: #3b82f6; font-weight: bold; font-size: 14px");
    console.log("%c[分析流程開始] " + new Date().toLocaleString(), "color: #3b82f6; font-weight: bold; font-size: 14px");
    
    // 啟動系統狀態輪詢 (Point 7, 10)
    fetchSystemStatus();
    pollingRef.current = setInterval(fetchSystemStatus, 5000);

    // 實作上傳看門狗 (Point 6)
    watchdogRef.current = setInterval(() => {
      const now = Date.now();
      const timeSinceLastProgress = now - lastProgressTimeRef.current;
      
      if (lastProgressRef.current > 0 && lastProgressRef.current < 99 && timeSinceLastProgress > 10000) {
        console.warn("%c🔍 Server Diagnostics", "font-weight: bold; color: #f59e0b");
        console.warn("⚠️ 上傳疑似被 Proxy 卡住 (Upload Stalled) - 10秒無進度更新");
      }
    }, 2000);

    try {
      const data = await apiService.runAnalysis(formData, apiKey, (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(percentCompleted);

        if (lastProgressRef.current === 0 && percentCompleted > 0) {
          console.log(
            `%c[LATENCY] upload_first_progress=${tsMs()} | percent=${percentCompleted}% | loaded=${progressEvent.loaded} | total=${progressEvent.total}`,
            "color: #f59e0b; font-weight: bold"
          );
        }

        for (let milestone = 10; milestone <= 100; milestone += 10) {
          if (
            percentCompleted >= milestone &&
            !loggedMilestonesRef.current.has(milestone)
          ) {
            loggedMilestonesRef.current.add(milestone);
            const sinceSendMs = Date.now() - (clientSendTimeRef.current || startTime);
            console.log(
              `%c[LATENCY] upload_progress_${milestone}=${tsMs()} | elapsed_since_send_ms=${sinceSendMs}`,
              "color: #f59e0b; font-weight: bold"
            );
          }
        }
        
        if (percentCompleted !== lastProgressRef.current) {
          lastProgressRef.current = percentCompleted;
          lastProgressTimeRef.current = Date.now();
        }
      });
      
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log("%c[完成] 收到後端響應, 總耗時:", "color: #8b5cf6; font-weight: bold", elapsed, "秒");
      console.log(
        "%c[LATENCY] 前端時間軸摘要 — 請搭配後端 docker logs 中 [LATENCY] 行對照 req_id",
        "color: #3b82f6; font-weight: bold"
      );
      console.table({
        "T0 client_send": "見 [LATENCY] client_send_timestamp",
        "T1 upload_progress": "見 upload_progress_10 ~ _100",
        "T2 client_upload_complete": "見 [LATENCY] client_upload_complete",
        "T3 server_receive": "後端 [LATENCY] server_receive_timestamp",
        "T4 server_file_ready": "後端 [LATENCY] server_file_ready_timestamp",
        "T5 client_response": "見 [LATENCY] client_response_received",
      });
      
      setAnalysisData(data);
      onSuccess?.(data);
      return { success: true, data };
    } catch (error) {
      // 支援原生 Error 或 Axios 格式的錯誤提取 (Point 8)
      const errorMessage = error.message || "Unknown error";
      setAnalysisError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setIsAnalyzing(false);
      clearTimers();
    }
  }, [apiKey]);

  return {
    isAnalyzing,
    analysisData,
    analysisError,
    uploadProgress,
    runAnalysis,
  };
};

