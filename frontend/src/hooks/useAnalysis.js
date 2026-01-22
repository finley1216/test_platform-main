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
        "GPU Status": status.gpu?.devices?.map(d => {
          // 處理 null/undefined 值（後端返回 None 時會變成 null）
          const hasUtil = d.gpu_util_percent != null && typeof d.gpu_util_percent === 'number';
          const hasMem = d.mem_util_percent != null && typeof d.mem_util_percent === 'number';
          
          let statusStr = d.name;
          if (hasUtil) {
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

    const startTime = Date.now();
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
        
        if (percentCompleted !== lastProgressRef.current) {
          lastProgressRef.current = percentCompleted;
          lastProgressTimeRef.current = Date.now();
        }
      });
      
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log("%c[完成] 收到後端響應, 總耗時:", "color: #8b5cf6; font-weight: bold", elapsed, "秒");
      
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

