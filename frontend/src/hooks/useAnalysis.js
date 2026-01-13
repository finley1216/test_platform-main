import { useState, useCallback } from "react";
import apiService from "../services/api";

export const useAnalysis = (apiKey) => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [analysisError, setAnalysisError] = useState("");

  const runAnalysis = useCallback(async (formData, onSuccess) => {
    setIsAnalyzing(true);
    setAnalysisData(null);
    setAnalysisError("");

    const startTime = Date.now();
    console.log("%c" + "=".repeat(80), "color: #3b82f6; font-weight: bold; font-size: 14px");
    console.log("%c[上傳流程開始] " + new Date().toLocaleString(), "color: #3b82f6; font-weight: bold; font-size: 14px");
    console.log("%c" + "=".repeat(80), "color: #3b82f6; font-weight: bold; font-size: 14px");
    
    // 記錄上傳參數
    console.log("%c[步驟 1/5] 準備上傳參數", "color: #8b5cf6; font-weight: bold");
    const modelType = formData.get("model_type");
    const segmentDuration = formData.get("segment_duration");
    const overlap = formData.get("overlap");
    const file = formData.get("file");
    const videoUrl = formData.get("video_url");
    const videoId = formData.get("video_id");
    
    console.log("  模型類型:", modelType);
    console.log("  片段長度:", segmentDuration, "秒");
    console.log("  重疊時間:", overlap, "秒");
    
    if (file) {
      console.log("  上傳來源: 本地文件");
      console.log("  文件名稱:", file.name);
      console.log("  文件大小:", (file.size / 1024 / 1024).toFixed(2), "MB");
      console.log("  文件類型:", file.type);
    } else if (videoId) {
      console.log("  上傳來源: 已存在的影片");
      console.log("  影片 ID:", videoId);
    } else if (videoUrl) {
      console.log("  上傳來源: URL");
      console.log("  影片 URL:", videoUrl);
    }
    
    console.log("%c[步驟 2/5] 發送請求到後端", "color: #8b5cf6; font-weight: bold");
    console.log("  API 端點: /api/v1/segment_pipeline_multipart");
    console.log("  請求時間:", new Date().toLocaleString());

    try {
      const data = await apiService.runAnalysis(formData, apiKey);
      
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log("%c[步驟 5/5] 收到後端響應", "color: #8b5cf6; font-weight: bold");
      console.log("  響應時間:", new Date().toLocaleString());
      console.log("  總耗時:", elapsed, "秒");
      
      setAnalysisData(data);
      
      // 輸出分析結果到 console
      console.log("%c" + "=".repeat(80), "color: #10b981; font-weight: bold; font-size: 14px");
      console.log("%c✓ 分析完成！", "color: #10b981; font-weight: bold; font-size: 16px");
      console.log("%c" + "=".repeat(80), "color: #10b981; font-weight: bold; font-size: 14px");
      console.log("%c[處理結果統計]", "color: #10b981; font-weight: bold");
      console.log("  模型類型:", data.model_type || "N/A");
      console.log("  總片段數:", data.total_segments || 0);
      console.log("  成功片段數:", data.success_segments || 0);
      console.log("  失敗片段數:", (data.total_segments || 0) - (data.success_segments || 0));
      console.log("  後端處理時間:", (data.process_time_sec || 0).toFixed(2), "秒");
      console.log("  總時間（含上傳）:", (data.total_time_sec || 0).toFixed(2), "秒");
      console.log("  前端總耗時:", elapsed, "秒");
      
      if (data.results && data.results.length > 0) {
        console.log("%c[片段分析結果]", "color: #3b82f6; font-weight: bold; font-size: 12px");
        data.results.forEach((result, idx) => {
          const status = result.success ? "✓" : "✗";
          const statusColor = result.success ? "#10b981" : "#ef4444";
          console.log(
            "%c  " + status + " 片段 " + (idx + 1) + ": " + (result.segment || "N/A"),
            `color: ${statusColor}; font-size: 12px`
          );
          if (result.error) {
            console.log("%c    錯誤: " + result.error, "color: #ef4444; font-size: 11px");
          }
          if (result.parsed && result.parsed.frame_analysis) {
            const events = result.parsed.frame_analysis.events || {};
            const activeEvents = Object.keys(events).filter(k => k !== "reason" && events[k] === true);
            if (activeEvents.length > 0) {
              console.log("%c    偵測到事件: " + activeEvents.join(", "), "color: #f59e0b; font-size: 11px");
            }
            if (result.parsed.summary_independent) {
              const summary = result.parsed.summary_independent;
              const preview = summary.length > 50 ? summary.substring(0, 50) + "..." : summary;
              console.log("%c    摘要: " + preview, "color: #6b7280; font-size: 11px");
            }
          }
          if (result.raw_detection && result.raw_detection.yolo_skipped) {
            console.log("%c    [YOLO 已停用]", "color: #9ca3af; font-size: 11px");
          }
        });
      }
      
      if (data.rag_auto_indexed) {
        console.log("%c[RAG 索引]", "color: #a78bfa; font-weight: bold; font-size: 12px");
        console.log("%c  " + (data.rag_auto_indexed.message || "N/A"), "color: #6b7280; font-size: 11px");
      }
      
      console.log("%c" + "=".repeat(60), "color: #3b82f6; font-weight: bold; font-size: 14px");
      
      onSuccess?.(data);
      
      if (data.results && data.results.length > 0) {
        console.log("%c[片段詳情]", "color: #6b7280; font-weight: bold");
        data.results.forEach((result, idx) => {
          const status = result.success ? "✓" : "✗";
          const color = result.success ? "#10b981" : "#ef4444";
          console.log(`%c  ${status} 片段 ${idx}: ${result.segment || 'N/A'} (${result.time_range || 'N/A'})`, `color: ${color}`);
        });
      }
      
      console.log("%c" + "=".repeat(80), "color: #10b981; font-weight: bold; font-size: 14px");
      
      return { success: true, data };
    } catch (error) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      const errorMessage = error.message || "Unknown error occurred";
      setAnalysisError(errorMessage);
      
      console.log("%c" + "=".repeat(80), "color: #ef4444; font-weight: bold; font-size: 14px");
      console.log("%c✗ 分析失敗！", "color: #ef4444; font-weight: bold; font-size: 16px");
      console.log("%c" + "=".repeat(80), "color: #ef4444; font-weight: bold; font-size: 14px");
      console.log("%c[錯誤信息]", "color: #ef4444; font-weight: bold");
      console.log("  錯誤訊息:", errorMessage);
      console.log("  發生時間:", new Date().toLocaleString());
      console.log("  已耗時:", elapsed, "秒");
      
      // 如果有錯誤詳情，也記錄下來
      if (error.detail) {
        console.log("  詳細錯誤:", error.detail);
      }
      if (error.status) {
        console.log("  HTTP 狀態碼:", error.status);
      }
      
      console.log("%c" + "=".repeat(80), "color: #ef4444; font-weight: bold; font-size: 14px");
      
      return { success: false, error: errorMessage };
    } finally {
      setIsAnalyzing(false);
    }
  }, [apiKey]);

  return {
    isAnalyzing,
    analysisData,
    analysisError,
    runAnalysis,
  };
};

