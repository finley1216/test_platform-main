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

    console.log("%c" + "=".repeat(60), "color: #3b82f6; font-weight: bold; font-size: 14px");
    console.log("%c[開始分析]", "color: #3b82f6; font-weight: bold; font-size: 14px");
    console.log("%c正在上傳影片並開始分析...", "color: #6b7280");

    try {
      const data = await apiService.runAnalysis(formData, apiKey);
      setAnalysisData(data);
      
      // 輸出分析結果到 console
      console.log("%c[分析完成]", "color: #10b981; font-weight: bold; font-size: 14px");
      console.log("%c  模型類型: " + (data.model_type || "N/A"), "color: #6b7280; font-size: 12px");
      console.log("%c  總片段數: " + (data.total_segments || 0), "color: #6b7280; font-size: 12px");
      console.log("%c  成功片段數: " + (data.success_segments || 0), "color: #10b981; font-size: 12px");
      console.log("%c  總處理時間: " + (data.total_time_sec || 0) + " 秒", "color: #6b7280; font-size: 12px");
      
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
      return { success: true, data };
    } catch (error) {
      const errorMessage = error.message || "Unknown error occurred";
      setAnalysisError(errorMessage);
      
      console.log("%c[分析失敗]", "color: #ef4444; font-weight: bold; font-size: 14px");
      console.log("%c  錯誤訊息: " + errorMessage, "color: #ef4444; font-size: 12px");
      console.log("%c" + "=".repeat(60), "color: #3b82f6; font-weight: bold; font-size: 14px");
      
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

