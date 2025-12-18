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

    try {
      const data = await apiService.runAnalysis(formData, apiKey);
      setAnalysisData(data);
      onSuccess?.(data);
      return { success: true, data };
    } catch (error) {
      const errorMessage = error.message || "Unknown error occurred";
      setAnalysisError(errorMessage);
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

