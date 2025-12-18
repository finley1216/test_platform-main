import { useState, useCallback, useEffect } from "react";
import apiService from "../services/api";

export const useRAG = (apiKey, authenticated) => {
  const [isSearching, setIsSearching] = useState(false);
  const [ragData, setRagData] = useState(null);
  const [ragError, setRagError] = useState("");
  const [ragStats, setRagStats] = useState(0);

  const updateStats = useCallback(async () => {
    if (!authenticated || !apiKey) {
      setRagStats(0);
      return;
    }
    const count = await apiService.getRagStats(apiKey);
    setRagStats(count);
  }, [apiKey, authenticated]);

  const search = useCallback(async (query, topK, threshold) => {
    setIsSearching(true);
    setRagData(null);
    setRagError("");

    try {
      const data = await apiService.searchRAG(query, topK, threshold, apiKey);
      setRagData(data);
      return { success: true, data, count: data.hits?.length || 0 };
    } catch (error) {
      const errorMessage = error.message || "Unknown error";
      setRagError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setIsSearching(false);
    }
  }, [apiKey]);

  const answer = useCallback(async (query, topK, threshold) => {
    setIsSearching(true);
    setRagData(null);
    setRagError("");

    try {
      const data = await apiService.answerRAG(query, topK, threshold, apiKey);
      setRagData(data);
      return { success: true, data };
    } catch (error) {
      const errorMessage = error.message || "Unknown error";
      setRagError(errorMessage);
      return { success: false, error: errorMessage };
    } finally {
      setIsSearching(false);
    }
  }, [apiKey]);

  useEffect(() => {
    if (authenticated && apiKey) {
      updateStats();
    }
  }, [authenticated, apiKey, updateStats]);

  return {
    isSearching,
    ragData,
    ragError,
    ragStats,
    search,
    answer,
    updateStats,
  };
};

