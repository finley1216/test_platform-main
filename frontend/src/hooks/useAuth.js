import { useState, useEffect, useCallback } from "react";
import { STORAGE_KEYS } from "../utils/constants";
import apiService from "../services/api";

export const useAuth = () => {
  const [authenticated, setAuthenticated] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [authMessage, setAuthMessage] = useState({ text: "", type: "" });
  const [isAdmin, setIsAdmin] = useState(false);

  const showMessage = useCallback((text, type = "info") => {
    setAuthMessage({ text, type });
    setTimeout(() => setAuthMessage({ text: "", type: "" }), 5000);
  }, []);

  const verify = useCallback(async (key) => {
    if (!key || typeof key !== "string") {
      showMessage("請輸入 API Key", "error");
      return false;
    }
    const trimmedKey = key.trim();
    if (!trimmedKey) {
      showMessage("請輸入 API Key", "error");
      return false;
    }

    const result = await apiService.verifyAuth(trimmedKey);
    if (result && result.ok) {
      setAuthenticated(true);
      setApiKey(trimmedKey);
      setIsAdmin(result.is_admin || false);
      localStorage.setItem(STORAGE_KEYS.API_KEY, trimmedKey);
      showMessage("✓ 驗證成功", "success");
      return true;
    } else {
      setAuthenticated(false);
      setIsAdmin(false);
      showMessage("驗證失敗，無效的 API Key", "error");
      return false;
    }
  }, [showMessage]);

  const logout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEYS.API_KEY);
    setApiKey("");
    setAuthenticated(false);
    showMessage("已登出", "info");
  }, [showMessage]);

  useEffect(() => {
    const initialize = async () => {
      const savedKey = localStorage.getItem(STORAGE_KEYS.API_KEY);
      if (savedKey) {
        setApiKey(savedKey);
        const result = await apiService.verifyAuth(savedKey);
        if (result && result.ok) {
          setAuthenticated(true);
          setIsAdmin(result.is_admin || false);
        } else {
          setAuthenticated(false);
          setIsAdmin(false);
        }
      }
    };
    initialize();
  }, []);

  return {
    authenticated,
    apiKey,
    authMessage,
    isAdmin,
    verify,
    logout,
    setApiKey,
  };
};

