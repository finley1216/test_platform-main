import { getApiBaseUrl } from "../utils/constants";

class ApiService {
  constructor() {
    // 在運行時動態獲取 API base URL
    this.baseUrl = getApiBaseUrl();
    console.log("API Service initialized with base URL:", this.baseUrl);
  }
  
  // 提供方法讓外部可以更新 base URL（如果需要）
  updateBaseUrl() {
    this.baseUrl = getApiBaseUrl();
    console.log("API Service base URL updated to:", this.baseUrl);
  }

  async request(endpoint, options = {}) {
    const { headers = {}, body, method = "GET", apiKey } = options;

    const config = {
      method,
      headers: {
        "Content-Type": "application/json",
        ...headers,
      },
    };

    if (apiKey) {
      config.headers["X-API-Key"] = apiKey;
    }

    if (body && !(body instanceof FormData)) {
      config.body = JSON.stringify(body);
    } else if (body) {
      config.body = body;
      delete config.headers["Content-Type"];
    }

    const url = `${this.baseUrl}${endpoint}`;
    console.log(`Making request to: ${url}`, { method, hasApiKey: !!apiKey });

    let response;
    try {
      response = await fetch(url, config);
    } catch (error) {
      console.error("Network error during fetch:", error);
      throw new Error(`Network error: ${error.message}. Please check if the backend is running and accessible at ${url}`);
    }

    // Check if response has content before trying to parse JSON
    const contentType = response.headers.get("content-type");
    let data;

    if (contentType && contentType.includes("application/json")) {
      try {
        data = await response.json();
      } catch (error) {
        console.error("Failed to parse JSON response:", error);
        throw new Error(`Invalid JSON response from server (HTTP ${response.status})`);
      }
    } else {
      // If not JSON, read as text for error message
      const text = await response.text();
      throw new Error(`Unexpected response format (HTTP ${response.status}): ${text.substring(0, 100)}`);
    }

    if (!response.ok) {
      throw new Error(data.error || data.message || `HTTP ${response.status}`);
    }

    return data;
  }

  async verifyAuth(apiKey) {
    try {
      console.log("Verifying API key with /auth/verify endpoint...");
      const data = await this.request("/auth/verify", {
        method: "GET",
        apiKey,
      });
      console.log("Auth verification response:", data);
      return data; // 返回完整數據，包含 is_admin
    } catch (error) {
      console.error("Auth verification failed:", error);
      return { ok: false, is_admin: false };
    }
  }

  async getRagStats(apiKey) {
    try {
      const data = await this.request("/rag/stats", {
        method: "GET",
        apiKey,
      });
      return data.count || 0;
    } catch {
      return 0;
    }
  }

  async getDefaultPrompts(apiKey) {
    try {
      const data = await this.request("/prompts/defaults", {
        method: "GET",
        apiKey,
      });
      return {
        event: data.event_prompt || "無法取得 Prompt",
        summary: data.summary_prompt || "無法取得 Prompt",
      };
    } catch (e) {
      console.error("Failed to fetch default prompts", e);
      return {
        event: "無法取得 Prompt",
        summary: "無法取得 Prompt",
      };
    }
  }

  async checkOllamaStatus(apiKey) {
    try {
      const data = await this.request("/health/ollama", {
        method: "GET",
        apiKey,
      });
      return data;
    } catch (e) {
      console.error("Failed to check Ollama status", e);
      return {
        status: "error",
        available: false,
        error: e.message,
      };
    }
  }

  async runAnalysis(formData, apiKey) {
    return this.request("/v1/segment_pipeline_multipart", {
      method: "POST",
      body: formData,
      apiKey,
      headers: {},
    });
  }

  async searchRAG(query, topK, threshold, apiKey) {
    const data = await this.request("/rag/search", {
      method: "POST",
      body: {
        query,
        top_k: topK,
        score_threshold: threshold,
      },
      apiKey,
    });
    
    // [NEW] 在控制台輸出日期解析資訊、關鍵字資訊和 embedding 查詢資訊
    if (data.date_parsed || data.keywords_found || data.event_types_found || data.embedding_query) {
      console.log("%c" + "=".repeat(60), "color: #60a5fa; font-weight: bold");
      console.log("%c[查詢解析] 原始查詢: " + query, "color: #60a5fa; font-weight: bold");
      
      if (data.embedding_query) {
        console.log("%c[向量搜索] Embedding 查詢文本: " + data.embedding_query, "color: #a78bfa; font-weight: bold");
        console.log("%c[向量搜索] 說明: 此文本用於生成 embedding 向量進行語義搜索", "color: #a78bfa; font-size: 11px");
      }
      
      if (data.date_parsed) {
        const dp = data.date_parsed;
        console.log("%c[日期解析] 模式: " + dp.mode, "color: #60a5fa");
        console.log("%c[日期解析] 解析到的日期: " + (dp.picked_date || "N/A"), "color: #60a5fa");
        if (dp.time_start) {
          const start = dp.time_start.substring(0, 19);
          const end = dp.time_end ? dp.time_end.substring(0, 19) : "N/A";
          console.log("%c[日期解析] 時間範圍: " + start + " ~ " + end, "color: #60a5fa");
        }
      }
      
      if (data.keywords_found && data.keywords_found.length > 0) {
        console.log("%c[關鍵字匹配] " + data.keywords_found.join(", "), "color: #34d399; font-weight: bold");
      }
      
      if (data.event_types_found && data.event_types_found.length > 0) {
        console.log("%c[事件類型] " + data.event_types_found.join(", "), "color: #fbbf24; font-weight: bold");
      }
      
      if (data.hits && data.hits.length > 0) {
        console.log("%c[搜索結果] 找到 " + data.hits.length + " 筆匹配記錄", "color: #10b981; font-weight: bold");
        // 顯示前 3 筆的摘要片段
        data.hits.slice(0, 3).forEach((hit, idx) => {
          const summary = hit.summary || "";
          const preview = summary.length > 50 ? summary.substring(0, 50) + "..." : summary;
          console.log("%c  結果 " + (idx + 1) + ": " + preview, "color: #6b7280; font-size: 12px");
        });
      }
      
      console.log("%c" + "=".repeat(60), "color: #60a5fa; font-weight: bold");
    }
    
    return data;
  }

  async answerRAG(query, topK, threshold, apiKey) {
    const data = await this.request("/rag/answer", {
      method: "POST",
      body: {
        query,
        top_k: topK,
        score_threshold: threshold,
      },
      apiKey,
    });
    
    // [NEW] 在控制台輸出日期解析資訊、關鍵字資訊和 embedding 查詢資訊
    if (data.date_parsed || data.keywords_found || data.event_types_found || data.embedding_query) {
      console.log("%c" + "=".repeat(60), "color: #60a5fa; font-weight: bold");
      console.log("%c[查詢解析] 原始查詢: " + query, "color: #60a5fa; font-weight: bold");
      
      if (data.embedding_query) {
        console.log("%c[向量搜索] Embedding 查詢文本: " + data.embedding_query, "color: #a78bfa; font-weight: bold");
        console.log("%c[向量搜索] 說明: 此文本用於生成 embedding 向量進行語義搜索", "color: #a78bfa; font-size: 11px");
      }
      
      if (data.date_parsed) {
        const dp = data.date_parsed;
        console.log("%c[日期解析] 模式: " + dp.mode, "color: #60a5fa");
        console.log("%c[日期解析] 解析到的日期: " + (dp.picked_date || "N/A"), "color: #60a5fa");
        if (dp.time_start) {
          const start = dp.time_start.substring(0, 19);
          const end = dp.time_end ? dp.time_end.substring(0, 19) : "N/A";
          console.log("%c[日期解析] 時間範圍: " + start + " ~ " + end, "color: #60a5fa");
        }
      }
      
      if (data.keywords_found && data.keywords_found.length > 0) {
        console.log("%c[關鍵字匹配] " + data.keywords_found.join(", "), "color: #34d399; font-weight: bold");
      }
      
      if (data.event_types_found && data.event_types_found.length > 0) {
        console.log("%c[事件類型] " + data.event_types_found.join(", "), "color: #fbbf24; font-weight: bold");
      }
      
      if (data.hits && data.hits.length > 0) {
        console.log("%c[搜索結果] 找到 " + data.hits.length + " 筆匹配記錄", "color: #10b981; font-weight: bold");
        // 顯示前 3 筆的摘要片段
        data.hits.slice(0, 3).forEach((hit, idx) => {
          const summary = hit.summary || "";
          const preview = summary.length > 50 ? summary.substring(0, 50) + "..." : summary;
          console.log("%c  結果 " + (idx + 1) + ": " + preview, "color: #6b7280; font-size: 12px");
        });
      }
      
      console.log("%c" + "=".repeat(60), "color: #60a5fa; font-weight: bold");
    }
    
    return data;
  }

  async downloadFile(path, filename, apiKey) {
    // 處理路徑：確保以 / 開頭，並正確構建 URL
    let normalizedPath = path;
    if (!normalizedPath.startsWith("/")) {
      normalizedPath = `/${normalizedPath}`;
    }
    
    const targetUrl = path.startsWith("http")
      ? path
      : `${this.baseUrl}${normalizedPath}`;

    const response = await fetch(targetUrl, {
      headers: {
        "X-API-Key": apiKey,
      },
    });

    if (!response.ok) {
      throw new Error(`下載失敗 (HTTP ${response.status})`);
    }

    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("text/html")) {
      throw new Error(
        "路徑錯誤：伺服器回傳了網頁而非影片，請檢查 API_BASE 設定"
      );
    }

    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
    
    // 注意：在 HTTP 頁面上使用 blob URL 會產生瀏覽器安全警告
    // "The file at 'blob:...' was loaded over an insecure connection"
    // 這是預期的警告，不影響下載功能。要消除警告，需要：
    // 1. 使用 HTTPS（推薦，需要配置 SSL 證書）
    // 2. 或使用直接下載連結（但需要後端支援 CORS 和適當的 Content-Disposition 標頭）
  }

  // 影片管理 API
  async listVideos(apiKey) {
    return this.request("/v1/videos/list", {
      method: "GET",
      apiKey,
    });
  }

  async getVideoInfo(videoId, apiKey) {
    return this.request(`/v1/videos/${videoId}`, {
      method: "GET",
      apiKey,
    });
  }

  async setVideoEvent(videoId, eventLabel, eventDescription, apiKey) {
    return this.request(`/v1/videos/${videoId}/event`, {
      method: "POST",
      body: {
        event_label: eventLabel,
        event_description: eventDescription || "",
      },
      apiKey,
    });
  }

  async removeVideoEvent(videoId, apiKey) {
    return this.request(`/v1/videos/${videoId}/event`, {
      method: "DELETE",
      apiKey,
    });
  }

  async getVideoCategories(apiKey) {
    return this.request("/v1/videos/categories", {
      method: "GET",
      apiKey,
    });
  }

  async moveVideoToCategory(videoId, category, eventDescription, apiKey) {
    return this.request(`/v1/videos/${videoId}/move`, {
      method: "POST",
      body: {
        category,
        event_description: eventDescription || "",
      },
      apiKey,
    });
  }

  // 以圖搜圖 API
  async searchImage(formData, apiKey, timeoutMs = 60000) {
    // 使用 fetch 直接調用以支持超時和進度追蹤
    // 預設超時時間為 60 秒（以圖搜圖需要載入 CLIP 模型和執行向量搜索）
    const url = `${this.baseUrl}/v1/search/image`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
    
    try {
      const startTime = Date.now();
      console.log(`[以圖搜圖] 發送請求到: ${url}`);
      console.log(`[以圖搜圖] 超時設置: ${timeoutMs / 1000} 秒`);
      console.log(`[以圖搜圖] 開始時間: ${new Date().toISOString()}`);
      
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "X-API-Key": apiKey,
        },
        body: formData,
        signal: controller.signal,
      });
      
      const elapsed = Date.now() - startTime;
      console.log(`[以圖搜圖] 收到回應，耗時: ${elapsed}ms`);
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}`;
        let errorDetail = null;
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorData.error || errorData.message || errorMessage;
          errorDetail = errorData;
          console.error("[以圖搜圖] 後端返回錯誤:", errorData);
        } catch (e) {
          // 如果無法解析 JSON，嘗試讀取文字
          try {
            const text = await response.text();
            if (text) {
              errorMessage = text.substring(0, 500);
              errorDetail = { raw_text: text };
              console.error("[以圖搜圖] 後端返回錯誤（文字格式):", text);
            }
          } catch (e2) {
            console.error("[以圖搜圖] 無法讀取錯誤回應:", e2);
          }
        }
        
        // 創建包含詳細信息的錯誤
        const error = new Error(errorMessage);
        error.status = response.status;
        error.detail = errorDetail;
        throw error;
      }
      
      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      
      // 處理不同類型的錯誤
      if (error.name === "AbortError") {
        throw new Error("請求超時（超過 60 秒）");
      }
      
      // 處理網路錯誤
      if (error.message && error.message.includes("Failed to fetch")) {
        const errorMsg = `無法連接到後端服務器。請檢查：
1. 後端服務是否正在運行（${this.baseUrl}）
2. 網路連線是否正常
3. 是否有 CORS 設定問題
4. API 基礎 URL 是否正確設定`;
        console.error("[以圖搜圖] 網路錯誤:", errorMsg);
        throw new Error(errorMsg);
      }
      
      // 如果是我們自己拋出的錯誤，直接傳遞
      if (error.message) {
        throw error;
      }
      
      // 其他未知錯誤
      throw new Error(`搜索失敗: ${error.toString()}`);
    }
  }
}

const apiService = new ApiService();
export default apiService;

