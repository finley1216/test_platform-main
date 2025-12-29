import { API_BASE } from "../utils/constants";

class ApiService {
  constructor() {
    this.baseUrl = API_BASE;
    console.log("API Service initialized with base URL:", this.baseUrl);
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
    const targetUrl = path.startsWith("http")
      ? path
      : `${this.baseUrl}/${path.replace(/^\//, "")}`;

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
}

const apiService = new ApiService();
export default apiService;

