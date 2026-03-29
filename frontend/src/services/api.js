import { getApiBaseUrl } from "../utils/constants";

class ApiService {
  constructor() {
    this.baseUrl = getApiBaseUrl();
    console.log("API Service initialized with base URL:", this.baseUrl);
  }

  updateBaseUrl() {
    this.baseUrl = getApiBaseUrl();
    console.log("API Service base URL updated to:", this.baseUrl);
  }

  // 通用的 Fetch 包裝器，帶有診斷功能；可選 timeout（毫秒），逾時則中止請求
  async request(endpoint, options = {}) {
    const { headers = {}, body, method = "GET", apiKey, responseType = 'json', timeout } = options;
    const url = endpoint.startsWith("http") ? endpoint : `${this.baseUrl}${endpoint}`;

    const fetchOptions = {
      method,
      headers: {
        "Content-Type": "application/json",
        ...headers,
      },
    };

    let timeoutId;
    if (timeout != null && timeout > 0) {
      const controller = new AbortController();
      fetchOptions.signal = controller.signal;
      timeoutId = setTimeout(() => controller.abort(), timeout);
    }

    if (apiKey) fetchOptions.headers["X-API-Key"] = apiKey;

    if (body && !(body instanceof FormData)) {
      fetchOptions.body = JSON.stringify(body);
    } else if (body) {
      fetchOptions.body = body;
      delete fetchOptions.headers["Content-Type"];
    }

    try {
      console.log(`🚀 [Request] ${method} ${url}`);
      const response = await fetch(url, fetchOptions);
      if (timeoutId != null) clearTimeout(timeoutId);

      if (!response.ok) {
        console.group('🔍 Server Diagnostics');
        if (response.status === 502) {
          console.error("❌ Backend Unresponsive. Possible Causes: Timeout (>60s), OOM Crash, or Container Restarting.");
        }
        
        let errorDetail = `HTTP ${response.status}`;
        try {
          const data = await response.json();
          errorDetail = data.detail || data.message || errorDetail;
          console.error(`❌ ${method} ${endpoint} Failed:`, errorDetail);
          console.log("Raw Error Data:", data);
        } catch (e) {
          const text = await response.text();
          console.error(`❌ ${method} ${endpoint} Failed (Non-JSON):`, text.substring(0, 200));
        }
        console.groupEnd();
        
        const error = new Error(errorDetail);
        error.status = response.status;
        throw error;
      }

      if (responseType === 'blob') return await response.blob();
      return await response.json();

    } catch (error) {
      if (timeoutId != null) clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        console.group('🔍 Server Diagnostics');
        console.error(`❌ Request aborted after ${timeout}ms (timeout).`);
        console.groupEnd();
        throw new Error(`請求逾時（${Math.round(timeout / 1000)} 秒），請稍後再試或檢查後端負載。`);
      }
      if (error.name === 'TypeError' && error.message === 'Failed to fetch') {
        console.group('🔍 Server Diagnostics');
        console.error("❌ Network Failure. Check Proxy/Firewall settings.");
        console.groupEnd();
        throw new Error("Network Failure. Please check if the backend is running and reachable.");
      }
      throw error;
    }
  }

  async verifyAuth(apiKey) {
    try {
      return await this.request("/auth/verify", { method: "GET", apiKey });
    } catch (error) {
      return { ok: false, is_admin: false };
    }
  }

  async getRagStats(apiKey) {
    try {
      const data = await this.request("/rag/stats", { method: "GET", apiKey });
      return data.count || 0;
    } catch {
      return 0;
    }
  }

  async getDefaultPrompts(apiKey) {
    try {
      const data = await this.request("/prompts/defaults", { method: "GET", apiKey });
      return {
        event: data.event_prompt || "無法取得 Prompt",
        summary: data.summary_prompt || "無法取得 Prompt",
      };
    } catch (e) {
      return { event: "無法取得 Prompt", summary: "無法取得 Prompt" };
    }
  }

  async checkOllamaStatus(apiKey) {
    try {
      return await this.request("/health/ollama", { method: "GET", apiKey });
    } catch (e) {
      return { status: "error", available: false, error: e.message };
    }
  }

  async getSystemStatus(apiKey) {
    try {
      return await this.request("/v1/system/status", { method: "GET", apiKey });
    } catch (e) {
      return null;
    }
  }

  async getVlmStatus(apiKey) {
    try {
      return await this.request("/v1/system/vlm", { method: "GET", apiKey });
    } catch (e) {
      return null;
    }
  }

  async selectVlmProfile(apiKey, profileId) {
    return await this.request("/v1/system/vlm/select", {
      method: "POST",
      body: { profile_id: profileId },
      apiKey,
    });
  }

  // 使用 XMLHttpRequest 以支援上傳進度追蹤 (Point 6)
  // 本地影片分析可能需數分鐘，設定長逾時避免 proxy/XHR 提前中斷
  async runAnalysis(formData, apiKey, onUploadProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const url = `${this.baseUrl}/v1/segment_pipeline_multipart`;
      
      xhr.open("POST", url);
      xhr.setRequestHeader("X-API-Key", apiKey);
      xhr.timeout = 600000; // 10 分鐘，配合後端長時間分析

      if (onUploadProgress) {
        xhr.upload.onprogress = onUploadProgress;
      }

      xhr.ontimeout = () => {
        reject(new Error("請求逾時（分析時間過長），請檢查後端是否完成；結果可於稍後由日誌輪詢顯示。"));
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (e) {
            resolve(xhr.responseText);
          }
        } else {
          console.group('🔍 Server Diagnostics');
          if (xhr.status === 502) {
            console.error("❌ Backend Unresponsive. Possible Causes: Timeout (>60s), OOM Crash, or Container Restarting.");
          }
          
          let detail = `HTTP ${xhr.status}`;
          try {
            const data = JSON.parse(xhr.responseText);
            detail = data.detail || data.message || detail;
            console.error("❌ Analysis Failed:", detail);
            console.log("Raw Error Data:", data);
          } catch (e) {
            console.error("❌ Analysis Failed:", xhr.responseText.substring(0, 200));
          }
          console.groupEnd();
          
          const error = new Error(detail);
          error.status = xhr.status;
          reject(error);
        }
      };

      xhr.onerror = () => {
        console.group('🔍 Server Diagnostics');
        console.error("❌ Network Failure. Check Proxy/Firewall settings.");
        console.groupEnd();
        reject(new Error("Network Failure during upload."));
      };

      xhr.send(formData);
    });
  }

  /** 單段即時分析：傳 video_id + start_time + duration；後端 VLM+YOLO 可能需 1～3 分鐘，設定長逾時；連線失敗時自動重試一次 */
  async runInstantSegmentAnalysis(videoId, startTime, duration, apiKey) {
    const doRequest = () => this.request("/v1/analyze_single_segment", {
      method: "POST",
      body: { video_id: videoId, start_time: startTime, duration: duration ?? 10 },
      apiKey,
      timeout: 300000, // 5 分鐘，配合後端單段分析耗時
    });
    try {
      return await doRequest();
    } catch (err) {
      const isNetworkOrTimeout = err?.message?.includes("Network Failure") || err?.message?.includes("請求逾時") || err?.message?.includes("timeout");
      if (isNetworkOrTimeout) {
        await new Promise((r) => setTimeout(r, 2000)); // 等 2 秒後重試一次
        return await doRequest();
      }
      throw err;
    }
  }

  async searchRAG(query, topK, threshold, apiKey) {
    const data = await this.request("/rag/search", {
      method: "POST",
      body: { query, top_k: topK, score_threshold: threshold },
      apiKey,
    });
    this._logRagInfo(query, data);
    return data;
  }

  async answerRAG(query, topK, threshold, apiKey) {
    const data = await this.request("/rag/answer", {
      method: "POST",
      body: { query, top_k: topK, score_threshold: threshold },
      apiKey,
    });
    this._logRagInfo(query, data);
    return data;
  }

  _logRagInfo(query, data) {
    if (data.date_parsed || data.keywords_found || data.event_types_found || data.embedding_query) {
      console.log("%c" + "=".repeat(60), "color: #60a5fa; font-weight: bold");
      console.log("%c[查詢解析] 原始查詢: " + query, "color: #60a5fa; font-weight: bold");
      if (data.date_parsed) {
        const d = data.date_parsed;
        console.log("%c[查詢解析] 日期: 模式=" + (d.mode || "—") + (d.time_start ? " | " + d.time_start + " ~ " + (d.time_end || "—") : ""), "color: #34d399; font-weight: bold");
      }
      if (data.event_types_found && data.event_types_found.length > 0) {
        console.log("%c[查詢解析] 事件類型: " + data.event_types_found.join("、"), "color: #f59e0b; font-weight: bold");
      }
      if (data.keywords_found && data.keywords_found.length > 0) {
        console.log("%c[查詢解析] 關鍵字/特殊詞白名單對應: " + data.keywords_found.join("、"), "color: #a78bfa; font-weight: bold");
      }
      if (data.embedding_query) console.log("%c[向量搜索] Embedding 查詢文本: " + data.embedding_query, "color: #a78bfa; font-weight: bold");
      console.log("%c" + "=".repeat(60), "color: #60a5fa; font-weight: bold");
    }
  }

  async downloadFile(path, filename, apiKey) {
    const blob = await this.request(path, { method: "GET", apiKey, responseType: 'blob' });
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(downloadUrl);
  }

  async startRTSP(data, apiKey) { return await this.request("/v1/rtsp/start", { method: "POST", body: data, apiKey }); }
  async stopRTSP(data, apiKey) { return await this.request("/v1/rtsp/stop", { method: "POST", body: data, apiKey }); }
  async getRTSPStatus(apiKey) { return await this.request("/v1/rtsp/status", { method: "GET", apiKey }); }
  async listVideos(apiKey) { return await this.request("/v1/videos/list", { method: "GET", apiKey }); }
  async getVideoInfo(videoId, apiKey) { return await this.request(`/v1/videos/${videoId}`, { method: "GET", apiKey }); }
  async setVideoEvent(videoId, label, desc, apiKey) { return await this.request(`/v1/videos/${videoId}/event`, { method: "POST", body: { event_label: label, event_description: desc || "" }, apiKey }); }
  async removeVideoEvent(videoId, apiKey) { return await this.request(`/v1/videos/${videoId}/event`, { method: "DELETE", apiKey }); }
  async getVideoCategories(apiKey) { return await this.request("/v1/videos/categories", { method: "GET", apiKey }); }
  async moveVideoToCategory(videoId, cat, desc, apiKey) { return await this.request(`/v1/videos/${videoId}/move`, { method: "POST", body: { category: cat, event_description: desc || "" }, apiKey }); }
  async listDetectionItems(apiKey, enabledOnly = false) { return await this.request(`/detection-items?enabled_only=${enabledOnly}`, { method: "GET", apiKey }); }
  async createDetectionItem(data, apiKey) { return await this.request("/detection-items", { method: "POST", body: data, apiKey }); }
  async updateDetectionItem(id, data, apiKey) { return await this.request(`/detection-items/${id}`, { method: "PUT", body: data, apiKey }); }
  async deleteDetectionItem(id, apiKey) { return await this.request(`/detection-items/${id}`, { method: "DELETE", apiKey }); }
  async regeneratePrompt(apiKey) { return await this.request("/detection-items/regenerate-prompt", { method: "POST", apiKey }); }
  async previewPrompt(apiKey) { return await this.request("/detection-items/preview-prompt/content", { method: "GET", apiKey }); }
  async searchImage(formData, apiKey) { return await this.request("/v1/search/image", { method: "POST", body: formData, apiKey }); }
}

const apiService = new ApiService();
export default apiService;
