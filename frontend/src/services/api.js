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
    return this.request("/rag/search", {
      method: "POST",
      body: {
        query,
        top_k: topK,
        score_threshold: threshold,
      },
      apiKey,
    });
  }

  async answerRAG(query, topK, threshold, apiKey) {
    return this.request("/rag/answer", {
      method: "POST",
      body: {
        query,
        top_k: topK,
        score_threshold: threshold,
      },
      apiKey,
    });
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

