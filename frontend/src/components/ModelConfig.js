import React, { useState, useEffect } from "react";
import apiService from "../services/api";
import VideoSelector from "./VideoSelector";

/** 標題列右側：狀態燈 + 單行文字（高對比）；詳情與自動編排說明用 title 提示 */
function VlmHeaderStatus({ vlmStatus, vlmUiLocked }) {
  const sw = vlmStatus.switch;
  const ready = vlmStatus.readiness?.ready;
  const orch = vlmStatus.orchestration;

  let led = "#64748b";
  let label = "連線中";
  let pulse = true;
  let labelClass = "vlm-header-status-label vlm-header-status-label--warn";
  if (sw?.phase === "error") {
    led = "#dc2626";
    label = "錯誤";
    pulse = false;
    labelClass = "vlm-header-status-label vlm-header-status-label--error";
  } else if (vlmUiLocked || sw?.phase === "loading") {
    led = "#d97706";
    label = "載入中";
    pulse = true;
    labelClass = "vlm-header-status-label vlm-header-status-label--warn";
  } else if (ready) {
    led = "#16a34a";
    label = "就緒";
    pulse = false;
    labelClass = "vlm-header-status-label vlm-header-status-label--ok";
  }

  const errDetail =
    sw?.phase === "error" ? sw?.message || sw?.last_error || "後端錯誤" : "";
  const orchHint =
    orch?.enabled === true
      ? "已啟用自動編排：切換模型時會對主機執行 docker compose（停舊 vLLM/Ollama、啟動所選後端）。"
      : orch && orch.enabled === false
      ? "未啟用自動編排：僅儲存偏好，請在主機手動 docker compose stop/up。"
      : "";
  const extra =
    !ready && sw?.phase !== "error" && !(vlmUiLocked || sw?.phase === "loading")
      ? vlmStatus.readiness?.detail || ""
      : "";
  const titleParts = [errDetail, extra, orchHint].filter(Boolean);
  const title = titleParts.length ? titleParts.join("\n\n") : undefined;

  return (
    <div
      className="vlm-header-status"
      title={title}
      role="status"
      aria-live="polite"
    >
      <span
        aria-hidden
        className={`vlm-header-status-led${pulse ? " vlm-header-status-led--pulse" : ""}`}
        style={{ background: led }}
      />
      <span className={labelClass}>{label}</span>
    </div>
  );
}

const ModelConfig = ({
  modelType,
  qwenModel,
  vlmStatus,
  vlmUiLocked = false,
  source,
  videoUrl,
  videoFile,
  fileName,
  selectedVideoId,
  apiKey,
  authenticated,
  onModelTypeChange,
  onQwenModelChange,
  onSourceChange,
  onVideoUrlChange,
  onFileChange,
  onSelectedVideoChange,
}) => {
  const [availableVideos, setAvailableVideos] = useState([]);
  const [loadingVideos, setLoadingVideos] = useState(false);

  // 載入已上傳的影片列表
  useEffect(() => {
    if (authenticated && apiKey && source === "upload") {
      loadVideos();
    }
  }, [authenticated, apiKey, source]);

  const loadVideos = async () => {
    setLoadingVideos(true);
    try {
      const data = await apiService.listVideos(apiKey);
      setAvailableVideos(data.videos || []);
    } catch (error) {
      console.error("Failed to load videos:", error);
      setAvailableVideos([]);
    } finally {
      setLoadingVideos(false);
    }
  };
  return (
    <div className="card">
      <div className="card-header model-config-card-header">
        <div className="card-title">
          <span>⚙️</span>
          <span>Model & Source Configuration</span>
        </div>
        {(modelType === "qwen" || modelType === "vllm_qwen") && vlmStatus && (
          <VlmHeaderStatus vlmStatus={vlmStatus} vlmUiLocked={vlmUiLocked} />
        )}
      </div>
      <div className="form-grid">
        <div className="form-group">
          <label className="form-label">Model Type</label>
          <select
            className="form-select"
            value={modelType}
            disabled={vlmUiLocked}
            onChange={(e) => onModelTypeChange(e.target.value)}
          >
            <option value="qwen">Qwen (Multimodal via Ollama)</option>
            <option value="vllm_qwen">Qwen (vLLM)</option>
            <option value="moondream">Moondream (Local)</option>
            {/* <option value="gemini">Google Gemini (Cloud API)</option> */}
          </select>
        </div>

        {(modelType === "qwen" || modelType === "gemini" || modelType === "vllm_qwen") && (
          <div className="form-group">
            <label className="form-label">
              {modelType === "gemini"
                ? "Gemini Model Version"
                : modelType === "vllm_qwen"
                ? "vLLM Model"
                : "Qwen Model Version"}
            </label>
            <select
              className="form-select"
              value={qwenModel}
              disabled={vlmUiLocked}
              onChange={(e) => onQwenModelChange(e.target.value)}
            >
              {modelType === "gemini" ? (
                <>
                  <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                  <option value="gemini-2.5-pro">gemini-2.5-pro</option>
                </>
              ) : modelType === "vllm_qwen" ? (
                <>
                  <option value="Qwen/Qwen2.5-VL-7B-Instruct-AWQ">Qwen2.5-VL-7B-Instruct-AWQ</option>
                  <option value="Qwen/Qwen3-VL-8B-Instruct-FP8">Qwen3-VL-8B-Instruct-FP8</option>
                </>
              ) : (
                <>
                  <option value="qwen2.5vl:latest">qwen2.5vl:latest</option>
                  <option value="qwen3-vl:8b">qwen3-vl:8b</option>
                </>
              )}
            </select>
          </div>
        )}

        {modelType === "moondream" && (
          <div className="form-group">
            <label className="form-label">Model Version</label>
            <select
              className="form-select"
              value={qwenModel}
              disabled={vlmUiLocked}
              onChange={(e) => onQwenModelChange(e.target.value)}
            >
              <option value="moondream3-preview">moondream3-preview</option>
              <option value="moondream-2b-2025-04-14">moondream-2b-2025-04-14</option>
            </select>
          </div>
        )}

        <div className="form-group">
          <label className="form-label">Video Source</label>
          <select
            className="form-select"
            value={source}
            onChange={(e) => {
              onSourceChange(e.target.value);
              // 切換來源時清除選擇
              if (e.target.value === "url") {
                onSelectedVideoChange(null);
                onFileChange({ target: { files: [] } });
              } else if (e.target.value === "existing") {
                loadVideos();
              }
            }}
          >
            <option value="upload">Local Upload</option>
            <option value="url">URL</option>
            <option value="existing">選擇已上傳影片</option>
          </select>
        </div>
      </div>

      {source === "url" && (
        <div className="form-group">
          <label className="form-label">Video URL</label>
          <input
            className="form-input"
            placeholder="https://example.com/video.mp4"
            value={videoUrl}
            onChange={(e) => onVideoUrlChange(e.target.value)}
          />
        </div>
      )}

      {source === "existing" && (
        <VideoSelector
          selectedVideoId={selectedVideoId}
          onVideoChange={(videoId) => {
            onSelectedVideoChange(videoId);
            onFileChange({ target: { files: [] } });
          }}
          apiKey={apiKey}
          authenticated={authenticated}
        />
      )}

      {source === "upload" && (
        <>
        <div className="form-group">
          <label className="form-label">Upload Video File</label>
          <label
            htmlFor="video_file"
            className={`file-upload-area ${fileName ? "has-file" : ""}`}
          >
            <div className="file-upload-icon">{fileName ? "✓" : "📁"}</div>
            <div className="file-upload-text">
              {fileName || "Click to select video file"}
            </div>
            <input
              id="video_file"
              type="file"
              accept="video/*"
                onChange={(e) => {
                  onFileChange(e);
                  // 清除已選擇的影片
                  onSelectedVideoChange(null);
                }}
            />
          </label>
        </div>
          {availableVideos.length > 0 && (
            <div className="form-group" style={{ marginTop: "12px" }}>
              <div style={{ fontSize: "12px", color: "#666", marginBottom: "4px" }}>
                提示：已上傳過的影片可以直接選擇，無需重新上傳
              </div>
              <button
                type="button"
                onClick={() => {
                  onSourceChange("existing");
                  loadVideos();
                }}
                style={{
                  padding: "6px 12px",
                  fontSize: "12px",
                  background: "#2a2a2a",
                  border: "1px solid #444",
                  borderRadius: "4px",
                  cursor: "pointer",
                  color: "#ccc",
                  transition: "all 0.2s ease",
                }}
                onMouseEnter={(e) => {
                  e.target.style.background = "#3a3a3a";
                  e.target.style.borderColor = "#555";
                }}
                onMouseLeave={(e) => {
                  e.target.style.background = "#2a2a2a";
                  e.target.style.borderColor = "#444";
                }}
              >
                查看已上傳影片 ({availableVideos.length})
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ModelConfig;

