import React, { useState, useEffect } from "react";
import apiService from "../services/api";
import VideoSelector from "./VideoSelector";

const ModelConfig = ({
  modelType,
  qwenModel,
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
      <div className="card-header">
        <div className="card-title">
          <span>⚙️</span>
          <span>Model & Source Configuration</span>
        </div>
      </div>
      <div className="form-grid">
        <div className="form-group">
          <label className="form-label">Model Type</label>
          <select
            className="form-select"
            value={modelType}
            onChange={(e) => {
              const v = e.target.value;
              onModelTypeChange(v);
              if (v === "moondream") onQwenModelChange("moondream3-preview");
            }}
          >
            <option value="qwen">Qwen (Multimodal via Ollama)</option>
            <option value="moondream">Moondream (Local)</option>
            {/* <option value="gemini">Google Gemini (Cloud API)</option> */}
          </select>
        </div>

        {(modelType === "qwen" || modelType === "gemini") && (
          <div className="form-group">
            <label className="form-label">
              {modelType === "gemini"
                ? "Gemini Model Version"
                : "Qwen Model Version"}
            </label>
            <select
              className="form-select"
              value={qwenModel}
              onChange={(e) => onQwenModelChange(e.target.value)}
            >
              {modelType === "gemini" ? (
                <>
                  <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                  <option value="gemini-2.5-pro">gemini-2.5-pro</option>
                </>
              ) : (
                <>
                  <option value="qwen3-vl:8b">qwen3-vl:8b</option>
                  <option value="qwen2.5vl:latest">qwen2.5vl:latest</option>
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

