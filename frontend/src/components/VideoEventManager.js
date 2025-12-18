import React, { useState, useEffect } from "react";
import apiService from "../services/api";

const VideoEventManager = ({ videoId, apiKey, authenticated, onEventUpdated }) => {
  const [eventLabel, setEventLabel] = useState("");
  const [eventDescription, setEventDescription] = useState("");
  const [currentEvent, setCurrentEvent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // 預定義的事件類型
  const eventTypes = [
    "火災",
    "淹水積水",
    "人員倒地不起",
    "門禁遮臉入場",
    "車道併排阻塞",
    "離開吸菸區吸菸",
    "聚眾逗留",
    "安全門破壞/撬動",
    "其他",
  ];

  useEffect(() => {
    if (videoId && authenticated && apiKey) {
      loadVideoInfo();
    }
  }, [videoId, authenticated, apiKey]);

  const loadVideoInfo = async () => {
    if (!videoId) return;
    try {
      const data = await apiService.getVideoInfo(videoId, apiKey);
      if (data.event_label) {
        setCurrentEvent({
          label: data.event_label,
          description: data.event_description || "",
          setBy: data.event_set_by || "",
          setAt: data.event_set_at || "",
        });
        setEventLabel(data.event_label);
        setEventDescription(data.event_description || "");
      } else {
        setCurrentEvent(null);
        setEventLabel("");
        setEventDescription("");
      }
    } catch (error) {
      console.error("Failed to load video info:", error);
    }
  };

  const handleSetEvent = async () => {
    if (!eventLabel.trim()) {
      setMessage("請選擇或輸入事件類型");
      return;
    }

    setLoading(true);
    setMessage("");
    try {
      const result = await apiService.setVideoEvent(
        videoId,
        eventLabel,
        eventDescription,
        apiKey
      );
      setMessage(result.message || "事件標籤已設置");
      setCurrentEvent({
        label: eventLabel,
        description: eventDescription,
        setBy: "admin",
        setAt: new Date().toLocaleString("zh-TW"),
      });
      onEventUpdated?.();
      // 3秒後清除訊息
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      setMessage(`設置失敗：${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveEvent = async () => {
    if (!window.confirm("確定要移除事件標籤嗎？")) return;

    setLoading(true);
    setMessage("");
    try {
      await apiService.removeVideoEvent(videoId, apiKey);
      setMessage("事件標籤已移除");
      setCurrentEvent(null);
      setEventLabel("");
      setEventDescription("");
      onEventUpdated?.();
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      setMessage(`移除失敗：${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!videoId || !authenticated) return null;

  return (
    <div className="card" style={{ marginTop: "16px" }}>
      <div className="card-header">
        <div className="card-title">
          <span>🏷️</span>
          <span>事件標籤管理（管理者功能）</span>
        </div>
      </div>
      <div className="form-group">
        <label className="form-label">影片 ID</label>
        <div style={{ padding: "8px", background: "#f5f5f5", borderRadius: "4px" }}>
          {videoId}
        </div>
      </div>

      {currentEvent && (
        <div
          style={{
            padding: "12px",
            background: "#e8f4f8",
            borderRadius: "8px",
            marginBottom: "16px",
            border: "1px solid #b3d9e6",
          }}
        >
          <div style={{ fontWeight: "bold", marginBottom: "4px" }}>
            當前事件標籤：{currentEvent.label}
          </div>
          {currentEvent.description && (
            <div style={{ fontSize: "13px", color: "#666", marginBottom: "4px" }}>
              {currentEvent.description}
            </div>
          )}
          <div style={{ fontSize: "12px", color: "#999" }}>
            設置者：{currentEvent.setBy} | 設置時間：{currentEvent.setAt}
          </div>
        </div>
      )}

      <div className="form-group">
        <label className="form-label">事件類型</label>
        <select
          className="form-select"
          value={eventLabel}
          onChange={(e) => setEventLabel(e.target.value)}
        >
          <option value="">-- 請選擇事件類型 --</option>
          {eventTypes.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </select>
        <input
          className="form-input"
          style={{ marginTop: "8px" }}
          placeholder="或輸入自定義事件類型"
          value={eventLabel}
          onChange={(e) => setEventLabel(e.target.value)}
        />
      </div>

      <div className="form-group">
        <label className="form-label">事件描述（選填）</label>
        <textarea
          className="form-input"
          rows="3"
          placeholder="輸入事件詳細描述..."
          value={eventDescription}
          onChange={(e) => setEventDescription(e.target.value)}
        />
      </div>

      {message && (
        <div
          style={{
            padding: "8px 12px",
            marginBottom: "12px",
            borderRadius: "4px",
            background: message.includes("失敗") ? "#fee2e2" : "#d1fae5",
            color: message.includes("失敗") ? "#991b1b" : "#065f46",
          }}
        >
          {message}
        </div>
      )}

      <div style={{ display: "flex", gap: "8px" }}>
        <button
          className="btn btn-primary"
          onClick={handleSetEvent}
          disabled={loading || !eventLabel.trim()}
        >
          {loading ? "處理中..." : currentEvent ? "更新事件標籤" : "設置事件標籤"}
        </button>
        {currentEvent && (
          <button
            className="btn btn-ghost"
            onClick={handleRemoveEvent}
            disabled={loading}
            style={{ color: "#ef4444" }}
          >
            移除標籤
          </button>
        )}
      </div>
    </div>
  );
};

export default VideoEventManager;

