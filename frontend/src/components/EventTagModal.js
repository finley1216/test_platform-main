import React, { useState, useEffect } from "react";
import apiService from "../services/api";

const EventTagModal = ({ isOpen, onClose, apiKey }) => {
  const [videos, setVideos] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedVideoId, setSelectedVideoId] = useState("");
  const [eventLabel, setEventLabel] = useState("");
  const [eventDescription, setEventDescription] = useState("");
  const [currentEvent, setCurrentEvent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingVideos, setLoadingVideos] = useState(false);
  const [message, setMessage] = useState("");
  const [showMoveDialog, setShowMoveDialog] = useState(false);
  const [moveCategory, setMoveCategory] = useState("");

  // 預定義的事件類型
  const eventTypes = [
    "火災生成",
    "水災生成",
    "人員倒地不起",
    "門禁遮臉入場",
    "車道併排阻塞",
    "離開吸菸區吸菸",
    "聚眾逗留",
    "安全門破壞/撬動",
    "其他",
  ];

  useEffect(() => {
    if (isOpen && apiKey) {
      loadVideos();
    }
  }, [isOpen, apiKey]);

  useEffect(() => {
    if (selectedVideoId && apiKey) {
      loadVideoInfo();
    } else {
      setCurrentEvent(null);
      setEventLabel("");
      setEventDescription("");
    }
  }, [selectedVideoId, apiKey]);

  const loadVideos = async () => {
    setLoadingVideos(true);
    try {
      const data = await apiService.listVideos(apiKey);
      setVideos(data.videos || []);
      setCategories(data.categories || []);
      
      // 載入分類列表
      try {
        const catData = await apiService.getVideoCategories(apiKey);
        if (catData.categories) {
          setCategories(catData.categories);
        }
      } catch (e) {
        console.error("Failed to load categories:", e);
      }
    } catch (error) {
      console.error("Failed to load videos:", error);
      setVideos([]);
      setCategories([]);
    } finally {
      setLoadingVideos(false);
    }
  };

  const loadVideoInfo = async () => {
    if (!selectedVideoId) return;
    try {
      const data = await apiService.getVideoInfo(selectedVideoId, apiKey);
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
      setCurrentEvent(null);
    }
  };

  const handleSetEvent = async () => {
    if (!eventLabel.trim()) {
      setMessage("請選擇或輸入事件類型");
      return;
    }

    if (!selectedVideoId) {
      setMessage("請先選擇影片");
      return;
    }

    setLoading(true);
    setMessage("");
    try {
      const result = await apiService.setVideoEvent(
        selectedVideoId,
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
      loadVideos(); // 重新載入影片列表以更新標籤顯示
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
      await apiService.removeVideoEvent(selectedVideoId, apiKey);
      setMessage("事件標籤已移除");
      setCurrentEvent(null);
      setEventLabel("");
      setEventDescription("");
      loadVideos(); // 重新載入影片列表
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      setMessage(`移除失敗：${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleMoveToCategory = async () => {
    if (!moveCategory.trim()) {
      setMessage("請選擇分類");
      return;
    }

    if (!selectedVideoId) {
      setMessage("請先選擇影片");
      return;
    }

    // 檢查影片是否來自 segment（只有 segment 中的影片可以移動）
    const selectedVideo = videos.find((v) => v.video_id === selectedVideoId);
    if (!selectedVideo || selectedVideo.source !== "segment") {
      setMessage("只能移動 segment 中的影片到分類資料夾");
      return;
    }

    setLoading(true);
    setMessage("");
    try {
      const result = await apiService.moveVideoToCategory(
        selectedVideoId,
        moveCategory,
        eventDescription,
        apiKey
      );
      setMessage(result.message || "影片已移動到分類");
      setShowMoveDialog(false);
      setMoveCategory("");
      setSelectedVideoId("");
      setEventLabel("");
      setEventDescription("");
      loadVideos(); // 重新載入影片列表
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      setMessage(`移動失敗：${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: "#1a1a1a",
          borderRadius: "8px",
          padding: "24px",
          width: "90%",
          maxWidth: "600px",
          maxHeight: "90vh",
          overflow: "auto",
          border: "1px solid #333",
          boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "24px",
            borderBottom: "1px solid #333",
            paddingBottom: "16px",
          }}
        >
          <h2 style={{ color: "#fff", margin: 0, fontSize: "20px", fontWeight: "600" }}>
            事件標籤管理
          </h2>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "#999",
              fontSize: "24px",
              cursor: "pointer",
              padding: "0",
              width: "32px",
              height: "32px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            ×
          </button>
        </div>

        <div style={{ marginBottom: "20px" }}>
          <label
            style={{
              display: "block",
              color: "#ccc",
              marginBottom: "8px",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            選擇影片
          </label>
          {loadingVideos ? (
            <div
              style={{
                padding: "12px",
                background: "#2a2a2a",
                borderRadius: "4px",
                color: "#999",
                textAlign: "center",
              }}
            >
              載入中...
            </div>
          ) : (
            <>
              <select
                value={selectedVideoId}
                onChange={(e) => setSelectedVideoId(e.target.value)}
                style={{
                  width: "100%",
                  padding: "10px",
                  background: "#2a2a2a",
                  border: "1px solid #444",
                  borderRadius: "4px",
                  color: "#fff",
                  fontSize: "14px",
                  marginBottom: "8px",
                }}
              >
                <option value="">-- 請選擇影片 --</option>
                {videos.map((video) => (
                    <option key={video.video_id} value={video.video_id}>
                      {video.display_name}
                      {video.category && ` [${video.category}]`}
                      {video.event_label && !video.category && ` [${video.event_label}]`}
                      {video.source === "segment" &&
                        ` (${video.success_segments}/${video.total_segments} 片段)`}
                      {video.source === "video_lib" && ` [未處理]`}
                    </option>
                  ))}
              </select>
              {selectedVideoId && (
                <div
                  style={{
                    marginTop: "12px",
                    padding: "12px",
                    paddingBottom: "16px",
                    background: "#2a2a2a",
                    borderRadius: "6px",
                    border: "1px solid #444",
                    fontSize: "13px",
                    minHeight: "auto",
                    height: "auto",
                    maxHeight: "none",
                    overflow: "visible",
                    wordWrap: "break-word",
                    whiteSpace: "normal",
                    display: "block",
                  }}
                >
                  {(() => {
                    const selectedVideo = videos.find(
                      (v) => v.video_id === selectedVideoId
                    );
                    if (!selectedVideo) return null;
                    return (
                      <>
                        <div
                          style={{
                            color: "#fff",
                            fontWeight: "600",
                            marginBottom: "8px",
                            wordBreak: "break-word",
                          }}
                        >
                          {selectedVideo.display_name}
                        </div>
                        {/* 如果分類和事件類型相同，只顯示事件類型 */}
                        {selectedVideo.event_label && (
                          <div style={{ color: "#999", marginBottom: "4px", wordBreak: "break-word" }}>
                            事件類型：{selectedVideo.event_label}
                          </div>
                        )}
                        {selectedVideo.category && 
                         selectedVideo.category !== selectedVideo.event_label && (
                          <div style={{ color: "#999", marginBottom: "4px", wordBreak: "break-word" }}>
                            分類：{selectedVideo.category}
                          </div>
                        )}
                        {selectedVideo.event_description && (
                          <div
                            style={{
                              color: "#ccc",
                              marginTop: "8px",
                              paddingTop: "8px",
                              borderTop: "1px solid #444",
                              lineHeight: "1.5",
                              wordBreak: "break-word",
                              whiteSpace: "pre-wrap",
                            }}
                          >
                            <div
                              style={{
                                color: "#888",
                                fontSize: "12px",
                                marginBottom: "4px",
                              }}
                            >
                              事件描述：
                            </div>
                            {selectedVideo.event_description}
                          </div>
                        )}
                        {selectedVideo.source === "segment" && (
                          <div
                            style={{
                              color: "#999",
                              marginTop: "8px",
                              fontSize: "12px",
                              paddingBottom: "4px",
                              display: "block",
                              visibility: "visible",
                              opacity: 1,
                            }}
                          >
                            已分析：{selectedVideo.success_segments}/
                            {selectedVideo.total_segments} 片段
                          </div>
                        )}
                        {selectedVideo.source === "video_lib" && (
                          <div
                            style={{
                              color: "#999",
                              marginTop: "8px",
                              fontSize: "12px",
                              paddingBottom: "4px",
                              display: "block",
                              visibility: "visible",
                              opacity: 1,
                            }}
                          >
                            來源：影片庫（尚未分析）
                          </div>
                        )}
                      </>
                    );
                  })()}
                </div>
              )}
            </>
          )}
        </div>

        {currentEvent && (
          <div
            style={{
              padding: "16px",
              background: "#2a2a2a",
              borderRadius: "8px",
              marginBottom: "20px",
              border: "1px solid #444",
            }}
          >
            <div style={{ color: "#fff", fontWeight: "600", marginBottom: "8px" }}>
              當前事件標籤：{currentEvent.label}
            </div>
            {currentEvent.description && (
              <div style={{ color: "#999", fontSize: "13px", marginBottom: "8px" }}>
                {currentEvent.description}
              </div>
            )}
            <div style={{ color: "#666", fontSize: "12px" }}>
              設置者：{currentEvent.setBy} | 設置時間：{currentEvent.setAt}
            </div>
          </div>
        )}

        <div style={{ marginBottom: "20px" }}>
          <label
            style={{
              display: "block",
              color: "#ccc",
              marginBottom: "8px",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            事件類型
          </label>
          <select
            value={eventLabel}
            onChange={(e) => setEventLabel(e.target.value)}
            style={{
              width: "100%",
              padding: "10px",
              background: "#2a2a2a",
              border: "1px solid #444",
              borderRadius: "4px",
              color: "#fff",
              fontSize: "14px",
              marginBottom: "8px",
            }}
          >
            <option value="">-- 請選擇事件類型 --</option>
            {eventTypes.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
          <input
            type="text"
            placeholder="或輸入自定義事件類型"
            value={eventLabel}
            onChange={(e) => setEventLabel(e.target.value)}
            style={{
              width: "100%",
              padding: "10px",
              background: "#2a2a2a",
              border: "1px solid #444",
              borderRadius: "4px",
              color: "#fff",
              fontSize: "14px",
            }}
          />
        </div>

        <div style={{ marginBottom: "20px" }}>
          <label
            style={{
              display: "block",
              color: "#ccc",
              marginBottom: "8px",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            事件描述（選填）
          </label>
          <textarea
            rows="3"
            placeholder="輸入事件詳細描述..."
            value={eventDescription}
            onChange={(e) => setEventDescription(e.target.value)}
            style={{
              width: "100%",
              padding: "10px",
              background: "#2a2a2a",
              border: "1px solid #444",
              borderRadius: "4px",
              color: "#fff",
              fontSize: "14px",
              fontFamily: "inherit",
              resize: "vertical",
            }}
          />
        </div>

        {message && (
          <div
            style={{
              padding: "12px",
              marginBottom: "16px",
              borderRadius: "4px",
              background: message.includes("失敗") ? "#3a1a1a" : "#1a3a1a",
              color: message.includes("失敗") ? "#ff6b6b" : "#6bff6b",
              fontSize: "13px",
            }}
          >
            {message}
          </div>
        )}

        {/* 移動到分類對話框 */}
        {showMoveDialog && selectedVideoId && (
          <div
            style={{
              marginTop: "20px",
              marginBottom: "20px",
              padding: "16px",
              background: "#2a2a2a",
              borderRadius: "6px",
              border: "1px solid #555",
            }}
          >
            <div
              style={{
                color: "#fff",
                fontWeight: "600",
                marginBottom: "12px",
              }}
            >
              移動影片到分類資料夾
            </div>
            <div style={{ marginBottom: "12px" }}>
              <label
                style={{
                  display: "block",
                  color: "#ccc",
                  marginBottom: "6px",
                  fontSize: "13px",
                }}
              >
                選擇分類
              </label>
              <select
                value={moveCategory}
                onChange={(e) => setMoveCategory(e.target.value)}
                style={{
                  width: "100%",
                  padding: "8px",
                  background: "#1a1a1a",
                  border: "1px solid #444",
                  borderRadius: "4px",
                  color: "#fff",
                  fontSize: "13px",
                }}
              >
                <option value="">-- 請選擇分類 --</option>
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            </div>
            <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
              <button
                onClick={() => {
                  setShowMoveDialog(false);
                  setMoveCategory("");
                }}
                style={{
                  padding: "6px 16px",
                  background: "#2a2a2a",
                  border: "1px solid #444",
                  borderRadius: "4px",
                  color: "#ccc",
                  cursor: "pointer",
                  fontSize: "13px",
                }}
              >
                取消
              </button>
              <button
                onClick={handleMoveToCategory}
                disabled={loading || !moveCategory.trim()}
                style={{
                  padding: "6px 16px",
                  background: loading || !moveCategory.trim() ? "#2a2a2a" : "#4a4a4a",
                  border: "1px solid #666",
                  borderRadius: "4px",
                  color: loading || !moveCategory.trim() ? "#666" : "#fff",
                  cursor: loading || !moveCategory.trim() ? "not-allowed" : "pointer",
                  fontSize: "13px",
                  opacity: loading || !moveCategory.trim() ? 0.5 : 1,
                }}
              >
                {loading ? "處理中..." : "確認移動"}
              </button>
            </div>
          </div>
        )}

        <div style={{ display: "flex", gap: "12px", justifyContent: "flex-end", flexWrap: "wrap" }}>
          <button
            onClick={onClose}
            style={{
              padding: "10px 20px",
              background: "#2a2a2a",
              border: "1px solid #444",
              borderRadius: "4px",
              color: "#ccc",
              cursor: "pointer",
              fontSize: "14px",
            }}
          >
            關閉
          </button>
          {selectedVideoId &&
            videos.find((v) => v.video_id === selectedVideoId)?.source ===
              "segment" && (
              <button
                onClick={() => setShowMoveDialog(!showMoveDialog)}
                disabled={loading}
                style={{
                  padding: "10px 20px",
                  background: "#3a3a1a",
                  border: "1px solid #5a5a2a",
                  borderRadius: "4px",
                  color: "#ffd700",
                  cursor: loading ? "not-allowed" : "pointer",
                  fontSize: "14px",
                  opacity: loading ? 0.5 : 1,
                }}
              >
                {showMoveDialog ? "取消移動" : "移動到分類"}
              </button>
            )}
          {currentEvent && (
            <button
              onClick={handleRemoveEvent}
              disabled={loading}
              style={{
                padding: "10px 20px",
                background: "#3a1a1a",
                border: "1px solid #5a2a2a",
                borderRadius: "4px",
                color: "#ff6b6b",
                cursor: loading ? "not-allowed" : "pointer",
                fontSize: "14px",
                opacity: loading ? 0.5 : 1,
              }}
            >
              移除標籤
            </button>
          )}
          <button
            onClick={handleSetEvent}
            disabled={loading || !eventLabel.trim() || !selectedVideoId}
            style={{
              padding: "10px 20px",
              background: loading || !eventLabel.trim() || !selectedVideoId ? "#2a2a2a" : "#4a4a4a",
              border: "1px solid #666",
              borderRadius: "4px",
              color: loading || !eventLabel.trim() || !selectedVideoId ? "#666" : "#fff",
              cursor: loading || !eventLabel.trim() || !selectedVideoId ? "not-allowed" : "pointer",
              fontSize: "14px",
              opacity: loading || !eventLabel.trim() || !selectedVideoId ? 0.5 : 1,
            }}
          >
            {loading ? "處理中..." : currentEvent ? "更新標籤" : "設置標籤"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default EventTagModal;

