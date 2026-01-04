import React, { useState, useEffect } from "react";
import apiService from "../services/api";

const VideoSelector = ({
  selectedVideoId,
  onVideoChange,
  apiKey,
  authenticated,
}) => {
  const [videos, setVideos] = useState([]);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState("all");

  // 預定義的事件類型（與 EventTagModal 保持一致）
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
    if (authenticated && apiKey) {
      loadVideos();
    }
  }, [authenticated, apiKey]);

  const loadVideos = async () => {
    setLoading(true);
    try {
      const data = await apiService.listVideos(apiKey);
      setVideos(data.videos || []);
      // 使用預定義的事件類型列表，而不是後端返回的分類
      try {
        const catData = await apiService.getVideoCategories(apiKey);
        if (catData.categories) {
          setCategories(catData.categories);
        } else {
          // 如果後端沒有返回，使用預定義列表
          setCategories(eventTypes);
        }
      } catch (e) {
        // 如果 API 調用失敗，使用預定義列表
        console.warn("Failed to load categories, using predefined list:", e);
        setCategories(eventTypes);
      }
    } catch (error) {
      console.error("Failed to load videos:", error);
      setVideos([]);
      setCategories(eventTypes); // 使用預定義列表作為備用
    } finally {
      setLoading(false);
    }
  };

  // 按事件類型分組影片（只使用 event_label，不使用 category）
  // 只顯示有實際影片的事件類型
  const groupedVideos = videos.reduce((acc, video) => {
    // 只處理有 event_label 的影片，沒有 event_label 的歸類為"未分類"
    if (video.event_label) {
      const eventType = video.event_label;
      if (!acc[eventType]) {
        acc[eventType] = [];
      }
      acc[eventType].push(video);
    } else {
      // 沒有 event_label 的影片歸類為"未分類"
      if (!acc["未分類"]) {
        acc["未分類"] = [];
      }
      acc["未分類"].push(video);
    }
    return acc;
  }, {});

  // 過濾影片（只使用 event_label）
  const filteredVideos =
    selectedCategory === "all"
      ? videos
      : videos.filter(
          (v) =>
            (v.event_label || "未分類") === selectedCategory
        );

  // 當選擇事件類型改變時，清除已選擇的影片
  useEffect(() => {
    if (selectedCategory !== "all" && selectedVideoId) {
      const selectedVideo = videos.find((v) => v.video_id === selectedVideoId);
      if (selectedVideo && (selectedVideo.event_label || "未分類") !== selectedCategory) {
        onVideoChange("");
      }
    }
  }, [selectedCategory, selectedVideoId, videos, onVideoChange]);

  return (
    <div className="form-group" style={{ overflow: "visible" }}>
      <label className="form-label">選擇已上傳的影片</label>
      {loading ? (
        <div className="form-select" style={{ color: "#999" }}>
          載入中...
        </div>
      ) : videos.length === 0 ? (
        <div className="form-select" style={{ color: "#999" }}>
          尚無已上傳的影片
        </div>
      ) : (
        <>
          {/* 分類篩選 */}
          {categories.length > 0 && (
            <div style={{ marginBottom: "12px" }}>
              <select
                className="form-select"
                value={selectedCategory}
                onChange={(e) => {
                  setSelectedCategory(e.target.value);
                  // 切換分類時清除已選擇的影片
                  onVideoChange("");
                }}
                style={{ marginBottom: "8px" }}
              >
                <option value="all">所有事件類型</option>
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* 影片選擇 */}
          <select
            className="form-select"
            value={selectedVideoId || ""}
            onChange={(e) => onVideoChange(e.target.value)}
            style={{ marginBottom: "8px" }}
          >
            <option value="">-- 請選擇影片 --</option>
            {selectedCategory === "all" ? (
              // 顯示所有事件類型，使用 optgroup 分組（只顯示有實際影片的事件類型）
              Object.entries(groupedVideos)
                .filter(([eventType, categoryVideos]) => categoryVideos.length > 0)  // 只顯示有影片的事件類型
                .map(([eventType, categoryVideos]) => (
                <optgroup key={eventType} label={eventType}>
                  {categoryVideos.map((video) => (
                    <option key={video.video_id} value={video.video_id}>
                      {video.display_name}
                      {video.source === "segment" &&
                        ` (${video.success_segments}/${video.total_segments} 片段)`}
                    </option>
                  ))}
                </optgroup>
              ))
            ) : (
              // 只顯示選中分類的影片，不使用 optgroup
              filteredVideos.map((video) => (
                <option key={video.video_id} value={video.video_id}>
                  {video.display_name}
                  {video.source === "segment" &&
                    ` (${video.success_segments}/${video.total_segments} 片段)`}
                </option>
              ))
            )}
          </select>

          {/* 顯示選中影片的詳細信息 */}
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
                    {/* 只顯示事件類型，不顯示分類 */}
                    {selectedVideo.event_label && (
                      <div style={{ color: "#999", marginBottom: "4px", wordBreak: "break-word" }}>
                        事件類型：{selectedVideo.event_label}
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
                        <div style={{ color: "#888", fontSize: "12px", marginBottom: "4px" }}>
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
                  </>
                );
              })()}
            </div>
          )}

          <button
            type="button"
            onClick={loadVideos}
            style={{
              marginTop: "8px",
              padding: "6px 12px",
              fontSize: "12px",
              background: "#2a2a2a",
              border: "1px solid #444",
              borderRadius: "4px",
              color: "#ccc",
              cursor: "pointer",
            }}
          >
            重新載入
          </button>
        </>
      )}
    </div>
  );
};

export default VideoSelector;

