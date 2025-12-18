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
      setCategories(data.categories || []);
    } catch (error) {
      console.error("Failed to load videos:", error);
      setVideos([]);
      setCategories([]);
    } finally {
      setLoading(false);
    }
  };

  // 按分類分組影片
  const groupedVideos = videos.reduce((acc, video) => {
    const category = video.category || video.event_label || "未分類";
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(video);
    return acc;
  }, {});

  // 過濾影片
  const filteredVideos =
    selectedCategory === "all"
      ? videos
      : videos.filter(
          (v) =>
            (v.category || v.event_label || "未分類") === selectedCategory
        );

  // 當選擇分類改變時，清除已選擇的影片
  useEffect(() => {
    if (selectedCategory !== "all" && selectedVideoId) {
      const selectedVideo = videos.find((v) => v.video_id === selectedVideoId);
      if (selectedVideo && (selectedVideo.category || selectedVideo.event_label || "未分類") !== selectedCategory) {
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
                <option value="all">所有分類</option>
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
              // 顯示所有分類，使用 optgroup 分組
              Object.entries(groupedVideos).map(([category, categoryVideos]) => (
                <optgroup key={category} label={category}>
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

