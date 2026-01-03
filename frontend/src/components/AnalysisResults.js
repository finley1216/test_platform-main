import React, { useState, useEffect, useMemo, useRef } from "react";
import { EVENT_MAP, getApiBaseUrl } from "../utils/constants";
import apiService from "../services/api";

const AnalysisResults = ({ data, apiKey, authenticated }) => {
  const [showJson, setShowJson] = useState(false);
  const [videoEvent, setVideoEvent] = useState(null);
  const hasLoadedRef = useRef(false); // 用於追蹤是否已經載入過

  // 從 save_path 提取 video_id，並嘗試還原為 video_lib 格式
  // 使用 useMemo 避免每次渲染都創建新對象
  const videoIdInfo = useMemo(() => {
    if (!data?.save_path) return null;
    const parts = data.save_path.split("/");
    const segmentIndex = parts.indexOf("segment");
    if (segmentIndex >= 0 && segmentIndex + 1 < parts.length) {
      const segmentId = parts[segmentIndex + 1];
      
      // 檢查是否為 video_lib 影片的處理結果（格式：{category}_{video_name}）
      if (segmentId.includes("_")) {
        const underscoreIndex = segmentId.indexOf("_");
        const potentialCategory = segmentId.substring(0, underscoreIndex);
        const potentialVideoName = segmentId.substring(underscoreIndex + 1);
        
        // 嘗試還原為 video_lib 格式：category/video_name
        // 先嘗試用還原後的格式，如果失敗再用原始 segment ID
        return {
          videoLibId: `${potentialCategory}/${potentialVideoName}`,
          segmentId: segmentId,
        };
      }
      
      return {
        segmentId: segmentId,
      };
    }
    return null;
  }, [data?.save_path]); // 只有當 save_path 改變時才重新計算

  // 載入影片事件標籤
  useEffect(() => {
    if (!videoIdInfo || !authenticated || !apiKey) return;
    
    // 如果已經載入過，不再重複載入
    const cacheKey = `${videoIdInfo.videoLibId || videoIdInfo.segmentId}-${apiKey}`;
    if (hasLoadedRef.current === cacheKey) return;
    
    // 優先嘗試 video_lib 格式，如果失敗再嘗試 segment ID
    const tryLoadEvent = async (videoId) => {
      try {
        const info = await apiService.getVideoInfo(videoId, apiKey);
        if (info.event_label) {
          setVideoEvent({
            label: info.event_label,
            description: info.event_description || "",
          });
          hasLoadedRef.current = cacheKey; // 標記為已載入
          return true; // 成功載入
        }
      } catch (err) {
        console.error(`Failed to load video event for ${videoId}:`, err);
      }
      return false; // 載入失敗
    };

    const loadEvent = async () => {
      // 如果有 video_lib ID，先嘗試它
      if (videoIdInfo.videoLibId) {
        const success = await tryLoadEvent(videoIdInfo.videoLibId);
        if (success) return;
      }
      
      // 如果 video_lib ID 失敗或不存在，嘗試 segment ID
      if (videoIdInfo.segmentId) {
        await tryLoadEvent(videoIdInfo.segmentId);
        hasLoadedRef.current = cacheKey; // 即使失敗也標記，避免無限重試
      }
    };

    loadEvent();
  }, [videoIdInfo, authenticated, apiKey]);

  // Early returns 必須在所有 Hooks 之後
  if (!data) return null;
  if (data.error) {
    return (
      <div className="output-panel" style={{ color: "#ef4444" }}>
        Error: {JSON.stringify(data, null, 2)}
      </div>
    );
  }

  const results = data.results || [];
  const totalSegments = data.total_segments || 0;
  const totalTime = data.total_time_sec || 0;
  const modelType = data.model_type || "";
  
  // 提取 YOLO 輸出信息（從第一個結果的 raw_detection 中）
  const yoloOutput = results.length > 0 && results[0].raw_detection?.output 
    ? results[0].raw_detection.output 
    : null;

  let abnormalCount = 0;
  const anomalies = [];

  results.forEach((item) => {
    if (!item.parsed || !item.parsed.frame_analysis) return;
    const events = item.parsed.frame_analysis.events || {};
    const trueEvents = Object.keys(events).filter((k) => events[k] === true);

    if (trueEvents.length > 0) {
      abnormalCount++;
      anomalies.push({
        segment: item.segment,
        time: item.time_range,
        events: trueEvents.map((k) => EVENT_MAP[k] || k),
        reason: events.reason || "",
        summary: item.parsed.summary_independent || "",
      });
    }
  });

  return (
    <div>
      {videoEvent && (
        <div
          style={{
            padding: "16px 20px",
            background: "#2a2a2a",
            borderRadius: "8px",
            marginBottom: "16px",
            border: "1px solid #555",
            boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
          }}
        >
          <div style={{ 
            display: "flex", 
            alignItems: "center", 
            gap: "10px", 
            marginBottom: "12px",
            paddingBottom: "12px",
            borderBottom: "1px solid #444",
          }}>
            <span style={{ 
              fontSize: "14px", 
              fontWeight: "600",
              color: "#888",
              letterSpacing: "0.5px",
            }}>
              管理者標記
            </span>
            <span style={{ 
              fontSize: "16px", 
              fontWeight: "600",
              color: "#fff",
              letterSpacing: "0.3px",
            }}>
              {videoEvent.label}
            </span>
          </div>
          <div style={{ 
            marginBottom: "12px",
          }}>
            {videoEvent.description ? (
              <>
                <div style={{ 
                  fontSize: "12px",
                  color: "#888",
                  marginBottom: "6px",
                  fontWeight: "500",
                }}>
                  事件描述
                </div>
                <div style={{ 
                  color: "#ccc", 
                  fontSize: "14px",
                  lineHeight: "1.6",
                  paddingLeft: "2px",
                }}>
                  {videoEvent.description}
                </div>
              </>
            ) : (
              <div style={{ 
                fontSize: "12px",
                color: "#666",
                fontStyle: "italic",
              }}>
                無事件描述
              </div>
            )}
          </div>
          <div style={{ 
            marginTop: "12px",
            paddingTop: "12px",
            borderTop: "1px solid #444",
            fontSize: "12px", 
            color: "#999",
            lineHeight: "1.5",
          }}>
            提示：檢查下方模型判斷結果是否與標記事件一致
          </div>
        </div>
      )}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3,1fr)",
          gap: "12px",
          marginBottom: "24px",
        }}
      >
        <div
          style={{
            background: "#1a1a1a",
            padding: "16px",
            borderRadius: "8px",
            border: "1px solid #333",
            textAlign: "center",
          }}
        >
          <div
            style={{
              color: "#888",
              fontSize: "12px",
              textTransform: "uppercase",
            }}
          >
            總分析時長
          </div>
          <div style={{ color: "#fff", fontSize: "24px", fontWeight: "700" }}>
            {totalTime}s
          </div>
        </div>
        <div
          style={{
            background: "#1a1a1a",
            padding: "16px",
            borderRadius: "8px",
            border: "1px solid #333",
            textAlign: "center",
          }}
        >
          <div
            style={{
              color: "#888",
              fontSize: "12px",
              textTransform: "uppercase",
            }}
          >
            總片段數
          </div>
          <div style={{ color: "#fff", fontSize: "24px", fontWeight: "700" }}>
            {totalSegments}
          </div>
        </div>
        <div
          style={{
            background: abnormalCount > 0 ? "#450a0a" : "#064e3b",
            padding: "16px",
            borderRadius: "8px",
            border: `1px solid ${abnormalCount > 0 ? "#991b1b" : "#059669"}`,
            textAlign: "center",
          }}
        >
          <div
            style={{
              color: abnormalCount > 0 ? "#fecaca" : "#a7f3d0",
              fontSize: "12px",
              textTransform: "uppercase",
            }}
          >
            偵測異常
          </div>
          <div style={{ color: "#fff", fontSize: "24px", fontWeight: "700" }}>
            {abnormalCount}
          </div>
        </div>
      </div>

      {/* YOLO 標註影片顯示 */}
      {modelType === "yolo" && yoloOutput && yoloOutput.annotated_video && (
        <div
          style={{
            background: "#1a1a1a",
            padding: "20px",
            borderRadius: "8px",
            border: "1px solid #333",
            marginBottom: "24px",
          }}
        >
          <h4
            style={{
              color: "#fff",
              marginBottom: "16px",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            🎯 YOLO 物件偵測結果
          </h4>
          
          {/* 標註影片 */}
          <div style={{ marginBottom: "16px" }}>
            <div
              style={{
                color: "#888",
                fontSize: "12px",
                marginBottom: "8px",
                textTransform: "uppercase",
              }}
            >
              標註影片（含辨識框）
            </div>
            <video
              controls
              style={{
                width: "100%",
                maxWidth: "800px",
                borderRadius: "4px",
                background: "#000",
              }}
            >
              <source
                src={`${getApiBaseUrl()}/${yoloOutput.annotated_video}`}
                type="video/mp4"
              />
              您的瀏覽器不支援影片播放
            </video>
          </div>

          {/* 物件統計 */}
          {yoloOutput.object_count && Object.keys(yoloOutput.object_count).length > 0 && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
                gap: "12px",
                marginBottom: "16px",
              }}
            >
              {Object.entries(yoloOutput.object_count).map(([label, count]) => (
                <div
                  key={label}
                  style={{
                    background: "#2a2a2a",
                    padding: "12px",
                    borderRadius: "6px",
                    border: "1px solid #444",
                    textAlign: "center",
                  }}
                >
                  <div style={{ color: "#888", fontSize: "11px", marginBottom: "4px" }}>
                    {label}
                  </div>
                  <div style={{ color: "#fff", fontSize: "20px", fontWeight: "700" }}>
                    {count}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* 物件切片和下載連結 */}
          <div
            style={{
              display: "flex",
              gap: "12px",
              flexWrap: "wrap",
              marginTop: "16px",
            }}
          >
            {yoloOutput.object_crops_dir && (
              <a
                href={`${getApiBaseUrl()}/${yoloOutput.object_crops_dir}`}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  background: "#3b82f6",
                  color: "#fff",
                  padding: "8px 16px",
                  borderRadius: "6px",
                  textDecoration: "none",
                  fontSize: "13px",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                📁 查看物件切片
              </a>
            )}
            {yoloOutput.timestamp_file && (
              <a
                href={`${getApiBaseUrl()}/${yoloOutput.timestamp_file}`}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  background: "#10b981",
                  color: "#fff",
                  padding: "8px 16px",
                  borderRadius: "6px",
                  textDecoration: "none",
                  fontSize: "13px",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                }}
              >
                ⏱️ 查看時間戳記錄
              </a>
            )}
          </div>
        </div>
      )}

      {anomalies.length > 0 ? (
        <>
          <h4
            style={{
              color: "#fff",
              marginBottom: "12px",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            ⚠️ 異常事件列表{" "}
            <span
              className="badge badge-primary"
              style={{ background: "#dc2626" }}
            >
              {anomalies.length}
            </span>
          </h4>
          {anomalies.map((a, idx) => (
            <div
              key={idx}
              style={{
                background: "#2a1215",
                borderLeft: "4px solid #ef4444",
                padding: "16px",
                marginBottom: "12px",
                borderRadius: "0 8px 8px 0",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  marginBottom: "8px",
                }}
              >
                <span
                  style={{ color: "#fff", fontWeight: "700", fontSize: "16px" }}
                >
                  {a.time}
                </span>
                <span
                  style={{
                    background: "#ef4444",
                    color: "white",
                    padding: "2px 8px",
                    borderRadius: "4px",
                    fontSize: "12px",
                    fontWeight: "bold",
                  }}
                >
                  {a.events.join("、")}
                </span>
              </div>
              <div
                style={{
                  color: "#fca5a5",
                  fontSize: "14px",
                  marginBottom: "4px",
                }}
              >
                <strong>證據：</strong>
                {a.reason}
              </div>
              <div style={{ color: "#ccc", fontSize: "13px" }}>
                <strong>摘要：</strong>
                {a.summary}
              </div>
            </div>
          ))}
        </>
      ) : (
        <div
          style={{
            background: "#064e3b",
            color: "#d1fae5",
            padding: "16px",
            borderRadius: "8px",
            marginBottom: "24px",
            textAlign: "center",
            border: "1px solid #059669",
          }}
        >
          🛡️ 全程安全，未發現異常事件
        </div>
      )}

      <details
        style={{
          marginTop: "24px",
          background: "#171717",
          borderRadius: "8px",
          border: "1px solid #333",
        }}
      >
        <summary
          style={{
            padding: "16px",
            cursor: "pointer",
            color: "#fff",
            fontWeight: "600",
          }}
        >
          📄 查看所有片段詳情 ({results.length})
        </summary>
        <div style={{ padding: "0 16px 16px 16px" }}>
          {results.map((item, idx) => {
            const events = item.parsed?.frame_analysis?.events || {};
            const isSafe = Object.values(events).every((v) => v !== true);
            const summary = item.parsed?.summary_independent || "無摘要";
            const persons = item.parsed?.frame_analysis?.persons || [];

            return (
              <div
                key={idx}
                style={{
                  borderTop: "1px solid #333",
                  padding: "12px 0",
                  display: "flex",
                  gap: "16px",
                  alignItems: "start",
                }}
              >
                <div
                  style={{
                    minWidth: "120px",
                    fontFamily: "monospace",
                    color: "#888",
                    fontSize: "13px",
                  }}
                >
                  {item.time_range}
                </div>
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      marginBottom: "4px",
                    }}
                  >
                    <span
                      style={{
                        width: "8px",
                        height: "8px",
                        borderRadius: "50%",
                        display: "inline-block",
                        marginRight: "8px",
                        background: isSafe ? "#10b981" : "#ef4444",
                      }}
                    ></span>
                    <span
                      style={{
                        color: isSafe ? "#d1d5db" : "#fca5a5",
                        fontWeight: "600",
                        fontSize: "14px",
                      }}
                    >
                      {isSafe ? "正常" : "⚠️ 異常檢出"}
                    </span>
                    {persons.length > 0 && (
                      <span
                        style={{
                          background: "#374151",
                          color: "#e5e7eb",
                          padding: "2px 6px",
                          borderRadius: "4px",
                          fontSize: "11px",
                          marginLeft: "8px",
                        }}
                      >
                        👤 {persons.length} 人
                      </span>
                    )}
                  </div>
                  <div
                    style={{
                      color: "#9ca3af",
                      fontSize: "13px",
                      lineHeight: "1.5",
                    }}
                  >
                    {summary}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </details>

      <div style={{ marginTop: "16px", textAlign: "right" }}>
        <button
          className="btn btn-ghost"
          onClick={() => setShowJson(!showJson)}
          style={{ fontSize: "12px" }}
        >
          {showJson ? "隱藏原始 JSON" : "顯示原始 JSON"}
        </button>
        {showJson && (
          <div
            className="output-panel"
            style={{
              marginTop: "8px",
              textAlign: "left",
              whiteSpace: "pre-wrap",
            }}
          >
            {JSON.stringify(data, null, 2)}
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisResults;

