import React from "react";
import { EVENT_MAP } from "../utils/constants";
import apiService from "../services/api";

const RagResults = ({ data, apiKey }) => {
  if (!data) return null;

  const hits = data.hits || [];
  const backend = data.backend || {};
  const answer = data.answer || "";

  const handleDownload = async (path, filename) => {
    try {
      await apiService.downloadFile(path, filename, apiKey);
    } catch (e) {
      alert(`下載錯誤: ${e.message}`);
      console.error(e);
    }
  };

  if (hits.length === 0 && !answer) {
    return (
      <div style={{ textAlign: "center", padding: "20px", color: "#888" }}>
        未找到結果（可能皆低於相似度門檻）
      </div>
    );
  }

  return (
    <div>
      {answer && (
        <div
          style={{
            background: "#111827",
            border: "1px solid #374151",
            borderRadius: "4px",
            padding: "16px",
            marginBottom: "16px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "8px",
            }}
          >
            <div
              style={{ color: "#f9fafb", fontWeight: "600", fontSize: "15px" }}
            >
              💡 LLM 回答
            </div>
            <div style={{ color: "#6b7280", fontSize: "12px" }}>
              LLM: {backend.llm || "N/A"}　/　向量模型:{" "}
              {backend.embed_model || "N/A"}
            </div>
          </div>
          <div
            style={{
              color: "#e5e7eb",
              fontSize: "14px",
              lineHeight: "1.8",
              whiteSpace: "pre-wrap",
            }}
          >
            {answer}
          </div>
        </div>
      )}

      {!answer && (
        <div
          style={{
            background: "#222",
            padding: "8px",
            marginBottom: "12px",
            borderRadius: "4px",
            fontSize: "12px",
            color: "#aaa",
          }}
        >
          Backend: {backend.search_engine || "N/A"} | 找到 {hits.length} 筆
        </div>
      )}

      {hits.map((h, i) => {
        const events =
          (h.events_true || [])
            .map((e) => EVENT_MAP[e] || e)
            .join("、") || "無事件";
        const videoPath = h.video || "";
        const segment = h.segment || "";
        // 後端歷史資料有些 video 會帶路徑分隔（例如 人員追蹤_20260528/K8-22），
        // 但實際 segment 目錄使用底線命名（人員追蹤_20260528_K8-22）。
        const normalizedVideoPath = videoPath.replace(/\//g, "_");
        // 構建完整路徑：確保包含 /segment/ 前綴
        let fullVideoPath = "";
        if (videoPath && segment) {
          // 如果 videoPath 已經包含 /segment/，直接使用
          if (videoPath.startsWith("/segment/")) {
            fullVideoPath = `${videoPath}/${segment}`;
          } else {
            // 否則添加 /segment/ 前綴
            fullVideoPath = `/segment/${normalizedVideoPath}/${segment}`;
          }
        }

        if (answer) {
          return (
            <div
              key={i}
              style={{
                borderTop: "1px solid #333",
                padding: "8px 0",
                fontSize: "12px",
                color: "#888",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <div style={{ flex: 1 }}>
                <span style={{ color: "#666", fontSize: "11px", marginRight: "8px" }}>
                  相似度: {h.score ? (h.score * 100).toFixed(1) + "%" : "N/A"}
                </span>
                [{i + 1}] {h.time_range} - {h.summary}
              </div>
              {fullVideoPath && (
                <button
                  onClick={() => handleDownload(fullVideoPath, segment)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "#3b82f6",
                    cursor: "pointer",
                    marginLeft: "8px",
                  }}
                >
                  ⬇
                </button>
              )}
            </div>
          );
        }

        return (
          <div
            key={i}
            style={{
              background: "#1a1a1a",
              border: "1px solid #333",
              padding: "12px",
              marginBottom: "8px",
              borderRadius: "4px",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "4px",
              }}
            >
              <span style={{ color: "#fff", fontWeight: "bold" }}>
                #{i + 1}
              </span>
              <span style={{ color: "#888", fontSize: "12px" }}>
                Score: {(h.score * 100).toFixed(1)}%
              </span>
            </div>

            {fullVideoPath && (
              <div
                style={{
                  background: "#0a0a0a",
                  border: "1px solid #2a2a2a",
                  borderRadius: "4px",
                  padding: "8px",
                  marginBottom: "8px",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "12px",
                  }}
                >
                  <div
                    style={{
                      flex: 1,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    <span style={{ color: "#aaa", fontSize: "12px" }}>
                      {segment}
                    </span>
                  </div>

                  <button
                    onClick={() => handleDownload(fullVideoPath, segment)}
                    style={{
                      background: "#333",
                      color: "#fff",
                      padding: "4px 12px",
                      borderRadius: "4px",
                      textDecoration: "none",
                      fontSize: "12px",
                      whiteSpace: "nowrap",
                      border: "1px solid #555",
                      cursor: "pointer",
                    }}
                  >
                    ⬇ 下載
                  </button>
                </div>
              </div>
            )}

            <div
              style={{ color: "#ddd", fontSize: "14px", marginBottom: "4px" }}
            >
              <span style={{ color: "#888" }}>時間：</span>
              {h.time_range}{" "}
              <span style={{ marginLeft: "8px", color: "#888" }}>事件：</span>
              {events}
            </div>
            <div style={{ color: "#ccc", fontSize: "13px", lineHeight: "1.4" }}>
              {h.summary}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default RagResults;

