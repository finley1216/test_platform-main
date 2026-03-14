import React, { useState } from "react";
import apiService from "../services/api";

const ImageSearchResults = ({ data, apiKey }) => {
  const { query_type, query_info, total_results, results, threshold, label_filter, debug } = data || {};
  const [showVectorInfo, setShowVectorInfo] = useState(false);

  const getImageUrl = (cropPath, videoName) => {
    if (!cropPath) return null;
    // 如果路徑是絕對路徑或 URL，直接返回
    if (cropPath.startsWith("http")) {
      return cropPath;
    }
    // 靜態檔案在後端的 /segment 下，目錄結構為 /segment/{video}/yolo_output/object_crops/xxx.jpg
    // video 為影片名稱（如 車輛追蹤_K8-008），不是 segment 檔名（segment_003.mp4）
    const baseUrl = apiService.baseUrl.replace(/\/$/, "");
    const segmentBaseUrl = baseUrl.replace(/\/api\/?$/, "") || baseUrl;
    // 如果路徑已經以 segment/ 開頭（完整相對路徑），直接接在 origin 後
    if (cropPath.startsWith("segment/")) {
      return `${segmentBaseUrl}/${cropPath}`;
    }
    // 路徑是相對影片目錄的（例如 "yolo_output/object_crops/xxx.jpg"），需加上 video 名稱作為父目錄
    if (videoName && (cropPath.includes("yolo_output") || cropPath.includes("object_crops"))) {
      const path = cropPath.startsWith("/") ? cropPath.slice(1) : cropPath;
      return `${segmentBaseUrl}/segment/${encodeURIComponent(videoName)}/${path}`;
    }
    // 若已有 /segment 開頭（絕對路徑形式）
    if (cropPath.startsWith("/segment")) {
      return `${segmentBaseUrl}${cropPath}`;
    }
    // 其餘：當作相對 origin 的路徑
    const normalizedPath = cropPath.startsWith("/") ? cropPath : `/${cropPath}`;
    return `${segmentBaseUrl}${normalizedPath}`;
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return "N/A";
    const minutes = Math.floor(timestamp / 60);
    const seconds = Math.floor(timestamp % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const getSimilarityColor = (similarity) => {
    if (similarity >= 0.9) return "#10b981"; // 綠色
    if (similarity >= 0.8) return "#3b82f6"; // 藍色
    if (similarity >= 0.7) return "#f59e0b"; // 橙色
    return "#ef4444"; // 紅色
  };

  if (!data || !results) {
    return <div style={{ color: "#888" }}>無搜尋結果</div>;
  }

  // 計算相似度（如果兩個向量都存在）
  const calculateSimilarity = (vec1, vec2) => {
    if (!vec1 || !vec2 || vec1.length !== vec2.length) return null;
    try {
      const dotProduct = vec1.reduce((sum, a, i) => sum + a * vec2[i], 0);
      const norm1 = Math.sqrt(vec1.reduce((sum, a) => sum + a * a, 0));
      const norm2 = Math.sqrt(vec2.reduce((sum, a) => sum + a * a, 0));
      return dotProduct / (norm1 * norm2 + 1e-12);
    } catch (e) {
      return null;
    }
  };

  const querySimilarity = debug?.query_embedding && debug?.first_crop_embedding
    ? calculateSimilarity(debug.query_embedding, debug.first_crop_embedding)
    : null;

  return (
    <div style={{ padding: "16px 0" }}>
      {/* 搜尋資訊 */}
      <div
        style={{
          marginBottom: "20px",
          padding: "12px",
          background: "var(--gray-50)",
          borderRadius: "8px",
          border: "1px solid var(--gray-200)",
        }}
      >
        <div style={{ display: "flex", gap: "16px", flexWrap: "wrap", alignItems: "center" }}>
          <div>
            <strong>查詢類型:</strong> {query_type === "image" ? "圖片" : "文字"}
          </div>
          {query_type === "image" && query_info.filename && (
            <div>
              <strong>檔案:</strong> {query_info.filename}
            </div>
          )}
          {query_type === "text" && query_info.text && (
            <div>
              <strong>描述:</strong> "{query_info.text}"
            </div>
          )}
          <div>
            <strong>找到結果:</strong> {total_results} 筆
          </div>
          <div>
            <strong>相似度門檻:</strong> {(threshold * 100).toFixed(0)}%
          </div>
          {label_filter && (
            <div>
              <strong>類別過濾:</strong> {label_filter}
            </div>
          )}
          {debug && (
            <div>
              <button
                onClick={() => setShowVectorInfo(!showVectorInfo)}
                style={{
                  padding: "6px 12px",
                  background: showVectorInfo ? "var(--primary)" : "var(--gray-200)",
                  color: showVectorInfo ? "white" : "var(--gray-700)",
                  border: "none",
                  borderRadius: "4px",
                  cursor: "pointer",
                  fontSize: "13px",
                  fontWeight: "500",
                }}
              >
                {showVectorInfo ? "隱藏" : "顯示"}向量信息
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 向量信息顯示 */}
      {debug && showVectorInfo && (
        <div
          style={{
            marginBottom: "20px",
            padding: "16px",
            background: "var(--white)",
            borderRadius: "8px",
            border: "1px solid var(--gray-300)",
            boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
          }}
        >
          <h3 style={{ marginTop: 0, marginBottom: "16px", fontSize: "16px", fontWeight: "600" }}>
            📊 向量信息
          </h3>
          
          {/* 查詢向量 */}
          <div style={{ marginBottom: "20px" }}>
            <h4 style={{ marginTop: 0, marginBottom: "8px", fontSize: "14px", fontWeight: "600", color: "var(--primary)" }}>
              查詢向量 (Query Embedding)
            </h4>
            <div style={{ fontSize: "13px", color: "#666", marginBottom: "8px" }}>
              <div><strong>維度:</strong> {debug.query_embedding_dim || "N/A"}</div>
              {debug.query_embedding_sample && (
                <div style={{ marginTop: "4px" }}>
                  <strong>前10個值:</strong>
                  <div
                    style={{
                      marginTop: "4px",
                      padding: "8px",
                      background: "var(--gray-50)",
                      borderRadius: "4px",
                      fontFamily: "monospace",
                      fontSize: "12px",
                      wordBreak: "break-all",
                    }}
                  >
                    [{debug.query_embedding_sample.map((v, i) => (
                      <span key={i}>
                        {v.toFixed(6)}
                        {i < debug.query_embedding_sample.length - 1 ? ", " : ""}
                      </span>
                    ))}]
                  </div>
                </div>
              )}
              {debug.query_embedding && (
                <div style={{ marginTop: "8px" }}>
                  <details>
                    <summary style={{ cursor: "pointer", color: "var(--primary)", fontSize: "12px" }}>
                      查看完整向量 ({debug.query_embedding.length} 維)
                    </summary>
                    <div
                      style={{
                        marginTop: "8px",
                        padding: "8px",
                        background: "var(--gray-50)",
                        borderRadius: "4px",
                        fontFamily: "monospace",
                        fontSize: "11px",
                        maxHeight: "200px",
                        overflow: "auto",
                        wordBreak: "break-all",
                      }}
                    >
                      [{debug.query_embedding.map((v, i) => (
                        <span key={i}>
                          {v.toFixed(6)}
                          {i < debug.query_embedding.length - 1 ? ", " : ""}
                        </span>
                      ))}]
                    </div>
                  </details>
                </div>
              )}
            </div>
          </div>

          {/* 第一筆資料向量 */}
          {debug.first_crop_info && (
            <div style={{ marginBottom: "20px", paddingTop: "16px", borderTop: "1px solid var(--gray-200)" }}>
              <h4 style={{ marginTop: 0, marginBottom: "8px", fontSize: "14px", fontWeight: "600", color: "var(--primary)" }}>
                第一筆資料向量 (First Crop Embedding)
              </h4>
              <div style={{ fontSize: "13px", color: "#666", marginBottom: "8px" }}>
                <div><strong>ID:</strong> {debug.first_crop_info.id || "N/A"}</div>
                <div><strong>類別:</strong> {debug.first_crop_info.label || "N/A"}</div>
                <div><strong>路徑:</strong> {debug.first_crop_info.crop_path || "N/A"}</div>
                <div style={{ marginTop: "8px" }}><strong>向量維度:</strong> {debug.first_crop_embedding_dim || "N/A"}</div>
                {debug.first_crop_embedding_sample && (
                  <div style={{ marginTop: "4px" }}>
                    <strong>前10個值:</strong>
                    <div
                      style={{
                        marginTop: "4px",
                        padding: "8px",
                        background: "var(--gray-50)",
                        borderRadius: "4px",
                        fontFamily: "monospace",
                        fontSize: "12px",
                        wordBreak: "break-all",
                      }}
                    >
                      [{debug.first_crop_embedding_sample.map((v, i) => (
                        <span key={i}>
                          {v.toFixed(6)}
                          {i < debug.first_crop_embedding_sample.length - 1 ? ", " : ""}
                        </span>
                      ))}]
                    </div>
                  </div>
                )}
                {debug.first_crop_embedding && (
                  <div style={{ marginTop: "8px" }}>
                    <details>
                      <summary style={{ cursor: "pointer", color: "var(--primary)", fontSize: "12px" }}>
                        查看完整向量 ({debug.first_crop_embedding.length} 維)
                      </summary>
                      <div
                        style={{
                          marginTop: "8px",
                          padding: "8px",
                          background: "var(--gray-50)",
                          borderRadius: "4px",
                          fontFamily: "monospace",
                          fontSize: "11px",
                          maxHeight: "200px",
                          overflow: "auto",
                          wordBreak: "break-all",
                        }}
                      >
                        [{debug.first_crop_embedding.map((v, i) => (
                          <span key={i}>
                            {v.toFixed(6)}
                            {i < debug.first_crop_embedding.length - 1 ? ", " : ""}
                          </span>
                        ))}]
                      </div>
                    </details>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 相似度計算 */}
          {querySimilarity !== null && (
            <div style={{ paddingTop: "16px", borderTop: "1px solid var(--gray-200)" }}>
              <h4 style={{ marginTop: 0, marginBottom: "8px", fontSize: "14px", fontWeight: "600", color: "var(--primary)" }}>
                相似度計算
              </h4>
              <div style={{ fontSize: "13px", color: "#666" }}>
                <div>
                  <strong>查詢向量與第一筆資料的相似度:</strong>
                  <span
                    style={{
                      marginLeft: "8px",
                      padding: "4px 8px",
                      background: getSimilarityColor(querySimilarity),
                      color: "white",
                      borderRadius: "4px",
                      fontWeight: "600",
                    }}
                  >
                    {(querySimilarity * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* 結果列表 */}
      {results.length === 0 ? (
        <div style={{ color: "#888", textAlign: "center", padding: "40px" }}>
          沒有找到符合條件的結果
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
            gap: "16px",
          }}
        >
          {results.map((result, index) => {
            const imageUrl = getImageUrl(result.crop_path, result.video);
            return (
              <div
                key={result.crop_id || index}
                style={{
                  border: "1px solid var(--gray-300)",
                  borderRadius: "8px",
                  padding: "12px",
                  background: "var(--white)",
                  transition: "var(--transition)",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow = "var(--shadow-md)";
                  e.currentTarget.style.transform = "translateY(-2px)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = "none";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                {/* 圖片 */}
                {imageUrl ? (
                  <div style={{ marginBottom: "12px", textAlign: "center" }}>
                    <img
                      src={imageUrl}
                      alt={`Crop ${result.crop_id}`}
                      style={{
                        maxWidth: "100%",
                        maxHeight: "200px",
                        borderRadius: "4px",
                        border: "1px solid var(--gray-200)",
                        objectFit: "contain",
                      }}
                      onError={(e) => {
                        e.target.style.display = "none";
                        e.target.nextSibling.style.display = "block";
                      }}
                    />
                    <div
                      style={{
                        display: "none",
                        padding: "40px",
                        color: "#888",
                        fontSize: "14px",
                      }}
                    >
                      圖片載入失敗
                    </div>
                  </div>
                ) : (
                  <div
                    style={{
                      padding: "40px",
                      textAlign: "center",
                      color: "#888",
                      background: "var(--gray-50)",
                      borderRadius: "4px",
                      marginBottom: "12px",
                    }}
                  >
                    無圖片路徑
                  </div>
                )}

                {/* 相似度 */}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "8px",
                  }}
                >
                  <span style={{ fontSize: "12px", color: "#666" }}>相似度</span>
                  <span
                    style={{
                      fontSize: "16px",
                      fontWeight: "bold",
                      color: getSimilarityColor(result.similarity),
                    }}
                  >
                    {(result.similarity * 100).toFixed(1)}%
                  </span>
                </div>

                {/* 物件資訊 */}
                <div style={{ fontSize: "13px", color: "#666", marginBottom: "8px" }}>
                  <div>
                    <strong>類別:</strong> {result.label || "N/A"}
                  </div>
                  <div>
                    <strong>信心分數:</strong>{" "}
                    {result.score ? (result.score * 100).toFixed(1) + "%" : "N/A"}
                  </div>
                  <div>
                    <strong>時間戳:</strong> {formatTimestamp(result.timestamp)}
                  </div>
                  {result.frame !== null && result.frame !== undefined && (
                    <div>
                      <strong>幀號:</strong> {result.frame}
                    </div>
                  )}
                </div>

                {/* 影片資訊 */}
                <div
                  style={{
                    marginTop: "8px",
                    paddingTop: "8px",
                    borderTop: "1px solid var(--gray-200)",
                    fontSize: "12px",
                    color: "#888",
                  }}
                >
                  <div>
                    <strong>影片:</strong> {result.video || "N/A"}
                  </div>
                  <div>
                    <strong>片段:</strong> {result.segment || "N/A"}
                  </div>
                  {result.time_range && (
                    <div>
                      <strong>時間範圍:</strong> {result.time_range}
                    </div>
                  )}
                  {result.location && (
                    <div>
                      <strong>位置:</strong> {result.location}
                    </div>
                  )}
                  {result.camera && (
                    <div>
                      <strong>攝影機:</strong> {result.camera}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ImageSearchResults;

