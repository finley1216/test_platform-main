import React from "react";
import RagResults from "./RagResults";

const RAGSearch = ({
  ragQuery,
  ragTopK,
  ragThreshold,
  ragStats,
  isSearching,
  ragData,
  ragError,
  apiKey,
  onQueryChange,
  onTopKChange,
  onThresholdChange,
  onSearch,
  onAnswer,
}) => {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span className="card-title-icon">🧭</span>
          <span>影片智慧搜尋</span>
          <span
            className="badge badge-secondary"
            style={{ marginLeft: "8px" }}
          >
            索引 {ragStats}
          </span>
        </div>
      </div>
      <div className="form-grid">
        <div className="form-group">
          <label className="form-label">查詢關鍵字</label>
          <input
            className="form-input"
            value={ragQuery}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && onSearch()}
          />
        </div>

        <div
          className="form-grid"
          style={{
            gridTemplateColumns: "1fr 1fr",
            gap: "16px",
            marginTop: 0,
          }}
        >
          <div className="form-group">
            <label className="form-label">搜尋數量上限 (Top K)</label>
            <input
              type="number"
              min="1"
              max="50"
              className="form-input"
              value={ragTopK}
              onChange={(e) => onTopKChange(e.target.value)}
            />
          </div>
          <div className="form-group">
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                marginBottom: "4px",
              }}
            >
              <label className="form-label">相似度門檻 (Min Score)</label>
              <span
                style={{
                  color: "#4ade80",
                  fontWeight: "bold",
                  fontSize: "14px",
                }}
              >
                {(ragThreshold * 100).toFixed(0)}%
              </span>
            </div>
            <div
              style={{
                display: "flex",
                gap: "8px",
                alignItems: "center",
              }}
            >
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={ragThreshold}
                onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                style={{
                  flex: 1,
                  accentColor: "#059669",
                  cursor: "pointer",
                }}
              />
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                className="form-input"
                style={{ width: "70px", textAlign: "center" }}
                value={ragThreshold}
                onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
              />
            </div>
          </div>
        </div>
      </div>
      <div className="btn-group" style={{ margin: "8px 0 12px" }}>
        <button
          onClick={onSearch}
          className="btn btn-primary"
          disabled={isSearching}
        >
          {isSearching ? "搜尋中..." : "搜尋"}
        </button>
        <button
          onClick={onAnswer}
          className="btn btn-secondary"
          style={{ marginLeft: "8px" }}
          disabled={isSearching}
        >
          {isSearching ? "思考中..." : "搜尋 + LLM 回答"}
        </button>
      </div>
      <div className="output-section">
        <div className="output-header">
          <h3 className="output-title">RAG 查詢結果</h3>
        </div>
        <div className="output-panel">
          {isSearching && (
            <div className="status-message info">🔍 搜尋/生成中...</div>
          )}
          {ragError && (
            <div style={{ color: "#ef4444" }}>Error: {ragError}</div>
          )}
          {!isSearching && !ragError && !ragData && (
            <div style={{ color: "#888" }}>尚未查詢</div>
          )}
          {!isSearching && ragData && (
            <RagResults data={ragData} apiKey={apiKey} />
          )}
        </div>
      </div>
    </div>
  );
};

export default RAGSearch;

