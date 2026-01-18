import React from "react";

const Navbar = ({ authenticated, isAdmin, onEventTagClick, onDetectionItemsClick, onRTSPClick }) => {
  return (
    <nav className="top-nav">
      <div className="nav-container">
        <div className="nav-brand">
          <div className="nav-logo">
            <span>⬡</span>
            <span>日月光測試平台</span>
          </div>
          <div className="nav-version">v2.4 (Backend Filter)</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          {authenticated && (
            <button
              onClick={onRTSPClick}
              style={{
                padding: "6px 16px",
                background: "#2a3a2a",
                border: "1px solid #4a5a4a",
                borderRadius: "4px",
                color: "#6bff6b",
                fontSize: "13px",
                cursor: "pointer",
                fontWeight: "500",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                e.target.style.background = "#3a4a3a";
                e.target.style.borderColor = "#5a6a5a";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "#2a3a2a";
                e.target.style.borderColor = "#4a5a4a";
              }}
            >
              📡 RTSP 監控
            </button>
          )}
          {authenticated && (
            <button
              onClick={onDetectionItemsClick}
              style={{
                padding: "6px 16px",
                background: "#2a2a3a",
                border: "1px solid #4a4a5a",
                borderRadius: "4px",
                color: "#6bc3ff",
                fontSize: "13px",
                cursor: "pointer",
                fontWeight: "500",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                e.target.style.background = "#3a3a4a";
                e.target.style.borderColor = "#5a5a6a";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "#2a2a3a";
                e.target.style.borderColor = "#4a5a4a";
              }}
            >
              偵測項目管理
            </button>
          )}
          {authenticated && isAdmin && (
            <button
              onClick={onEventTagClick}
              style={{
                padding: "6px 16px",
                background: "#2a2a3a",
                border: "1px solid #4a4a5a",
                borderRadius: "4px",
                color: "#fff",
                fontSize: "13px",
                cursor: "pointer",
                fontWeight: "500",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                e.target.style.background = "#3a3a4a";
                e.target.style.borderColor = "#5a5a6a";
              }}
              onMouseLeave={(e) => {
                e.target.style.background = "#2a2a3a";
                e.target.style.borderColor = "#4a4a5a";
              }}
            >
              事件標籤管理
            </button>
          )}
        <div className="auth-status">
          <div
            className={`status-indicator ${
              authenticated ? "active" : "inactive"
            }`}
          ></div>
          <span className={`status-text ${authenticated ? "active" : ""}`}>
            {authenticated ? "已驗證" : "未驗證"}
          </span>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
