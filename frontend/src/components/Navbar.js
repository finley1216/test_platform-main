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
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {authenticated && (
            <button
              onClick={onRTSPClick}
              style={{
                padding: "8px 20px",
                background: "linear-gradient(180deg, #2d2d2d 0%, #1e1e1e 100%)",
                border: "1px solid #444",
                borderRadius: "4px",
                color: "#e0e0e0",
                fontSize: "13px",
                cursor: "pointer",
                fontWeight: "500",
                transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                letterSpacing: "0.5px"
              }}
              onMouseEnter={(e) => {
                e.target.style.borderColor = "#4CAF50";
                e.target.style.color = "#4CAF50";
                e.target.style.boxShadow = "0 0 10px rgba(76, 175, 80, 0.2)";
              }}
              onMouseLeave={(e) => {
                e.target.style.borderColor = "#444";
                e.target.style.color = "#e0e0e0";
                e.target.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
              }}
            >
              RTSP 監控
            </button>
          )}
          {authenticated && (
            <button
              onClick={onDetectionItemsClick}
              style={{
                padding: "8px 20px",
                background: "linear-gradient(180deg, #2d2d2d 0%, #1e1e1e 100%)",
                border: "1px solid #444",
                borderRadius: "4px",
                color: "#e0e0e0",
                fontSize: "13px",
                cursor: "pointer",
                fontWeight: "500",
                transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                letterSpacing: "0.5px"
              }}
              onMouseEnter={(e) => {
                e.target.style.borderColor = "#2196F3";
                e.target.style.color = "#2196F3";
                e.target.style.boxShadow = "0 0 10px rgba(33, 150, 243, 0.2)";
              }}
              onMouseLeave={(e) => {
                e.target.style.borderColor = "#444";
                e.target.style.color = "#e0e0e0";
                e.target.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
              }}
            >
              偵測項目管理
            </button>
          )}
          {authenticated && isAdmin && (
            <button
              onClick={onEventTagClick}
              style={{
                padding: "8px 20px",
                background: "linear-gradient(180deg, #2d2d2d 0%, #1e1e1e 100%)",
                border: "1px solid #444",
                borderRadius: "4px",
                color: "#e0e0e0",
                fontSize: "13px",
                cursor: "pointer",
                fontWeight: "500",
                transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                letterSpacing: "0.5px"
              }}
              onMouseEnter={(e) => {
                e.target.style.borderColor = "#9c27b0";
                e.target.style.color = "#9c27b0";
                e.target.style.boxShadow = "0 0 10px rgba(156, 39, 176, 0.2)";
              }}
              onMouseLeave={(e) => {
                e.target.style.borderColor = "#444";
                e.target.style.color = "#e0e0e0";
                e.target.style.boxShadow = "0 2px 4px rgba(0,0,0,0.2)";
              }}
            >
              事件標籤管理
            </button>
          )}
        <div className="auth-status" style={{ background: "rgba(255,255,255,0.05)", padding: "6px 12px", borderRadius: "20px", border: "1px solid rgba(255,255,255,0.1)" }}>
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
