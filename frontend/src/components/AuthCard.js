import React from "react";

const AuthCard = ({ authenticated, apiKey, authMessage, onVerify, onLogout, onApiKeyChange }) => {
  return (
    <div className="card auth-card">
      <div className="card-header">
        <div className="card-title">
          <span className="card-title-icon">🔐</span>
          <span>Authentication</span>
        </div>
        <p className="card-description">
          Enter your API key to authenticate.
        </p>
      </div>

      {!authenticated ? (
        <div className="form-group">
          <label className="form-label">API Key</label>
          <input
            type="password"
            className="form-input"
            placeholder="Enter server key (e.g. my-secret-key)"
            value={apiKey}
            onChange={(e) => onApiKeyChange(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && onVerify(apiKey)}
          />
          <div className="btn-group" style={{ marginTop: "16px" }}>
            <button onClick={() => onVerify(apiKey)} className="btn btn-primary">
              <span>🔓</span> 驗證並啟用
            </button>
          </div>
        </div>
      ) : (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div style={{ color: "#4ade80" }}>✓ 驗證成功</div>
          <button onClick={onLogout} className="btn btn-ghost">
            登出
          </button>
        </div>
      )}
      {authMessage.text && (
        <div
          className={`status-message ${authMessage.type}`}
          style={{ marginTop: "12px" }}
        >
          {authMessage.text}
        </div>
      )}
    </div>
  );
};

export default AuthCard;

