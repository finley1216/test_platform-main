import React from "react";

const OWLSettings = ({ owlLabels, owlFps, owlThr, onLabelsChange, onFpsChange, onThrChange }) => {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span>🦉</span>
          <span>OWL-V2 Settings</span>
        </div>
      </div>
      <div className="form-grid">
        <div className="form-group">
          <label className="form-label">Labels</label>
          <input
            className="form-input"
            value={owlLabels}
            onChange={(e) => onLabelsChange(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label className="form-label">Sampling Rate (FPS)</label>
          <input
            type="number"
            step="0.1"
            className="form-input"
            value={owlFps}
            onChange={(e) => onFpsChange(parseFloat(e.target.value))}
          />
          <p className="form-hint">
            例如: 0.5 = 2秒一張; 2.0 = 每秒兩張
          </p>
        </div>
        <div className="form-group">
          <label className="form-label">Threshold</label>
          <input
            type="number"
            step="0.01"
            className="form-input"
            value={owlThr}
            onChange={(e) => onThrChange(e.target.value)}
          />
        </div>
      </div>
    </div>
  );
};

export default OWLSettings;

