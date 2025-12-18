import React from "react";

const SegmentationParams = ({ segDur, overlap, onSegDurChange, onOverlapChange }) => {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span>✂️</span>
          <span>Segmentation Parameters</span>
        </div>
      </div>
      <div className="form-grid">
        <div className="form-group">
          <label className="form-label">Segment Duration (s)</label>
          <input
            type="number"
            step="0.1"
            className="form-input"
            value={segDur}
            onChange={(e) => onSegDurChange(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label className="form-label">Overlap (s)</label>
          <input
            type="number"
            step="0.1"
            className="form-input"
            value={overlap}
            onChange={(e) => onOverlapChange(e.target.value)}
          />
        </div>
      </div>
    </div>
  );
};

export default SegmentationParams;

