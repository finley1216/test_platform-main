import React from "react";
import { autoResizeTextarea } from "../utils/helpers";
import PromptModal from "./PromptModal";

const DetectionSettings = ({
  samplingFps,
  targetShort,
  eventDetectionPrompt,
  summaryPrompt,
  defaultPrompts,
  showEventPromptModal,
  showSummaryPromptModal,
  onSamplingFpsChange,
  onTargetShortChange,
  onEventPromptChange,
  onSummaryPromptChange,
  onShowEventModal,
  onHideEventModal,
  onShowSummaryModal,
  onHideSummaryModal,
  onApplyEventPrompt,
  onApplySummaryPrompt,
}) => {
  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title">
          <span>🤖</span>
          <span>Detection setting</span>
        </div>
      </div>
      <div className="form-grid">
        <div className="form-group">
          <label className="form-label">Sampling Rate (FPS)</label>
          <input
            type="number"
            step="0.5"
            className="form-input"
            value={samplingFps}
            onChange={(e) => onSamplingFpsChange(parseFloat(e.target.value))}
          />
        </div>
        <div className="form-group">
          <label className="form-label">Resolution (px)</label>
          <input
            type="number"
            className="form-input"
            value={targetShort}
            onChange={(e) => onTargetShortChange(e.target.value)}
          />
        </div>
      </div>

      <div className="form-group">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "8px",
          }}
        >
          <label className="form-label">Event Detection Prompt</label>
          <button
            type="button"
            className="btn-view-default"
            onClick={onShowEventModal}
          >
            查看後端預設 Prompt
          </button>
        </div>
        <textarea
          className="form-textarea"
          placeholder="自訂 Prompt 將覆蓋伺服器預設值 • 留空則使用預設"
          value={eventDetectionPrompt}
          onChange={(e) => {
            onEventPromptChange(e.target.value);
            autoResizeTextarea(e);
          }}
        />
      </div>

      <div className="form-group">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "8px",
          }}
        >
          <label className="form-label">Summary Prompt</label>
          <button
            type="button"
            className="btn-view-default"
            onClick={onShowSummaryModal}
          >
            查看後端預設 Prompt
          </button>
        </div>
        <textarea
          className="form-textarea"
          placeholder="自訂 Prompt 將覆蓋伺服器預設值 • 留空則使用預設"
          value={summaryPrompt}
          onChange={(e) => {
            onSummaryPromptChange(e.target.value);
            autoResizeTextarea(e);
          }}
        />
      </div>

      <PromptModal
        isOpen={showEventPromptModal}
        title="後端 Event Detection Prompt"
        content={defaultPrompts.event}
        onClose={onHideEventModal}
        onApply={() => onApplyEventPrompt(defaultPrompts.event)}
      />

      <PromptModal
        isOpen={showSummaryPromptModal}
        title="後端 Summary Prompt"
        content={defaultPrompts.summary}
        onClose={onHideSummaryModal}
        onApply={() => onApplySummaryPrompt(defaultPrompts.summary)}
      />
    </div>
  );
};

export default DetectionSettings;

