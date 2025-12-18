import React from "react";

const PromptModal = ({ isOpen, title, content, onClose, onApply }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="modal-title">{title}</h3>
          <button className="modal-close-btn" onClick={onClose}>
            ×
          </button>
        </div>
        <div className="modal-body">
          <pre className="modal-pre">{content}</pre>
        </div>
        <div className="modal-footer">
          <button onClick={onClose} className="btn btn-secondary">
            關閉
          </button>
          <button onClick={onApply} className="btn btn-primary">
            套用到輸入框
          </button>
        </div>
      </div>
    </div>
  );
};

export default PromptModal;

