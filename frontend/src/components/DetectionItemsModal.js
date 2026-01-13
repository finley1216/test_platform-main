import React, { useState, useEffect } from "react";
import apiService from "../services/api";

const DetectionItemsModal = ({ isOpen, onClose, apiKey }) => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [editingItem, setEditingItem] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showPromptPreview, setShowPromptPreview] = useState(false);
  const [promptPreview, setPromptPreview] = useState("");
  
  // 表單狀態
  const [formData, setFormData] = useState({
    name: "",
    name_en: "",
    name_zh: "",
    description: "",
    is_enabled: true,
  });

  useEffect(() => {
    if (isOpen && apiKey) {
      loadItems();
    }
  }, [isOpen, apiKey]);

  const loadItems = async () => {
    setLoading(true);
    try {
      const data = await apiService.listDetectionItems(apiKey, false);
      setItems(data || []);
    } catch (error) {
      console.error("Failed to load detection items:", error);
      setMessage(`載入失敗：${error.message}`);
      setItems([]);
    } finally {
      setLoading(false);
    }
  };

  const handleAdd = () => {
    setFormData({
      name: "",
      name_en: "",
      name_zh: "",
      description: "",
      is_enabled: true,
    });
    setEditingItem(null);
    setShowAddForm(true);
  };

  const handleEdit = (item) => {
    setFormData({
      name: item.name,
      name_en: item.name_en,
      name_zh: item.name_zh,
      description: item.description || "",
      is_enabled: item.is_enabled,
    });
    setEditingItem(item);
    setShowAddForm(true);
  };

  const handleSave = async () => {
    if (!formData.name.trim() || !formData.name_en.trim() || !formData.name_zh.trim()) {
      setMessage("請填寫所有必填欄位（名稱、英文名稱、中文名稱）");
      return;
    }

    setLoading(true);
    setMessage("");
    try {
      if (editingItem) {
        // 更新現有項目
        await apiService.updateDetectionItem(editingItem.id, formData, apiKey);
        setMessage("偵測項目已更新，frame_prompt.md 已自動更新");
      } else {
        // 創建新項目
        await apiService.createDetectionItem(formData, apiKey);
        setMessage("偵測項目已創建，frame_prompt.md 已自動更新");
      }
      setShowAddForm(false);
      setEditingItem(null);
      loadItems();
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      setMessage(`儲存失敗：${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (itemId, itemName) => {
    if (!window.confirm(`確定要刪除偵測項目「${itemName}」嗎？\n刪除後 frame_prompt.md 會自動更新。`)) {
      return;
    }

    setLoading(true);
    setMessage("");
    try {
      await apiService.deleteDetectionItem(itemId, apiKey);
      setMessage("偵測項目已刪除，frame_prompt.md 已自動更新");
      loadItems();
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      setMessage(`刪除失敗：${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleEnabled = async (item) => {
    setLoading(true);
    setMessage("");
    try {
      await apiService.updateDetectionItem(
        item.id,
        { is_enabled: !item.is_enabled },
        apiKey
      );
      setMessage(`已${!item.is_enabled ? "啟用" : "停用"}偵測項目，frame_prompt.md 已自動更新`);
      loadItems();
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      setMessage(`更新失敗：${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handlePreviewPrompt = async () => {
    setLoading(true);
    try {
      const data = await apiService.previewPrompt(apiKey);
      setPromptPreview(data.prompt_content || "");
      setShowPromptPreview(true);
    } catch (error) {
      setMessage(`預覽失敗：${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleRegeneratePrompt = async () => {
    if (!window.confirm("確定要重新生成 frame_prompt.md 嗎？")) {
      return;
    }

    setLoading(true);
    setMessage("");
    try {
      const result = await apiService.regeneratePrompt(apiKey);
      setMessage(`frame_prompt.md 已重新生成（包含 ${result.enabled_items_count} 個啟用的偵測項目）`);
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      setMessage(`重新生成失敗：${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.7)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: "#1a1a1a",
          borderRadius: "8px",
          padding: "24px",
          width: "90%",
          maxWidth: "900px",
          maxHeight: "90vh",
          overflow: "auto",
          border: "1px solid #333",
          boxShadow: "0 4px 20px rgba(0, 0, 0, 0.5)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 標題列 */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "24px",
            borderBottom: "1px solid #333",
            paddingBottom: "16px",
          }}
        >
          <h2 style={{ color: "#fff", margin: 0, fontSize: "20px", fontWeight: "600" }}>
            偵測項目管理
          </h2>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "#999",
              fontSize: "24px",
              cursor: "pointer",
              padding: "0",
              width: "32px",
              height: "32px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            ×
          </button>
        </div>

        {/* 訊息提示 */}
        {message && (
          <div
            style={{
              padding: "12px",
              marginBottom: "16px",
              borderRadius: "4px",
              background: message.includes("失敗") ? "#3a1a1a" : "#1a3a1a",
              color: message.includes("失敗") ? "#ff6b6b" : "#6bff6b",
              fontSize: "13px",
            }}
          >
            {message}
          </div>
        )}

        {/* 操作按鈕 */}
        <div style={{ display: "flex", gap: "8px", marginBottom: "20px", flexWrap: "wrap" }}>
          <button
            onClick={handleAdd}
            disabled={loading || showAddForm}
            style={{
              padding: "8px 16px",
              background: showAddForm ? "#2a2a2a" : "#2a5a2a",
              border: "1px solid #4a4a4a",
              borderRadius: "4px",
              color: showAddForm ? "#666" : "#6bff6b",
              cursor: showAddForm ? "not-allowed" : "pointer",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            + 新增偵測項目
          </button>
          <button
            onClick={handlePreviewPrompt}
            disabled={loading}
            style={{
              padding: "8px 16px",
              background: "#2a2a3a",
              border: "1px solid #4a4a5a",
              borderRadius: "4px",
              color: "#6bc3ff",
              cursor: loading ? "not-allowed" : "pointer",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            ▶ 預覽 Prompt
          </button>
          <button
            onClick={handleRegeneratePrompt}
            disabled={loading}
            style={{
              padding: "8px 16px",
              background: "#3a3a2a",
              border: "1px solid #5a5a4a",
              borderRadius: "4px",
              color: "#ffd700",
              cursor: loading ? "not-allowed" : "pointer",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            ↻ 重新生成 Prompt
          </button>
        </div>

        {/* 新增/編輯表單 */}
        {showAddForm && (
          <div
            style={{
              padding: "16px",
              background: "#2a2a2a",
              borderRadius: "6px",
              marginBottom: "20px",
              border: "1px solid #444",
            }}
          >
            <div
              style={{
                color: "#fff",
                fontWeight: "600",
                marginBottom: "12px",
                fontSize: "16px",
              }}
            >
              {editingItem ? "編輯偵測項目" : "新增偵測項目"}
            </div>
            
            <div style={{ display: "grid", gap: "12px" }}>
              <div>
                <label style={{ display: "block", color: "#ccc", marginBottom: "6px", fontSize: "13px" }}>
                  唯一識別名稱 <span style={{ color: "#ff6b6b" }}>*</span>
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  placeholder="例如：fire"
                  style={{
                    width: "100%",
                    padding: "8px",
                    background: "#1a1a1a",
                    border: "1px solid #444",
                    borderRadius: "4px",
                    color: "#fff",
                    fontSize: "13px",
                  }}
                />
              </div>
              
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                <div>
                  <label style={{ display: "block", color: "#ccc", marginBottom: "6px", fontSize: "13px" }}>
                    英文名稱 <span style={{ color: "#ff6b6b" }}>*</span>
                  </label>
                  <input
                    type="text"
                    value={formData.name_en}
                    onChange={(e) => setFormData({ ...formData, name_en: e.target.value })}
                    placeholder="例如：fire"
                    style={{
                      width: "100%",
                      padding: "8px",
                      background: "#1a1a1a",
                      border: "1px solid #444",
                      borderRadius: "4px",
                      color: "#fff",
                      fontSize: "13px",
                    }}
                  />
                </div>
                
                <div>
                  <label style={{ display: "block", color: "#ccc", marginBottom: "6px", fontSize: "13px" }}>
                    中文名稱 <span style={{ color: "#ff6b6b" }}>*</span>
                  </label>
                  <input
                    type="text"
                    value={formData.name_zh}
                    onChange={(e) => setFormData({ ...formData, name_zh: e.target.value })}
                    placeholder="例如：火災"
                    style={{
                      width: "100%",
                      padding: "8px",
                      background: "#1a1a1a",
                      border: "1px solid #444",
                      borderRadius: "4px",
                      color: "#fff",
                      fontSize: "13px",
                    }}
                  />
                </div>
              </div>
              
              <div>
                <label style={{ display: "block", color: "#ccc", marginBottom: "6px", fontSize: "13px" }}>
                  偵測標準描述（用於 Prompt）
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  placeholder="例如：可見火焰或持續濃煙竄出"
                  rows="3"
                  style={{
                    width: "100%",
                    padding: "8px",
                    background: "#1a1a1a",
                    border: "1px solid #444",
                    borderRadius: "4px",
                    color: "#fff",
                    fontSize: "13px",
                    fontFamily: "inherit",
                    resize: "vertical",
                  }}
                />
              </div>
              
              <div>
                <label style={{ display: "flex", alignItems: "center", color: "#ccc", fontSize: "13px", cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={formData.is_enabled}
                    onChange={(e) => setFormData({ ...formData, is_enabled: e.target.checked })}
                    style={{ marginRight: "8px" }}
                  />
                  啟用此偵測項目
                </label>
              </div>
            </div>
            
            <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end", marginTop: "16px" }}>
              <button
                onClick={() => {
                  setShowAddForm(false);
                  setEditingItem(null);
                }}
                style={{
                  padding: "8px 16px",
                  background: "#2a2a2a",
                  border: "1px solid #444",
                  borderRadius: "4px",
                  color: "#ccc",
                  cursor: "pointer",
                  fontSize: "13px",
                }}
              >
                取消
              </button>
              <button
                onClick={handleSave}
                disabled={loading}
                style={{
                  padding: "8px 16px",
                  background: loading ? "#2a2a2a" : "#4a4a4a",
                  border: "1px solid #666",
                  borderRadius: "4px",
                  color: loading ? "#666" : "#fff",
                  cursor: loading ? "not-allowed" : "pointer",
                  fontSize: "13px",
                  fontWeight: "500",
                }}
              >
                {loading ? "處理中..." : "儲存"}
              </button>
            </div>
          </div>
        )}

        {/* Prompt 預覽視窗 */}
        {showPromptPreview && (
          <div
            style={{
              padding: "16px",
              background: "#2a2a2a",
              borderRadius: "6px",
              marginBottom: "20px",
              border: "1px solid #444",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
              <div style={{ color: "#fff", fontWeight: "600", fontSize: "16px" }}>
                Prompt 預覽
              </div>
              <button
                onClick={() => setShowPromptPreview(false)}
                style={{
                  background: "none",
                  border: "none",
                  color: "#999",
                  fontSize: "20px",
                  cursor: "pointer",
                  padding: "0",
                }}
              >
                ×
              </button>
            </div>
            <pre
              style={{
                background: "#1a1a1a",
                border: "1px solid #333",
                borderRadius: "4px",
                padding: "12px",
                color: "#ccc",
                fontSize: "12px",
                fontFamily: "monospace",
                overflow: "auto",
                maxHeight: "300px",
                whiteSpace: "pre-wrap",
                wordWrap: "break-word",
              }}
            >
              {promptPreview}
            </pre>
          </div>
        )}

        {/* 偵測項目列表 */}
        <div style={{ marginBottom: "20px" }}>
          <div
            style={{
              color: "#ccc",
              marginBottom: "12px",
              fontSize: "14px",
              fontWeight: "500",
            }}
          >
            偵測項目列表 ({items.length} 個)
          </div>
          
          {loading && !showAddForm ? (
            <div style={{ padding: "12px", background: "#2a2a2a", borderRadius: "4px", color: "#999", textAlign: "center" }}>
              載入中...
            </div>
          ) : items.length === 0 ? (
            <div style={{ padding: "12px", background: "#2a2a2a", borderRadius: "4px", color: "#999", textAlign: "center" }}>
              尚無偵測項目，點擊上方「新增偵測項目」開始建立
            </div>
          ) : (
            <div style={{ display: "grid", gap: "8px" }}>
              {items.map((item) => (
                <div
                  key={item.id}
                  style={{
                    padding: "12px",
                    background: "#2a2a2a",
                    borderRadius: "4px",
                    border: `1px solid ${item.is_enabled ? "#444" : "#333"}`,
                    opacity: item.is_enabled ? 1 : 0.6,
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "6px" }}>
                        <span style={{ color: "#fff", fontWeight: "600", fontSize: "15px" }}>
                          {item.name_zh}
                        </span>
                        <span style={{ color: "#999", fontSize: "13px" }}>
                          ({item.name_en})
                        </span>
                        {!item.is_enabled && (
                          <span
                            style={{
                              padding: "2px 8px",
                              background: "#3a1a1a",
                              border: "1px solid #5a2a2a",
                              borderRadius: "3px",
                              color: "#ff6b6b",
                              fontSize: "11px",
                            }}
                          >
                            已停用
                          </span>
                        )}
                      </div>
                      {item.description && (
                        <div style={{ color: "#999", fontSize: "12px", marginBottom: "4px" }}>
                          {item.description}
                        </div>
                      )}
                      <div style={{ color: "#666", fontSize: "11px" }}>
                        識別名稱：{item.name}
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: "6px", marginLeft: "12px" }}>
                      <button
                        onClick={() => handleToggleEnabled(item)}
                        disabled={loading}
                        title={item.is_enabled ? "停用" : "啟用"}
                        style={{
                          padding: "6px 12px",
                          background: item.is_enabled ? "#3a3a1a" : "#2a3a2a",
                          border: "1px solid #4a4a4a",
                          borderRadius: "3px",
                          color: item.is_enabled ? "#ffd700" : "#6bff6b",
                          cursor: loading ? "not-allowed" : "pointer",
                          fontSize: "12px",
                        }}
                      >
                        {item.is_enabled ? "■" : "▶"}
                      </button>
                      <button
                        onClick={() => handleEdit(item)}
                        disabled={loading || showAddForm}
                        title="編輯"
                        style={{
                          padding: "6px 12px",
                          background: "#2a2a3a",
                          border: "1px solid #4a4a4a",
                          borderRadius: "3px",
                          color: "#6bc3ff",
                          cursor: loading || showAddForm ? "not-allowed" : "pointer",
                          fontSize: "12px",
                        }}
                      >
                        ✎
                      </button>
                      <button
                        onClick={() => handleDelete(item.id, item.name_zh)}
                        disabled={loading}
                        title="刪除"
                        style={{
                          padding: "6px 12px",
                          background: "#3a1a1a",
                          border: "1px solid #5a2a2a",
                          borderRadius: "3px",
                          color: "#ff6b6b",
                          cursor: loading ? "not-allowed" : "pointer",
                          fontSize: "12px",
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 底部按鈕 */}
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={onClose}
            style={{
              padding: "10px 20px",
              background: "#2a2a2a",
              border: "1px solid #444",
              borderRadius: "4px",
              color: "#ccc",
              cursor: "pointer",
              fontSize: "14px",
            }}
          >
            關閉
          </button>
        </div>
      </div>
    </div>
  );
};

export default DetectionItemsModal;
