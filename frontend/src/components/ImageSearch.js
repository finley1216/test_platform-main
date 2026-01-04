import React, { useState } from "react";
import ImageSearchResults from "./ImageSearchResults";
import apiService from "../services/api";

const ImageSearch = ({ apiKey, authenticated }) => {
  const [queryType, setQueryType] = useState("image"); // "image" 或 "text"
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [textQuery, setTextQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [threshold, setThreshold] = useState(0.7);
  const [labelFilter, setLabelFilter] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [searchData, setSearchData] = useState(null);
  const [searchError, setSearchError] = useState(null);
  const [searchProgress, setSearchProgress] = useState("");

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      // 創建預覽
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSearch = async () => {
    if (!authenticated || !apiKey) {
      setSearchError("請先登入");
      return;
    }

    if (queryType === "image" && !imageFile) {
      setSearchError("請選擇查詢圖片");
      return;
    }

    if (queryType === "text" && !textQuery.trim()) {
      setSearchError("請輸入文字描述");
      return;
    }

    // 當相似度門檻為 0 時，給出警告
    if (threshold === 0) {
      const confirmMessage = "相似度門檻設為 0% 會查詢所有資料，可能導致搜索時間較長。\n\n建議：\n1. 將相似度門檻提高到 0.3 以上以加快搜索速度\n2. 或使用類別過濾來縮小搜索範圍\n\n是否繼續？";
      if (!window.confirm(confirmMessage)) {
        return;
      }
    }

    setIsSearching(true);
    setSearchError(null);
    setSearchData(null);
    setSearchProgress("準備搜索...");

    try {
      const formData = new FormData();
      
      if (queryType === "image") {
        setSearchProgress("上傳圖片中...");
        formData.append("file", imageFile);
      } else {
        setSearchProgress("處理文字描述...");
        formData.append("text_query", textQuery.trim());
      }
      
      formData.append("top_k", topK);
      formData.append("threshold", threshold);
      if (labelFilter.trim()) {
        formData.append("label_filter", labelFilter.trim());
      }

      setSearchProgress("生成 embedding 中...");
      
      // 設置超時時間為 10 秒
      const timeoutDuration = 10000; // 10 秒
      
      // 設置超時
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
          const timeoutSeconds = timeoutDuration / 1000;
          reject(new Error(`搜索超時（超過 ${timeoutSeconds} 秒）。可能原因：1) 後端處理時間過長 2) 後端服務無回應 3) 網路連線問題。請檢查後端日誌。`));
        }, timeoutDuration);
      });

      const searchPromise = apiService.searchImage(formData, apiKey, timeoutDuration);
      
      setSearchProgress("搜索資料庫中...");
      
      console.log("🔍 [以圖搜圖] 開始搜索，等待後端回應...");
      console.log(`🔍 [以圖搜圖] 超時設置: ${timeoutDuration / 1000} 秒`);
      
      const data = await Promise.race([searchPromise, timeoutPromise]);
      
      console.log("🔍 [以圖搜圖] 收到後端回應:", data);
      
      setSearchProgress("處理結果中...");
      
      // 顯示調試信息（向量信息）
      console.log("🔍 [以圖搜圖] 檢查 debug 信息...", data?.debug);
      
      if (data && data.debug) {
        console.log("=".repeat(60));
        console.log("🔍 [以圖搜圖調試信息]");
        console.log("=".repeat(60));
        
        // 查詢向量信息
        console.log("%c[查詢向量]", "color: #60a5fa; font-weight: bold; font-size: 14px");
        console.log("維度:", data.debug.query_embedding_dim);
        console.log("前10個值:", data.debug.query_embedding_sample);
        console.log("完整向量:", data.debug.query_embedding);
        
        // 第一筆資料向量信息
        if (data.debug.first_crop_info) {
          console.log("%c[資料庫第一筆資料]", "color: #34d399; font-weight: bold; font-size: 14px");
          console.log("ID:", data.debug.first_crop_info.id);
          console.log("類別:", data.debug.first_crop_info.label);
          console.log("路徑:", data.debug.first_crop_info.crop_path);
          console.log("向量維度:", data.debug.first_crop_embedding_dim);
          console.log("向量前10個值:", data.debug.first_crop_embedding_sample);
          console.log("完整向量:", data.debug.first_crop_embedding);
          
          // 如果兩個向量都存在，計算相似度
          if (data.debug.query_embedding && data.debug.first_crop_embedding) {
            try {
              const q = data.debug.query_embedding;
              const f = data.debug.first_crop_embedding;
              if (q.length === f.length && q.length === 512) {
                // 計算 cosine similarity
                let dot = 0, normQ = 0, normF = 0;
                for (let i = 0; i < q.length; i++) {
                  dot += q[i] * f[i];
                  normQ += q[i] * q[i];
                  normF += f[i] * f[i];
                }
                const similarity = dot / (Math.sqrt(normQ) * Math.sqrt(normF));
                console.log("%c[相似度計算]", "color: #fbbf24; font-weight: bold; font-size: 14px");
                console.log("查詢向量與第一筆資料的相似度:", similarity.toFixed(4));
                console.log("當前設定的 threshold:", threshold);
                console.log("是否符合 threshold:", similarity >= threshold ? "✅ 是" : "❌ 否");
              }
            } catch (e) {
              console.warn("計算相似度失敗:", e);
            }
          }
        } else {
          console.warn("%c[資料庫第一筆資料]", "color: #ef4444; font-weight: bold");
          console.warn("資料庫中沒有找到有 CLIP embedding 的記錄");
        }
        
        console.log("=".repeat(60));
      }
      
      setSearchData(data);
      setSearchProgress("");
    } catch (error) {
      console.error("以圖搜圖失敗:", error);
      console.error("錯誤詳情:", {
        message: error.message,
        status: error.status,
        detail: error.detail,
        stack: error.stack,
        name: error.name
      });
      
      // 即使出錯，也嘗試顯示部分信息
      if (error.response || error.data) {
        console.log("錯誤回應數據:", error.response || error.data);
      }
      
      let errorMessage = error.message || "搜索失敗";
      
      // 如果有詳細錯誤信息，添加到錯誤訊息中
      if (error.detail) {
        if (typeof error.detail === 'string') {
          errorMessage += `\n\n詳細信息: ${error.detail}`;
        } else if (error.detail.detail) {
          errorMessage += `\n\n詳細信息: ${error.detail.detail}`;
        } else if (error.detail.error) {
          errorMessage += `\n\n詳細信息: ${error.detail.error}`;
        } else if (error.detail.message) {
          errorMessage += `\n\n詳細信息: ${error.detail.message}`;
        } else {
          errorMessage += `\n\n錯誤詳情: ${JSON.stringify(error.detail, null, 2)}`;
        }
      }
      
      // 如果是 HTTP 500，添加提示
      if (error.status === 500) {
        errorMessage += "\n\n這是後端伺服器錯誤，請檢查：\n1. 後端日誌中的錯誤信息\n2. 後端服務是否正常運行\n3. 資料庫連接是否正常";
      }
      
      setSearchProgress("");
      
      // 如果是網路連線錯誤，提供更多提示
      if (errorMessage.includes("無法連接到後端服務器") || 
          errorMessage.includes("Failed to fetch") ||
          errorMessage.includes("Network error")) {
        setSearchError(
          errorMessage + "\n\n建議：\n" +
          "1. 確認後端服務是否正在運行\n" +
          "2. 檢查瀏覽器控制台中的 API 基礎 URL 設定\n" +
          "3. 確認網路連線正常\n" +
          "4. 檢查是否有 CORS 設定問題"
        );
      } 
      // 如果是超時錯誤，提供更多提示
      else if (errorMessage.includes("超時")) {
        let suggestions = errorMessage + "\n\n建議：\n";
        
        if (threshold === 0) {
          suggestions += "⚠️ 相似度門檻為 0% 會查詢所有資料，導致搜索時間過長\n";
          suggestions += "1. 將相似度門檻提高到 0.3 以上（建議: 0.5-0.7）\n";
          suggestions += "2. 使用類別過濾來縮小搜索範圍\n";
        } else {
          suggestions += "1. 檢查資料庫中是否有 object_crops 資料\n";
          suggestions += "2. 提高相似度門檻（目前: " + (threshold * 100).toFixed(0) + "%，建議: 0.5-0.7）\n";
        }
        suggestions += "3. 減少返回數量（目前: " + topK + "）\n";
        suggestions += "4. 使用類別過濾來縮小搜索範圍";
        
        setSearchError(suggestions);
      }
      // 其他錯誤
      else {
        setSearchError(errorMessage);
      }
    } finally {
      setIsSearching(false);
    }
  };

  const handleClear = () => {
    setImageFile(null);
    setImagePreview(null);
    setTextQuery("");
    setSearchData(null);
    setSearchError(null);
  };

  return (
    <div className="card" style={{ padding: "20px 20px 20px 16px" }}>
      <div className="card-header" style={{ marginBottom: "20px", paddingBottom: "12px" }}>
        <div className="card-title">
          <span className="card-title-icon">🔍</span>
          <span>以圖搜圖</span>
        </div>
      </div>

      <div className="form-grid" style={{ gap: "16px", marginBottom: "16px" }}>
        {/* 查詢類型選擇 */}
        <div className="form-group" style={{ margin: 0 }}>
          <label className="form-label" style={{ marginTop: 0 }}>查詢方式</label>
          <div style={{ display: "flex", gap: "12px" }}>
            <label style={{ display: "flex", alignItems: "center", cursor: "pointer" }}>
              <input
                type="radio"
                value="image"
                checked={queryType === "image"}
                onChange={(e) => {
                  setQueryType(e.target.value);
                  handleClear();
                }}
                style={{ marginRight: "6px" }}
              />
              圖片上傳
            </label>
            <label style={{ display: "flex", alignItems: "center", cursor: "pointer" }}>
              <input
                type="radio"
                value="text"
                checked={queryType === "text"}
                onChange={(e) => {
                  setQueryType(e.target.value);
                  handleClear();
                }}
                style={{ marginRight: "6px" }}
              />
              文字描述
            </label>
          </div>
        </div>

        {/* 圖片上傳 */}
        {queryType === "image" && (
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label" style={{ marginTop: 0 }}>查詢圖片</label>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              className="form-input"
              style={{ padding: "8px" }}
            />
            {imagePreview && (
              <div style={{ marginTop: "12px", textAlign: "center" }}>
                <img
                  src={imagePreview}
                  alt="預覽"
                  style={{
                    maxWidth: "300px",
                    maxHeight: "200px",
                    border: "1px solid var(--gray-300)",
                    borderRadius: "8px",
                    objectFit: "contain",
                  }}
                />
              </div>
            )}
          </div>
        )}

        {/* 文字描述 */}
        {queryType === "text" && (
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label" style={{ marginTop: 0 }}>文字描述</label>
            <input
              className="form-input"
              placeholder='例如："藍色衣服的人"、"紅色汽車"'
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && handleSearch()}
            />
          </div>
        )}
      </div>

      {/* 搜索參數 */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "14px",
          marginTop: 0,
          width: "100%",
        }}
      >
        <div className="form-group" style={{ margin: 0 }}>
          <label className="form-label" style={{ marginTop: 0 }}>返回數量 (Top K)</label>
          <input
            type="number"
            min="1"
            max="50"
            className="form-input"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value) || 10)}
          />
        </div>
        <div className="form-group" style={{ margin: 0 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: "4px",
            }}
          >
            <label className="form-label" style={{ marginTop: 0 }}>相似度門檻</label>
            <span
              style={{
                color: "#4ade80",
                fontWeight: "bold",
                fontSize: "14px",
              }}
            >
              {(threshold * 100).toFixed(0)}%
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
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
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
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
            />
          </div>
        </div>
        <div className="form-group" style={{ margin: 0 }}>
          <label className="form-label" style={{ marginTop: 0 }}>類別過濾（可選）</label>
          <input
            className="form-input"
            placeholder='例如："person", "car"'
            value={labelFilter}
            onChange={(e) => setLabelFilter(e.target.value)}
          />
        </div>
      </div>
      

      <div className="btn-group" style={{ marginTop: "20px", marginBottom: "12px" }}>
        <button
          onClick={handleSearch}
          className="btn btn-primary"
          disabled={isSearching}
        >
          {isSearching ? "搜尋中..." : "搜尋"}
        </button>
        <button
          onClick={handleClear}
          className="btn btn-secondary"
          style={{ marginLeft: "8px" }}
          disabled={isSearching}
        >
          清除
        </button>
      </div>

      <div className="output-section">
        <div className="output-header">
          <h3 className="output-title">搜尋結果</h3>
        </div>
        <div className="output-panel">
          {isSearching && (
            <div className="status-message info">
              <div style={{ marginBottom: "8px" }}>🔍 搜尋中...</div>
              {searchProgress && (
                <div style={{ fontSize: "13px", color: "#6b7280", marginTop: "4px" }}>
                  {searchProgress}
                </div>
              )}
              <div style={{ marginTop: "12px" }}>
                <div className="spinner" style={{ display: "inline-block", marginRight: "8px" }}></div>
                <span style={{ fontSize: "12px", color: "#6b7280" }}>
                  這可能需要一些時間，請稍候...
                </span>
              </div>
            </div>
          )}
          {searchError && (
            <div style={{ 
              color: "#ef4444", 
              whiteSpace: "pre-line",
              padding: "12px",
              background: "#fef2f2",
              borderRadius: "6px",
              border: "1px solid #fecaca"
            }}>
              <strong>錯誤:</strong> {searchError}
            </div>
          )}
          {!isSearching && !searchError && !searchData && (
            <div style={{ color: "#888" }}>尚未搜尋</div>
          )}
          {!isSearching && searchData && (
            <ImageSearchResults data={searchData} apiKey={apiKey} />
          )}
        </div>
      </div>
    </div>
  );
};

export default ImageSearch;

