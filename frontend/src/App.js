import React, { useState, useEffect } from "react";
import "./App.css";

// Hooks
import { useAuth } from "./hooks/useAuth";
import { useAnalysis } from "./hooks/useAnalysis";
import { useRAG } from "./hooks/useRAG";

// Components
import Navbar from "./components/Navbar";
import AuthCard from "./components/AuthCard";
import ModelConfig from "./components/ModelConfig";
import SegmentationParams from "./components/SegmentationParams";
import DetectionSettings from "./components/DetectionSettings";
import AnalysisResults from "./components/AnalysisResults";
import RAGSearch from "./components/RAGSearch";
import ImageSearch from "./components/ImageSearch";
import EventTagModal from "./components/EventTagModal";
import DetectionItemsModal from "./components/DetectionItemsModal";

// Services
import apiService from "./services/api";

function App() {
  // Authentication
  const {
    authenticated,
    apiKey,
    authMessage,
    isAdmin,
    verify,
    logout,
    setApiKey,
  } = useAuth();

  // Event Tag Modal state
  const [showEventTagModal, setShowEventTagModal] = useState(false);
  
  // Detection Items Modal state
  const [showDetectionItemsModal, setShowDetectionItemsModal] = useState(false);

  // Analysis
  const { isAnalyzing, analysisData, analysisError, runAnalysis } =
    useAnalysis(apiKey);

  // RAG
  const {
    isSearching,
    ragData,
    ragError,
    ragStats,
    search: searchRAG,
    answer: answerRAG,
    updateStats,
  } = useRAG(apiKey, authenticated);

  // Model configuration
  const [modelType, setModelType] = useState("qwen");
  const [qwenModel, setQwenModel] = useState("qwen3-vl:8b");
  const [source, setSource] = useState("upload");
  const [videoUrl, setVideoUrl] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [selectedVideoId, setSelectedVideoId] = useState(null);

  // Segmentation parameters
  const [segDur, setSegDur] = useState(10);
  const [overlap, setOverlap] = useState(0);

  // Detection settings
  const [samplingFps, setSamplingFps] = useState(0.5);
  const [targetShort, setTargetShort] = useState(720);
  const [eventDetectionPrompt, setEventDetectionPrompt] = useState("");
  const [summaryPrompt, setSummaryPrompt] = useState("");
  const [defaultPrompts, setDefaultPrompts] = useState({
    event: "Loading...",
    summary: "Loading...",
  });
  const [showEventPromptModal, setShowEventPromptModal] = useState(false);
  const [showSummaryPromptModal, setShowSummaryPromptModal] = useState(false);

  // RAG settings
  const [ragQuery, setRagQuery] = useState("");
  const [ragTopK, setRagTopK] = useState(5);
  const [ragThreshold, setRagThreshold] = useState(0.5);

  // Message handler (for non-auth messages - could be enhanced with toast notifications)
  const showMessage = (text, type = "info") => {
    // For now, just log to console. Could be enhanced with a toast notification system
    console.log(`[${type.toUpperCase()}] ${text}`);
  };

  // Initialize default prompts
  useEffect(() => {
    if (authenticated && apiKey) {
      apiService.getDefaultPrompts(apiKey).then(setDefaultPrompts);
    }
  }, [authenticated, apiKey]);

  // Check Ollama status and log to console
  useEffect(() => {
    if (authenticated && apiKey) {
      const checkOllama = async () => {
        console.log("%c" + "=".repeat(60), "color: #3b82f6; font-weight: bold; font-size: 14px");
        console.log("%c[Ollama 狀態檢查]", "color: #3b82f6; font-weight: bold; font-size: 14px");
        console.log("%c正在檢查 Ollama 服務狀態...", "color: #6b7280");
        
        const status = await apiService.checkOllamaStatus(apiKey);
        
        if (status.available) {
          console.log("%c✓ Ollama 服務可用", "color: #10b981; font-weight: bold");
          console.log("%c  服務地址: " + status.ollama_base, "color: #6b7280; font-size: 12px");
          console.log("%c  可用模型數量: " + (status.model_count || 0), "color: #6b7280; font-size: 12px");
          if (status.models && status.models.length > 0) {
            console.log("%c  可用模型列表:", "color: #6b7280; font-size: 12px");
            status.models.forEach((model, idx) => {
              console.log("%c    " + (idx + 1) + ". " + model, "color: #10b981; font-size: 12px");
            });
          } else {
            console.log("%c  ⚠️ 沒有找到可用模型", "color: #f59e0b; font-size: 12px");
          }
        } else {
          console.log("%c✗ Ollama 服務不可用", "color: #ef4444; font-weight: bold");
          console.log("%c  狀態: " + status.status, "color: #ef4444; font-size: 12px");
          if (status.error) {
            console.log("%c  錯誤訊息: " + status.error, "color: #ef4444; font-size: 12px");
          }
          console.log("%c  建議: 請檢查 Ollama 服務是否正常運行", "color: #f59e0b; font-size: 12px");
        }
        console.log("%c" + "=".repeat(60), "color: #3b82f6; font-weight: bold; font-size: 14px");
      };
      
      checkOllama();
    }
  }, [authenticated, apiKey]);

  // Update model when modelType changes
  useEffect(() => {
    if (modelType === "gemini") {
      setQwenModel("gemini-2.5-flash");
    } else if (modelType === "qwen") {
      setQwenModel("qwen3-vl:8b");
    }
  }, [modelType]);

  // File change handler
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setVideoFile(file);
      setFileName(file.name);
    }
  };

  // Run analysis handler
  const handleRun = async () => {
    if (!authenticated) {
      showMessage("尚未通過驗證，請先登入", "error");
      return;
    }

    const formData = new FormData();
    formData.append("model_type", modelType);
    formData.append("segment_duration", segDur);
    formData.append("overlap", overlap);

    if (modelType === "qwen" || modelType === "gemini") {
      formData.append("qwen_model", qwenModel);
      const calculatedFrames = Math.max(1, Math.ceil(samplingFps * segDur));
      formData.append("frames_per_segment", calculatedFrames);
      formData.append("target_short", targetShort);
      if (eventDetectionPrompt.trim()) {
        formData.append("event_detection_prompt", eventDetectionPrompt);
      }
      if (summaryPrompt.trim()) {
        formData.append("summary_prompt", summaryPrompt);
      }
    }

    if (source === "upload") {
      if (!videoFile) {
        showMessage("請選擇影片檔案", "warning");
        return;
      }
      formData.append("file", videoFile);
    } else if (source === "existing") {
      if (!selectedVideoId) {
        showMessage("請選擇已上傳的影片", "warning");
        return;
      }
      // 重新分析已存在的影片
      formData.append("video_id", selectedVideoId);
    } else {
      const url = videoUrl.trim();
      if (!url) {
        showMessage("請輸入影片 URL", "warning");
        return;
      }
      formData.append("video_url", url);
    }

    const result = await runAnalysis(formData, (data) => {
      if (data.rag_auto_indexed) {
        const rag = data.rag_auto_indexed;
        if (rag.success) {
          showMessage(`✓ 分析完成 | ${rag.message}`, "success");
          updateStats();
        } else if (rag.enabled === false) {
          showMessage("✓ 分析完成（RAG 自動索引已停用）", "success");
        } else {
          showMessage(
            `✓ 分析完成 | RAG 索引失敗：${rag.error || "未知錯誤"}`,
            "warning"
          );
        }
      } else {
        showMessage("✓ 分析完成", "success");
      }
    });

    if (!result.success) {
      showMessage("分析失敗，請檢查輸出", "error");
    }
  };

  // RAG search handler
  const handleRagSearch = async () => {
    if (!authenticated) {
      showMessage("請先登入", "error");
      return;
    }
    if (!ragQuery.trim()) {
      showMessage("請輸入查詢關鍵字", "warning");
      return;
    }

    const result = await searchRAG(ragQuery, ragTopK, ragThreshold);
    if (result.success) {
      if (result.count === 0) {
        showMessage(
          `找到 0 筆結果 (可能皆低於門檻 ${(ragThreshold * 100).toFixed(0)}%)`,
          "warning"
        );
      } else {
        showMessage(`✓ 找到 ${result.count} 筆相關結果`, "success");
      }
    } else {
      showMessage("查詢失敗", "error");
    }
  };

  // RAG answer handler
  const handleRagAnswer = async () => {
    if (!authenticated) {
      showMessage("請先登入", "error");
      return;
    }
    if (!ragQuery.trim()) {
      showMessage("請輸入查詢關鍵字", "warning");
      return;
    }

    const result = await answerRAG(ragQuery, ragTopK, ragThreshold);
    if (result.success) {
      showMessage("✓ RAG + LLM 回答完成", "success");
    } else {
      showMessage(`RAG + LLM 查詢錯誤: ${result.error}`, "error");
    }
  };

  return (
    <div className="app-wrapper">
      <Navbar 
        authenticated={authenticated} 
        isAdmin={isAdmin}
        onEventTagClick={() => setShowEventTagModal(true)}
        onDetectionItemsClick={() => setShowDetectionItemsModal(true)}
      />

      <main className="main-container">
        <div className="page-header">
          <h1 className="page-title">Video Analysis</h1>
          <p className="page-subtitle">
            Configure authentication, select processing models, and execute
            AI-powered video segmentation analysis.
          </p>
        </div>

        <AuthCard
          authenticated={authenticated}
          apiKey={apiKey}
          authMessage={authMessage}
          onVerify={verify}
          onLogout={logout}
          onApiKeyChange={setApiKey}
        />

        <div className={authenticated ? "" : "lock"}>
          <ModelConfig
            modelType={modelType}
            qwenModel={qwenModel}
            source={source}
            videoUrl={videoUrl}
            videoFile={videoFile}
            fileName={fileName}
            selectedVideoId={selectedVideoId}
            apiKey={apiKey}
            authenticated={authenticated}
            onModelTypeChange={setModelType}
            onQwenModelChange={setQwenModel}
            onSourceChange={setSource}
            onVideoUrlChange={setVideoUrl}
            onFileChange={handleFileChange}
            onSelectedVideoChange={setSelectedVideoId}
          />

          <SegmentationParams
            segDur={segDur}
            overlap={overlap}
            onSegDurChange={setSegDur}
            onOverlapChange={setOverlap}
          />

          {(modelType === "qwen" || modelType === "gemini") && (
            <DetectionSettings
              samplingFps={samplingFps}
              targetShort={targetShort}
              eventDetectionPrompt={eventDetectionPrompt}
              summaryPrompt={summaryPrompt}
              defaultPrompts={defaultPrompts}
              showEventPromptModal={showEventPromptModal}
              showSummaryPromptModal={showSummaryPromptModal}
              onSamplingFpsChange={setSamplingFps}
              onTargetShortChange={setTargetShort}
              onEventPromptChange={setEventDetectionPrompt}
              onSummaryPromptChange={setSummaryPrompt}
              onShowEventModal={() => setShowEventPromptModal(true)}
              onHideEventModal={() => setShowEventPromptModal(false)}
              onShowSummaryModal={() => setShowSummaryPromptModal(true)}
              onHideSummaryModal={() => setShowSummaryPromptModal(false)}
              onApplyEventPrompt={(prompt) => {
                setEventDetectionPrompt(prompt);
                setShowEventPromptModal(false);
              }}
              onApplySummaryPrompt={(prompt) => {
                setSummaryPrompt(prompt);
                setShowSummaryPromptModal(false);
              }}
            />
          )}

          <div className="card">
            <div className="btn-group">
              <button
                onClick={handleRun}
                className="btn btn-primary"
                disabled={isAnalyzing}
              >
                <span>▶</span>
                <span>
                  {isAnalyzing ? "Processing..." : "Execute"}
                </span>
              </button>
            </div>
          </div>

          <div className="output-section">
            <div className="output-header">
              <h3 className="output-title">Execution Output</h3>
            </div>
            <div className="output-panel">
              {isAnalyzing && (
                <div className="loading">
                  <span className="spinner"></span>Processing request...
                </div>
              )}
              {analysisError && (
                <div style={{ color: "#ef4444" }}>Error: {analysisError}</div>
              )}
              {!isAnalyzing && !analysisError && !analysisData && (
                <div style={{ color: "#888" }}>Awaiting execution...</div>
              )}
              {!isAnalyzing && analysisData && (
                <AnalysisResults data={analysisData} apiKey={apiKey} authenticated={authenticated} />
              )}
            </div>
          </div>

          <RAGSearch
            ragQuery={ragQuery}
            ragTopK={ragTopK}
            ragThreshold={ragThreshold}
            ragStats={ragStats}
            isSearching={isSearching}
            ragData={ragData}
            ragError={ragError}
            apiKey={apiKey}
            onQueryChange={setRagQuery}
            onTopKChange={setRagTopK}
            onThresholdChange={setRagThreshold}
            onSearch={handleRagSearch}
            onAnswer={handleRagAnswer}
          />

          <ImageSearch apiKey={apiKey} authenticated={authenticated} />
        </div>
      </main>

      <EventTagModal
        isOpen={showEventTagModal}
        onClose={() => setShowEventTagModal(false)}
        apiKey={apiKey}
      />

      <DetectionItemsModal
        isOpen={showDetectionItemsModal}
        onClose={() => setShowDetectionItemsModal(false)}
        apiKey={apiKey}
      />
    </div>
  );
}

export default App;
