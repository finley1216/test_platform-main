import React, { useState, useEffect, useRef } from "react";
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
import RTSPStatusModal from "./components/RTSPStatusModal";

// Services
import apiService from "./services/api";

/** 與後端 vlm_profile_service 的 profile_id 對齊 */
function mapToVlmProfileId(modelType, qwenModel) {
  const qm = (qwenModel || "").trim();
  if (modelType === "qwen" && qm === "qwen2.5vl:latest") return "ollama_qwen25";
  if (modelType === "vllm_qwen") {
    const qml = qm.toLowerCase();
    if (qml.includes("qwen3") && qml.includes("awq")) return "vllm_qwen3_awq";
    if (qml.includes("qwen3")) return "vllm_qwen3";
    return "vllm_qwen25";
  }
  return null;
}

/**
 * 依 /v1/system/vlm 的 probes 推斷「實際在服務」的 vLLM profile。
 * 解決：檔案偏好是 Qwen2.5 但 GPU 上只有 Qwen3 在跑 → 避免 UI 永遠顯示未就緒又鎖死。
 */
function inferVllmProfileIdFromProbes(vlm) {
  if (!vlm?.probes || vlm?.switch?.phase === "loading") return null;
  const sel = vlm.selected_profile_id;
  const mainOk = vlm.probes.vllm_main?.ok === true;
  const q3Ok = vlm.probes.vllm_qwen3?.ok === true;
  const q3AwqOk = vlm.probes.vllm_qwen3_awq?.ok === true;

  const available = [];
  if (mainOk) available.push("vllm_qwen25");
  if (q3Ok) available.push("vllm_qwen3");
  if (q3AwqOk) available.push("vllm_qwen3_awq");

  // 僅在「目前所選 profile 不可用」且「只有一個可用 profile」時自動對齊，避免切換時誤鎖。
  if (!available.length) return null;
  if (available.includes(sel)) return null;
  if (available.length === 1) return available[0];
  return null;
}

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

  // RTSP Modal state
  const [showRTSPModal, setShowRTSPModal] = useState(false);

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

  // Model configuration（預設與後端 DEFAULT_PROFILE vLLM Qwen2.5 一致）
  const [modelType, setModelType] = useState("vllm_qwen");
  const [qwenModel, setQwenModel] = useState("Qwen/Qwen2.5-VL-7B-Instruct-AWQ");
  const [vlmStatus, setVlmStatus] = useState(null);
  /** 使用者剛觸發切換、尚未收到後端 idle+ready 前，用於立即鎖 UI */
  const [vlmLocalSwitching, setVlmLocalSwitching] = useState(false);
  const vlmHydratedRef = useRef(false);
  /** 每次登入僅做一次：依 probes 與偏好不一致時自動對齊 UI + 後端偏好檔 */
  const vlmProbeAlignDoneRef = useRef(false);
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

  // Initialize default prompts and Ollama status sequentially
  useEffect(() => {
    if (authenticated && apiKey) {
      const initSequence = async () => {
        // 1. 取得預設 Prompts
        await apiService.getDefaultPrompts(apiKey).then(setDefaultPrompts);
        
        // 2. 檢查 Ollama 狀態 (放在最後，避免阻塞其他 API)
        console.log("%c" + "=".repeat(60), "color: #3b82f6; font-weight: bold; font-size: 14px");
        console.log("%c[Ollama 狀態檢查]", "color: #3b82f6; font-weight: bold; font-size: 14px");
        console.log("%c正在檢查 Ollama 服務狀態...", "color: #6b7280");
        
        let vlm = await apiService.getVlmStatus(apiKey);
        const alignTo = inferVllmProfileIdFromProbes(vlm);
        if (
          alignTo &&
          !vlmProbeAlignDoneRef.current &&
          vlm?.profiles?.length
        ) {
          const ap = vlm.profiles.find((x) => x.id === alignTo);
          if (ap) {
            vlmProbeAlignDoneRef.current = true;
            setModelType(ap.model_type);
            setQwenModel(ap.qwen_model);
            vlmHydratedRef.current = true;
            try {
              await apiService.selectVlmProfile(apiKey, alignTo);
              vlm = await apiService.getVlmStatus(apiKey);
            } catch (e) {
              console.error("VLM probe align:", e);
            }
          }
        } else if (
          !vlmHydratedRef.current &&
          vlm?.profiles &&
          vlm.selected_profile_id
        ) {
          const p = vlm.profiles.find((x) => x.id === vlm.selected_profile_id);
          if (p) {
            setModelType(p.model_type);
            setQwenModel(p.qwen_model);
            vlmHydratedRef.current = true;
          }
        }
        setVlmStatus(vlm);

        const skipOllamaHealth =
          vlm?.selected_profile_id === "vllm_qwen25" ||
          vlm?.selected_profile_id === "vllm_qwen3" ||
          vlm?.selected_profile_id === "vllm_qwen3_awq";
        if (skipOllamaHealth) {
          console.log(
            "%c[Ollama 狀態檢查] 略過（目前為 vLLM profile，避免對 Ollama 發請求／喚醒載入）",
            "color: #6b7280; font-size: 12px"
          );
        } else {
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
        }
        console.log("%c" + "=".repeat(60), "color: #3b82f6; font-weight: bold; font-size: 14px");
      };
      
      initSequence();
    }
  }, [authenticated, apiKey]);

  useEffect(() => {
    if (!authenticated) {
      vlmHydratedRef.current = false;
      vlmProbeAlignDoneRef.current = false;
    }
  }, [authenticated]);

  const usesVlmProfile =
    modelType === "qwen" || modelType === "vllm_qwen";
  const expectedVlmProfileId = mapToVlmProfileId(modelType, qwenModel);

  /** 後端已就緒且與目前下拉選項一致（避免輪詢到舊狀態就誤解鎖） */
  const vlmBackendReadyForSelection =
    !!vlmStatus &&
    vlmStatus.switch?.phase === "idle" &&
    vlmStatus.readiness?.ready === true &&
    expectedVlmProfileId &&
    vlmStatus.readiness?.profile_id === expectedVlmProfileId;

  /**
   * 使用者觸發切換後，直到後端 readiness 與選項一致才解除 vlmLocalSwitching。
   */
  const vlmSwitchBlocking =
    usesVlmProfile &&
    !!expectedVlmProfileId &&
    vlmStatus?.switch?.phase !== "error" &&
    vlmLocalSwitching;

  /** 後端正在 docker compose 切換（含重新整理頁面後仍卡在 loading）— 鎖模型選項、禁止分析 */
  const vlmBackendSwitching =
    usesVlmProfile && vlmStatus?.switch?.phase === "loading";

  const vlmUiLocked =
    usesVlmProfile &&
    !!expectedVlmProfileId &&
    vlmStatus?.switch?.phase !== "error" &&
    (vlmSwitchBlocking || vlmBackendSwitching);

  useEffect(() => {
    if (!authenticated || !apiKey) return undefined;
    const tick = () => {
      apiService.getVlmStatus(apiKey).then(setVlmStatus);
    };
    tick();
    const intervalMs = vlmUiLocked ? 1000 : 3000;
    const id = setInterval(tick, intervalMs);
    return () => clearInterval(id);
  }, [authenticated, apiKey, vlmUiLocked]);

  useEffect(() => {
    if (!vlmLocalSwitching || !vlmStatus) return;
    if (vlmStatus.switch?.phase === "error") {
      setVlmLocalSwitching(false);
      return;
    }
    if (vlmBackendReadyForSelection) {
      setVlmLocalSwitching(false);
    }
  }, [vlmStatus, vlmLocalSwitching, vlmBackendReadyForSelection]);

  const handleModelTypeChange = async (v) => {
    let qm = qwenModel;
    if (v === "gemini") {
      qm = "gemini-2.5-flash";
    } else if (v === "qwen") {
      qm = "qwen2.5vl:latest";
    } else if (v === "vllm_qwen") {
      qm = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ";
    } else if (v === "moondream") {
      qm = "moondream3-preview";
    }
    const pid = mapToVlmProfileId(v, qm);
    if (pid && apiKey) {
      setVlmLocalSwitching(true);
    } else {
      setVlmLocalSwitching(false);
    }
    setModelType(v);
    setQwenModel(qm);

    if (pid && apiKey) {
      try {
        await apiService.selectVlmProfile(apiKey, pid);
        const next = await apiService.getVlmStatus(apiKey);
        setVlmStatus(next);
        if (!next) setVlmLocalSwitching(false);
      } catch (e) {
        console.error(e);
        setVlmLocalSwitching(false);
      }
    }
  };

  const handleQwenModelChange = async (qm) => {
    const pid = mapToVlmProfileId(modelType, qm);
    if (pid && apiKey) {
      setVlmLocalSwitching(true);
    }
    setQwenModel(qm);

    if (pid && apiKey) {
      try {
        await apiService.selectVlmProfile(apiKey, pid);
        const next = await apiService.getVlmStatus(apiKey);
        setVlmStatus(next);
        if (!next) setVlmLocalSwitching(false);
      } catch (e) {
        console.error(e);
        setVlmLocalSwitching(false);
      }
    }
  };

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
    if (vlmUiLocked) {
      showMessage("VLM 後端仍在切換或載入中（VRAM 釋放／載入），請待就緒後再執行。", "warning");
      return;
    }

    const vlmPid = mapToVlmProfileId(modelType, qwenModel);
    if (vlmPid) {
      const r = vlmStatus?.readiness;
      if (!r?.ready || r.profile_id !== vlmPid) {
        showMessage(
          r?.detail || "VLM 後端尚未就緒或仍在切換中，請稍候再試。",
          "warning"
        );
        return;
      }
    }

    const formData = new FormData();
    formData.append("model_type", modelType);
    formData.append("segment_duration", segDur);
    formData.append("overlap", overlap);

    if (modelType === "qwen" || modelType === "gemini" || modelType === "moondream" || modelType === "vllm_qwen") {
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
        onRTSPClick={() => setShowRTSPModal(true)}
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
            vlmStatus={vlmStatus}
            vlmUiLocked={vlmUiLocked}
            source={source}
            videoUrl={videoUrl}
            videoFile={videoFile}
            fileName={fileName}
            selectedVideoId={selectedVideoId}
            apiKey={apiKey}
            authenticated={authenticated}
            onModelTypeChange={handleModelTypeChange}
            onQwenModelChange={handleQwenModelChange}
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

          {(modelType === "qwen" || modelType === "gemini" || modelType === "vllm_qwen") && (
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
                disabled={isAnalyzing || vlmUiLocked}
                title={
                  vlmUiLocked
                    ? "VLM 後端切換中（VRAM 釋放／載入），請待就緒後再執行"
                    : undefined
                }
              >
                <span>▶</span>
                <span>
                  {isAnalyzing
                    ? "Processing..."
                    : vlmUiLocked
                      ? "VLM 切換／載入中…"
                      : "Execute"}
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

      {showRTSPModal && (
        <RTSPStatusModal
          isOpen={showRTSPModal}
          onClose={() => setShowRTSPModal(false)}
          apiKey={apiKey}
        />
      )}
    </div>
  );
}

export default App;
