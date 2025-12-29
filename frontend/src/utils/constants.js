// 自動檢測 API 基礎 URL
// 優先順序：
// 1. 環境變數 REACT_APP_API_BASE（最高優先級）
// 2. localStorage 中的 api_base_url（用戶手動設定）
// 3. 根據當前訪問的 hostname 自動選擇：
//    - 如果是 localhost 或 127.0.0.1，使用 localhost:8080（第一和第二綁在一起）
//    - 如果是 140.117.176.88，使用 140.117.176.88:8080（第三和第四綁在一起）
//    - 否則使用當前 hostname 的 8080 端口
const getApiBase = () => {
  console.log(`[API 配置] 開始檢測 API base URL...`);
  console.log(`[API 配置] window.location:`, typeof window !== 'undefined' ? window.location.href : 'N/A');
  
  // 1. 優先使用環境變數
  if (process.env.REACT_APP_API_BASE) {
    console.log(`[API 配置] 使用環境變數 REACT_APP_API_BASE: ${process.env.REACT_APP_API_BASE}`);
    return process.env.REACT_APP_API_BASE;
  }
  console.log(`[API 配置] 環境變數 REACT_APP_API_BASE 未設置`);
  
  // 2. 檢查 localStorage 中是否有手動設定的 API 地址
  try {
    const savedApiBase = localStorage.getItem("api_base_url");
    if (savedApiBase) {
      console.log(`[API 配置] 使用 localStorage 中的 api_base_url: ${savedApiBase}`);
      return savedApiBase;
    }
    console.log(`[API 配置] localStorage 中沒有 api_base_url`);
  } catch (e) {
    // localStorage 可能不可用（例如某些隱私模式）
    console.warn("[API 配置] 無法訪問 localStorage:", e);
  }
  
  // 3. 根據當前訪問的 hostname 自動選擇
  if (typeof window === 'undefined' || !window.location) {
    console.warn("[API 配置] window.location 不可用，使用默認值 localhost:8080");
    return "http://localhost:8080";
  }
  
  const hostname = window.location.hostname;
  const protocol = window.location.protocol; // http: 或 https:
  
  console.log(`[API 配置] 檢測到 hostname: ${hostname}, protocol: ${protocol}`);
  
  let apiBase;
  if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "") {
    // localhost:3000 -> localhost:8080（第一和第二綁在一起）
    apiBase = "http://localhost:8080";
    console.log(`[API 配置] 匹配 localhost，選擇: ${apiBase}`);
  } else if (hostname === "140.117.176.88") {
    // 140.117.176.88:3000 -> 140.117.176.88:8080（第三和第四綁在一起）
    apiBase = "http://140.117.176.88:8080";
    console.log(`[API 配置] 匹配 140.117.176.88，選擇: ${apiBase}`);
  } else {
    // 其他情況：使用當前 hostname 的 8080 端口
    apiBase = `${protocol}//${hostname}:8080`;
    console.log(`[API 配置] 其他 hostname，選擇: ${apiBase}`);
  }
  
  return apiBase;
};

// 不在模組載入時執行，改為在運行時動態獲取
// 這樣可以確保 window.location 已經可用
export const getApiBaseUrl = () => {
  return getApiBase();
};

// 提供一個函數讓用戶可以手動設定 API 地址（例如在開發者控制台）
export const setApiBase = (url) => {
  try {
    localStorage.setItem("api_base_url", url);
    console.log("API 基礎 URL 已設定為:", url);
    console.log("請重新載入頁面以套用新設定");
  } catch (e) {
    console.error("無法保存 API 地址到 localStorage:", e);
  }
};

export const EVENT_MAP = {
  water_flood: "淹水積水",
  fire: "火災濃煙",
  abnormal_attire_face_cover_at_entry: "門禁遮臉",
  person_fallen_unmoving: "人員倒地",
  double_parking_lane_block: "車道阻塞",
  smoking_outside_zone: "違規吸菸",
  crowd_loitering: "聚眾逗留",
  security_door_tamper: "破壞門禁",
};

export const STORAGE_KEYS = {
  API_KEY: "seg_api_key",
};

