// 自動檢測 API 基礎 URL
// 優先順序：
// 1. 環境變數 REACT_APP_API_BASE（最高優先級）
// 2. localStorage 中的 api_base_url（用戶手動設定）
// 3. 根據當前訪問的 hostname 自動選擇：
//    - 如果是 localhost 或 127.0.0.1，使用 localhost:8080（適用於 SSH 隧道）
//    - 否則使用外部 IP（適用於直接訪問）
const getApiBase = () => {
  // 1. 優先使用環境變數
  if (process.env.REACT_APP_API_BASE) {
    return process.env.REACT_APP_API_BASE;
  }
  
  // 2. 檢查 localStorage 中是否有手動設定的 API 地址
  try {
    const savedApiBase = localStorage.getItem("api_base_url");
    if (savedApiBase) {
      return savedApiBase;
    }
  } catch (e) {
    // localStorage 可能不可用（例如某些隱私模式）
    console.warn("無法訪問 localStorage:", e);
  }
  
  // 3. 根據當前訪問的 hostname 自動選擇
  const hostname = window.location.hostname;
  if (hostname === "localhost" || hostname === "127.0.0.1" || hostname === "") {
    // 透過 SSH 隧道訪問時，假設後端也透過 SSH 隧道轉發到 localhost:8080
    return "http://localhost:8080";
  }
  
  // 預設使用外部 IP
  return "http://140.117.176.88:8080";
};

export const API_BASE = getApiBase();

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

