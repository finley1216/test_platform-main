export const API_BASE = process.env.REACT_APP_API_BASE || "http://140.117.176.88:8080";

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

