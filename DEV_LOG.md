# 開發日誌 (DEV_LOG)

## Git 操作流程

### 基本提交流程
```bash
git status              # 確認今天改了什麼
git add .               # 全部準備打包
git commit -m "訊息"    # 補充今天大概做了什麼
git push                # 全部上傳
```

### 修改上一個 commit
```bash
git commit --amend --no-edit    # 把變更併入「上一個 commit」
git push --force-with-lease     # 強制上傳上去
```

---

## 網頁執行流程

### 啟動步驟

**1. 啟動 63 的 Docker**
```bash
docker compose up -d
```

**2. 在 88 的 PowerShell 執行 SSH 端口轉發**
```powershell
ssh -g -N ^
  -L 0.0.0.0:3000:localhost:3000 ^
  -L 0.0.0.0:8080:localhost:8080 ^
  -L 0.0.0.0:11434:localhost:11434 ^
  -L 0.0.0.0:18001:localhost:18001 ^
  M133040024@140.117.176.63
```

**3. 訪問前端**
```
http://140.117.176.88:3000
```

### 已部署服務端口
- **3000** - 前端服務
- **8080** - 後端 API
- **11434** - Ollama
- **18001** - 其他服務

---

## 開發日誌

### 2025-12-19

#### Backend
- (待補充)

#### Frontend
- (待補充)

#### 備註
- 已部署到 140.117.176.88
- 所有端口已開放並可正常訪問