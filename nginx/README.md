# Nginx 反向代理

修改 `nginx.conf` 後，**必須在實際對外服務的機器上**重新載入設定才會生效。

## 若出現 502（例如 POST /api/v1/segment_pipeline_multipart）

1. 確認目前跑的設定就是本目錄的 `nginx.conf`（例如用 volume 掛載）。
2. 在該機器上執行：
   ```bash
   cd /path/to/test_platform-main
   docker compose exec nginx nginx -t && docker compose exec nginx nginx -s reload
   ```
3. 若 502 仍發生，請在發生當下執行 `docker compose logs backend --tail 80`，看後端是否有收到請求或崩潰（OOM/例外）。
