# YOLO-World 權重（離線 / Docker 用）

容器內無法從 GitHub 下載時，請在本機下載後放進此目錄：

```bash
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main
mkdir -p yolo_weights
wget -O yolo_weights/yolov8s-world.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-world.pt
```

或改把 `yolov8s-world.pt` 放到 **models/** 目錄（與 Moondream 同層）亦可，後端會自動使用 `/app/models/yolov8s-world.pt`。

完成後重新建立 backend 容器：

```bash
docker compose up -d backend --force-recreate
```
