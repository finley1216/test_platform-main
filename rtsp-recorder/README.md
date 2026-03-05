# Build docker image
- docker build -t rtsp-recorder .

# run docker image (detached mode: -d)
- docker run -d --restart unless-stopped --name rtsp-recorder -v ${dir}/videos:/app/videos rtsp-recorder

# remove container (if need)
- docker rm rtsp-recorder

# check docker runing container
- docker ps

# docker compose
- docker-compose up --build

# Conda Env
- conda create --name rtsp python=3.9
- conda activate rtsp
- conda deactivate
- conda env remove --name rtsp

- pip install -r requirements.txt

# Run recorder
- python src/main.py

# extract frames
- python src/utils/extract_frames.py
