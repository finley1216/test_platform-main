import cv2
import os
import time
import datetime
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

def record_stream(config, location, camera, rtsp_url):
    """Record RTSP stream with configurable parameters."""
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

    # Get configuration values
    output_dir = config["output"]["directory"]
    segment_duration = config["recording"]["segment_duration"]
    max_recording_duration = config["recording"]["max_duration"]
    retry_delay = config["recording"]["retry_delay"]

    while True:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduce memory usage
        # cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 seconds timeout for opening the stream
        # cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 seconds timeout for reading the stream

        if cap.isOpened():
            break

        logger.error(f"Could not open stream {location}-{camera}")
        time.sleep(retry_delay)

    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID") # *'mp4v', *"XVID"
    logger.info(f"Open stream: {location}-{camera} {width}x{height} {fps}FPS")

    total_recording_start = time.time()
    while True:
        # Check if max recording duration exceeded
        if time.time() - total_recording_start >= max_recording_duration:
            logger.info(f"Max recording duration ({max_recording_duration} seconds) reached for {location}-{camera}. Stopping recorder.")
            break

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(output_dir, f"{location}-{camera}-{timestamp}.avi")
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

        start_time = time.time()
        logger.info(f"Recording {location}-{camera} -> {filename}")

        while time.time() - start_time < segment_duration:
            result, frame = cap.read()
            if not result:
                logger.warning(f"Stream lost: {location}-{camera}")
                time.sleep(retry_delay)
                break
            out.write(frame)

        out.release()
        logger.info(f"Saved: {filename}")