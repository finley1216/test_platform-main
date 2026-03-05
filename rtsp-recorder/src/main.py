import os
import threading
import schedule
import time
import signal
import sys
from utils.config_loader import (
    get_recorder_config_path, get_stream_config_path, load_config,
)
from rtsp.recorder import *
from rtsp.housekeeping import *
from utils.logger_config import setup_logger


# Set up logging
logger = setup_logger(__name__)

class RtspRecorder:
    def __init__(self):
        self.config_path = str(get_recorder_config_path())
        self.stream_config_path = str(get_stream_config_path())
        self.config = load_config(self.config_path)
        self.stream_config = load_config(self.stream_config_path)
        self.streams = self.stream_config["rtsp_streams"]
        self.output_dir = self.config["output"]["directory"]
        self.recording_enabled = self.config["recording"]["enabled"]
        self.housekeeping_enabled = self.config["housekeeping"]["enabled"]
        self.cleanup_time = self.config["housekeeping"]["cleanup_time"]
        self.housekeeping_days = self.config["housekeeping"]["days_to_keep"]
        self.segment_duration = self.config["recording"]["segment_duration"]
        self.max_recording_duration = self.config["recording"]["max_duration"]
        self.retry_delay = self.config["recording"]["retry_delay"]
        self.schedule = schedule.Scheduler()

        # Track recording threads for proper cleanup
        self.recording_threads = []
        self.running = False

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

    def start(self):
        """Start recording threads for each enabled stream."""
        if not self.recording_enabled:
            logger.info("Recording is disabled")
            return

        if not self.streams:
            logger.warning("No enabled streams found in configuration")
            return

        self.running = True
        # Start recording threads for each enabled stream
        for stream in self.streams:
            if stream["enabled"]:
                camera = stream['camera']
                location = stream['location']
                url = stream['url']
                logger.info(f"Starting recording for {location}-{camera}: {url}")
                thread = threading.Thread(
                    target=record_stream,
                    args=(self.config, location,camera, url),
                    daemon=True,
                    name=f"Recorder-{location}-{camera}"
                )
                thread.start()
                self.recording_threads.append(thread)

    def stop(self):
        """Stop all recording threads gracefully."""
        logger.info("Stopping RTSP Recorder...")
        self.running = False

        # Wait for all recording threads to finish
        for thread in self.recording_threads:
            if thread.is_alive():
                logger.info(f"Waiting for thread {thread.name} to finish...")
                thread.join(timeout=10)  # Wait up to 10 seconds
                if thread.is_alive():
                    logger.warning(f"Thread {thread.name} did not stop gracefully")

        self.recording_threads.clear()
        logger.info("All recording threads stopped")

    def setup_cleanup(self):
        """Setup scheduled cleanup tasks."""
        if self.housekeeping_enabled:
            # Schedule the housekeeping function to run daily at the specified time
            self.schedule.every().day.at(self.cleanup_time).do(
                lambda: housekeeping(self.output_dir, self.housekeeping_days)
            )
            logger.info(f"Housekeeping enabled - will run daily at {self.cleanup_time}")
        else:
            logger.info("Housekeeping is disabled")

    def run_cleanup(self):
        self.schedule.run_pending()

    def cleanup(self):
        """Clean up resources and stop all threads."""
        logger.info("Cleaning up RTSP Recorder...")
        self.stop()
        logger.info("RTSP Recorder cleanup completed")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def main():
    """Main function with proper signal handling and cleanup."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting RTSP Recorder...")
    rtsp_recorder = None

    try:
        rtsp_recorder = RtspRecorder()
        rtsp_recorder.start()
        rtsp_recorder.setup_cleanup()

        logger.info("RTSP Recorder started successfully")

        # Main loop with proper error handling
        while True:
            try:
                rtsp_recorder.run_cleanup()
                time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)  # Wait before retrying

    except Exception as e:
        logger.error(f"Failed to start RTSP Recorder: {e}")
        raise
    finally:
        # Clean up resources
        if rtsp_recorder:
            rtsp_recorder.cleanup()
        logger.info("RTSP Recorder shutdown completed")

if __name__ == "__main__":
    main()