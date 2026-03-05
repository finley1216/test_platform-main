import cv2
import os

def extract_frames(input_folder, output_folder, interval_seconds=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp4', '.avi', '.mov')):  # Add other video formats if needed
            video_path = os.path.join(input_folder, filename)
            video_name = os.path.splitext(filename)[0]

            # Create a subfolder for each video to store its frames
            video_output_folder = os.path.join(output_folder, video_name)
            if not os.path.exists(video_output_folder):
                os.makedirs(video_output_folder)

            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
            frame_interval = int(fps * interval_seconds)  # Number of frames to skip

            frame_count = 0
            success, frame = cap.read()
            while success:
                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(video_output_folder, f"{video_name}_frame_{frame_count}.jpg")
                    cv2.imwrite(frame_filename, frame)
                success, frame = cap.read()
                frame_count += 1

            cap.release()
            print(f"Frames extracted from {filename} and saved to {video_output_folder}")


input_folder = 'video'
output_folder = 'frames'
interval_seconds = 1
extract_frames(input_folder, output_folder, interval_seconds)