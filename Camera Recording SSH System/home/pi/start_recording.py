from picamera2 import Picamera2, VideoEncoder
import sys
import os

# Directory for saving clips
save_dir = "/home/pi/recorded_clips"
os.makedirs(save_dir, exist_ok=True)

# Initialize Picamera2
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (1920, 1080)})
picam2.configure(video_config)
picam2.start()

recording = False
video_file = None

def start_recording(label):
    global recording, video_file
    if recording:
        print("Already recording!")
        return
    video_file = os.path.join(save_dir, f"{label}.h264")
    picam2.start_recording(VideoEncoder(), video_file)
    recording = True
    print(f"Recording started: {video_file}")

def stop_recording():
    global recording
    if not recording:
        print("No recording to stop!")
        return
    picam2.stop_recording()
    recording = False
    print(f"Recording stopped and saved: {video_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 start_recording.py <start/stop> [label]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "start" and len(sys.argv) == 3:
        start_recording(sys.argv[2])
    elif command == "stop":
        stop_recording()
    else:
        print("Invalid command!")