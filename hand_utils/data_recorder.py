import cv2
import os
import numpy as np
import time
from hand_tracker import HandTracker

#---------------------------------------------#
# DIRECTORY AND FILE SET-UP                   #
#---------------------------------------------#
label = "thanks"  # change this for other words
video_dir = f"recorded_videos/{label}"
landmark_dir = f"landmark_data/{label}"
os.makedirs(video_dir, exist_ok=True)
os.makedirs(landmark_dir, exist_ok=True)

#---------------------------------------------#
# INITIALISE CAMERA                           #
#---------------------------------------------#
def initialize_camera():
    cap = cv2.VideoCapture(0)  # 1 for iPhone, 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()
    return cap

#---------------------------------------------#
# SAVE LANDMARKS PER VIDEO                    #
#---------------------------------------------#
def save_landmarks(landmark_list, filename):
    np.save(filename, np.array(landmark_list))

#---------------------------------------------#
# GET NEXT AVAILABLE INDEX                    #
#---------------------------------------------#
def get_next_index(directory, prefix, suffix):
    existing = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(suffix)]
    nums = [int(f[len(prefix):-len(suffix)]) for f in existing if f[len(prefix):-len(suffix)].isdigit()]
    return max(nums, default=0) + 1

#---------------------------------------------#
# MAIN                                        #
#---------------------------------------------#
def main():
    cap = initialize_camera()
    hand_tracker = HandTracker()

    width = int(cap.get(3))
    height = int(cap.get(4))

    fps = 30
    video_duration = 1.5
    frame_number = int(fps * video_duration)

    recording = False
    frame_counter = 0
    video_writer = None
    landmark_list = []

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)

        processed_frame, results = hand_tracker.process_frame(frame)
        if processed_frame is None:
            continue

        hand_tracker.draw_landmarks(processed_frame, results)
        hand_tracker.draw_handedness(processed_frame, results)

        frame_array = hand_tracker.create_landmark_array(results)
        print(frame_array)

        cv2.imshow("Hand Tracking", processed_frame)
        key = cv2.waitKey(1) & 0xFF

        #
        # Start recording
        #
        if key == ord('r') and not recording:
            index = get_next_index(video_dir, f"video_{label}_", ".mp4")
            video_filename = os.path.join(video_dir, f"video_{label}_{index}.mp4")
            landmark_filename = os.path.join(landmark_dir, f"landmark_{label}_{index}.npy")

            # Countdown overlay with filename
            for i in range(3, 0, -1):
                ret, countdown_frame = cap.read()
                if not ret:
                    print("Error: Frame capture failed during countdown.")
                    break
                countdown_frame = cv2.flip(countdown_frame, 1)
                cv2.putText(countdown_frame, f'Starting in {i}', (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.putText(countdown_frame, f'Recording: video_{label}_{index}.mp4', (50, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("Hand Tracking", countdown_frame)
                cv2.waitKey(1000)

            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            recording = True
            frame_counter = 0
            landmark_list = []
            print(f"Recording started: {video_filename}")

        #
        # Record video + landmarks
        #
        if recording:
            video_writer.write(frame)
            landmark_list.append(frame_array)
            frame_counter += 1

            if frame_counter >= frame_number:
                recording = False
                video_writer.release()
                save_landmarks(landmark_list, landmark_filename)

                print(f"Recording saved: {video_filename}")
                print(f"Landmarks saved: {landmark_filename}")
                print(f"Saved array (first few): \n{landmark_list[:4]}")

                # Show "Recording Saved" message on screen
                for _ in range(30):  # show for ~1 second (30 frames at 30 FPS)
                    ret, end_frame = cap.read()
                    if not ret:
                        break
                    end_frame = cv2.flip(end_frame, 1)
                    cv2.putText(end_frame, "✅ Recording Saved!", (100, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                    cv2.imshow("Hand Tracking", end_frame)
                    cv2.waitKey(33)  # ~30 FPS

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
