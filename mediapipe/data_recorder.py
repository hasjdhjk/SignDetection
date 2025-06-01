# 
#
# This file is used for data recording. For each recording, a .npy and a .mp4 file is created. 
#
#


import cv2
import os
import numpy as np
import time
from hand_tracker import HandTracker


#---------------------------------------------#
# DIRECTORY AND FILE SET-UP                   #
#---------------------------------------------#
video_dir = "recorded_videos/A"
landmark_dir = "landmark_data/A"
video_title = "video_A_1.mp4"
landmark_title = "landmark_A_1.npy"
os.makedirs(video_dir, exist_ok=True)
os.makedirs(landmark_dir, exist_ok=True)


#---------------------------------------------#
# INITIALISE CAMERA                           #
#---------------------------------------------#
def initialize_camera():
    cap = cv2.VideoCapture(1)  # 1 for iPhone, 0 for default webcam
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
# MAIN                                        #
#---------------------------------------------#
def main():
    cap = initialize_camera()
    hand_tracker = HandTracker()

    width = int(cap.get(3))
    height = int(cap.get(4))

    # 
    # Change fps and video duration (s) here. 
    #
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

        # Skip displaying if processing is throttled
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
            # Countdown overlay
            for i in range(3, 0, -1):
                ret, countdown_frame = cap.read()
                if not ret:
                    print("Error: Frame capture failed during countdown.")
                    break
                countdown_frame = cv2.flip(countdown_frame, 1)
                cv2.putText(countdown_frame, f'Starting in {i}', (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.imshow("Hand Tracking", countdown_frame)
                cv2.waitKey(1000)


            video_filename = os.path.join(video_dir, video_title)
            landmark_filename = os.path.join(landmark_dir, landmark_title)
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

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
