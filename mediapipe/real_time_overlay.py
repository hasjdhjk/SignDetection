# real_time_overlay
import cv2
import os
import numpy as np
from hand_tracker import HandTracker


#---------------------------------------------#
# DIRECTORY SET-UP                            #
# modify per video recording                  #
#---------------------------------------------#
video_dir = "recorded_videos"
landmark_dir = "landmark_data"
os.makedirs(video_dir, exist_ok=True)
os.makedirs(landmark_dir, exist_ok=True)


#---------------------------------------------#
# INITIALISE CAMERA                           #
# first function called                       #
#---------------------------------------------#
def initialize_camera():
    cap = cv2.VideoCapture(0)                 # 1 for when iPhone is close, 0 normally
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()
    return cap


#---------------------------------------------#
# SAVE LANDMARKS PER VIDEO                    #
# one video per .npy file                     #
#---------------------------------------------#
def save_landmarks(landmark_list, filename):
    np.save(filename, np.array(landmark_list))  # Saves as .npy file


#---------------------------------------------#
# MAIN                                        #
#                                             #
#---------------------------------------------#
def main():
    # Initialising
    cap = initialize_camera()
    hand_tracker = HandTracker()

    # Constants
    width = int(cap.get(3))                    # In pixels
    height = int(cap.get(4))
    fps = 30
    video_duration = 1.5                         # In seconds
    frame_number = fps * video_duration

    recording = False
    frame_counter = 0
    video_writer = None
    landmark_list = []
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving videos

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Frame capture failed.")
            break

        # Flip the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Process the frame and get hand landmarks
        processed_frame, results = hand_tracker.process_frame(frame)

        # Print hand landmarks
        # hand_tracker.print_hand_landmarks(results)

        # Draw hand landmarks on the frame
        hand_tracker.draw_landmarks(processed_frame, results)

        # Draw handedness on the frame
        hand_tracker.draw_handedness(processed_frame, results)

        # Print hand landmarks TEST
        frame_array = hand_tracker.create_landmark_array(results)

        # Display the annotated frame
        cv2.imshow("Hand Tracking", processed_frame)
        key = cv2.waitKey(1) & 0xFF

        #
        # Start recording when spacebar is pressed
        # 
        if key == ord('r') and not recording:
            video_filename = os.path.join(video_dir, f"video_label_1.mp4")
            landmark_filename = os.path.join(landmark_dir, f"landmark_label_1.npy")
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
            recording = True
            frame_counter = 0
            landmark_list = []
            print(f"Recording started: {video_filename}")
        
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
                print(f"Saved array: \n {landmark_list[0:4]}")

        # Exit on 'q' key press
        if key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
