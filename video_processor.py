import cv2
from hand_tracker import HandTracker

#---------------------------------------------#
# INITIALISE CAMERA                           #
# first function called                       #
#---------------------------------------------#
def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()
    return cap


#---------------------------------------------#
# MAIN                                        #
#                                             #
#---------------------------------------------#
def main():
    hand_tracker = HandTracker()
    # VIDEO = a function that loads video. 
    # In real_time_overlay, the class type of 'cap' is <class 'cv2.VideoCapture'>. 

    # Error handling for video processing
    while True:
        success, frame = VIDEO.read()
        if not success:
            print("Error: Video loading failed.")
            break

        # Process the frame and get hand landmarks
        processed_frame, results = hand_tracker.process_frame(frame)

        # Print hand landmarks
        hand_tracker.print_hand_landmarks(results)

        # Draw hand landmarks on the frame
        hand_tracker.draw_landmarks(processed_frame, results)

        # Draw handedness on the frame
        hand_tracker.draw_handedness(processed_frame, results)

        # Display the annotated frame
        cv2.imshow("Hand Tracking", processed_frame)


if __name__ == "__main__":
    main()
