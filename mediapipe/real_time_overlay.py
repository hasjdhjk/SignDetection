import cv2
import mediapipe
from hand_tracker import HandTracker

#---------------------------------------------#
# INITIALISE CAMERA                           #
# first function called                       #
#---------------------------------------------#
def initialize_camera():
    cap = cv2.VideoCapture(0)                 # 1 for when iphone is close, 0 normally
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()
    return cap


#---------------------------------------------#
# MAIN                                        #
#                                             #
#---------------------------------------------#
def main():
    """Main function to handle hand detection using HandTracker."""
    cap = initialize_camera()
    hand_tracker = HandTracker()

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Frame capture failed.")
            break

        # Flip the image horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)

        # Process the frame and get hand landmarks
        processed_frame, results = hand_tracker.process_frame(frame)

        # Print hand landmarks CONTROL
        # hand_tracker.print_hand_landmarks(results)

        # Print hand landmarks TEST
        final_array = hand_tracker.create_landmark_array(results)
        print(f"Final array: {final_array}")

        # Draw hand landmarks on the frame
        hand_tracker.draw_landmarks(processed_frame, results)

        # Draw handedness on the frame
        hand_tracker.draw_handedness(processed_frame, results)

        # Display the annotated frame
        cv2.imshow("Hand Tracking", processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
