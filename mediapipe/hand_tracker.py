# hand_tracker.py
import cv2
import mediapipe as mp
import numpy as np

class HandTracker:

#---------------------------------------------#
# INITIALISE ATTRIBUTES                       #
#                                             #
#---------------------------------------------#
    def __init__(self, max_num_hands = 2, 
                 model_complexity = 1, 
                 min_detection_confidence = 0.5, 
                 min_tracking_confidence = 0.5): 
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )


#---------------------------------------------#
# PROCESS INDIVIDUAL FRAMES                   #
#                                             #
#---------------------------------------------#
    def process_frame(self, frame):
        
        # Flip the frame horizontally. This is for webcam. 
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for Mediapipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Detect hands
        results = self.hands.process(rgb_frame)

        # Convert back to BGR for OpenCV
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        return processed_frame, results


#---------------------------------------------#
# DRAW 21 LANDMARKS ON THE HAND               #
#                                             #
#---------------------------------------------#
    def draw_landmarks(self, frame, results):
        
        if results.multi_hand_landmarks:
        
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks and connections
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

#---------------------------------------------#
# DRAW HANDEDNESS (RIGHT OR LEFT)             #
#                                             #
#---------------------------------------------#
    def draw_handedness(self, frame, results):
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Add handedness label
                label = hand_handedness.classification[0].label  # 'Left' or 'Right'
                confidence = hand_handedness.classification[0].score
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                h, w, _ = frame.shape
                x, y = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2f})",
                    (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
        return frame

#---------------------------------------------#
# PRINT HAND LANDMARKS                        #
#                                             #
#---------------------------------------------#
    def print_hand_landmarks(self, results):
        # Check if landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Print handedness
                handedness_label = hand_handedness.classification[0].label
                confidence = hand_handedness.classification[0].score
                print(f"Handedness: {handedness_label}, Confidence: {confidence:.2f}")

                # Print selected landmarks
                selected_landmarks = [0, 4, 8, 12, 16, 20]  # Wrist and fingertips
                print("Selected Landmarks:")
                for idx in selected_landmarks:
                    landmark = hand_landmarks.landmark[idx]
                    print(f"  Landmark {idx}: (x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f})")


#---------------------------------------------#
# CREATE LANDMARK ARRAY IN PROCESSABLE FORMAT #
#                                             #
#---------------------------------------------#
#
# LSTMs in both TensorFlow and PyTorch require data to be in a specific shape: 
# (batch_size, sequence_length, num_features),
# where batch_size is the number of sequences processed simultaneously during training, 
# sequence_length is the the number of frames in each sliding window,
# and num_features is the number of features per frame. 
#
    def create_landmark_array(self, results): 
        array = list()
        # Check if landmarks are detected
        if results.multi_hand_landmarks: 
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness): 
                # Append handedness to array
                handedness_label = hand_handedness.classification[0].label
                confidence = hand_handedness.classification[0].score
