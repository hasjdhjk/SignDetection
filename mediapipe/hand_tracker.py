import cv2
import mediapipe as mp
import numpy as np
import time

class HandTracker:

#---------------------------------------------#
# INITIALISE ATTRIBUTES                       #
#                                             #
#---------------------------------------------#
    def __init__(self, max_num_hands = 2, 
                 model_complexity = 1, 
                 min_detection_confidence = 0.6, 
                 min_tracking_confidence = 0.6): 
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.skipped_frames = 0
        self.last_sample_time = time.time()
        self.sample_interval = 1 / 20  # target ~230 FPS


#---------------------------------------------#
# PROCESS INDIVIDUAL FRAMES                   #
#                                             #
#---------------------------------------------#
    def process_frame(self, frame):
        current_time = time.time()
        if current_time - self.last_sample_time < self.sample_interval:
            return None, None  # Wait to maintain consistent sampling rate
        self.last_sample_time = current_time

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks is None:
            self.skipped_frames += 1

        return processed_frame, results


#---------------------------------------------#
# DRAW 21 LANDMARKS ON THE HAND               #
#                                             #
#---------------------------------------------#
    def draw_landmarks(self, frame, results):
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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
        if results and results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_handedness.classification[0].label
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
        if results and results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handedness_label = hand_handedness.classification[0].label
                confidence = hand_handedness.classification[0].score
                print(f"Handedness: {handedness_label}, Confidence: {confidence:.2f}")

                selected_landmarks = [0, 4, 8, 12, 16, 20]
                print("Selected Landmarks:")
                for idx in selected_landmarks:
                    landmark = hand_landmarks.landmark[idx]
                    print(f"  Landmark {idx}: (x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f})")

#---------------------------------------------#
# CREATE LANDMARK ARRAY IN PROCESSABLE FORMAT #
#                                             #
#---------------------------------------------#
    def create_landmark_array(self, results): 
        feature_array = [None, None]

        if results and results.multi_hand_landmarks: 
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness): 
                handedness_label = hand_handedness.classification[0].label
                confidence = hand_handedness.classification[0].score

                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                wrist = landmarks[0]
                landmarks -= wrist

                hand_data = landmarks.flatten().tolist()
                hand_data.append(confidence)

                if handedness_label == 'Right': 
                    feature_array[0] = hand_data
                elif handedness_label == 'Left': 
                    feature_array[1] = hand_data

        for i in range(len(feature_array)): 
            if feature_array[i] is None: 
                feature_array[i] = [0] * 64

        final_array = feature_array[0] + feature_array[1]

        # Consistency check
        if len(final_array) != 128:
            raise ValueError(f"Feature vector size mismatch: expected 128, got {len(final_array)}")

        # Add timestamp to the final feature array (optional; keep as float)
        final_array.append(time.time())

        return final_array
