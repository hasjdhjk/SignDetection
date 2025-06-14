import os
import sys

# Get the absolute path to the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to reach the project root
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add project root to the system path so you can import mediapipe.*
if project_root not in sys.path:
    sys.path.append(project_root)


import cv2
import torch
import numpy as np
from collections import deque
from hand_utils.hand_tracker import HandTracker
import json
import time
import torch.nn as nn

#---------------------------------------------#
# CONFIGURATION                               #
#---------------------------------------------#
MODEL_PATH = "sign_lstm.pth"
SEQ_LENGTH = 30
CONF_THRESHOLD = 0.8
TARGET_FPS = 30
FRAME_DURATION = 1 / TARGET_FPS

with open("label_map.json", "r") as f:
    label_map = json.load(f)
LABEL_MAP = {int(v): k for k, v in label_map.items()}

#---------------------------------------------#
# DEFINE SignLSTM MODEL                       #
#---------------------------------------------#
class SignLSTM(nn.Module):
    def __init__(self, input_size=135, hidden_size=64, num_layers=2, num_classes=3):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

#---------------------------------------------#
# INITIALIZE MODEL                            #
#---------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLSTM(input_size=135, hidden_size=64, num_layers=2, num_classes=len(LABEL_MAP))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

#---------------------------------------------#
# INITIALIZE COMPONENTS                       #
#---------------------------------------------#
window = deque(maxlen=SEQ_LENGTH)
hand_tracker = HandTracker()
cap = cv2.VideoCapture(0)

# Prepare JSON log
log_file = "prediction_log.json"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        prediction_log = json.load(f)
else:
    prediction_log = []

print("[INFO] Starting real-time sign detection...")
start_time = time.time()
prev_time = time.time()

#---------------------------------------------#
# MAIN LOOP                                   #
#---------------------------------------------#
while True:
    now = time.time()
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame capture failed.")
        break

    frame = cv2.flip(frame, 1)
    processed_frame, results = hand_tracker.process_frame(frame)
    hand_tracker.draw_landmarks(processed_frame, results)

    feature_vector = hand_tracker.create_landmark_array(results)
    
    if np.count_nonzero(feature_vector) > 0:
        window.append(feature_vector)

    label = "Detecting..."
    if len(window) == SEQ_LENGTH:
        with torch.no_grad():
            input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()

            if confidence > CONF_THRESHOLD:
                label = LABEL_MAP[pred_idx.item()]
                log_entry = {
                    "timestamp": round(time.time() - start_time, 2),
                    "label": label,
                    "confidence": round(confidence, 3)
                }
                prediction_log.append(log_entry)
                print(f"[PREDICTED] {label} ({confidence:.2f})")
            else:
                label = "No sign detected"

    # FPS calculation
    now = time.time()
    fps = 1 / (now - prev_time + 1e-6)  # avoid divide-by-zero
    prev_time = now  # update for next frame

    # Optional: smooth or clamp FPS for visual clarity
    fps = min(fps, 60)  # Cap displayed FPS to 60
    fps_text = f"FPS: {int(fps)}"

    # Label text
    cv2.putText(processed_frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    # FPS text (lower on screen)
    cv2.putText(processed_frame, fps_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)


    cv2.imshow("Real-Time Sign Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time_elapsed = time.time() - now
    if time_elapsed < FRAME_DURATION:
        time.sleep(FRAME_DURATION - time_elapsed)

#---------------------------------------------#
# CLEANUP                                     #
#---------------------------------------------#
with open(log_file, "w") as f:
    json.dump(prediction_log, f, indent=2)

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Log saved to {log_file}")
