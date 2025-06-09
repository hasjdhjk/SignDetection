import cv2
import torch
import numpy as np
from collections import deque
from .mediapipe.hand_tracker import HandTracker
import json
import time
import os

#---------------------------------------------#
# CONFIGURATION                               #
#                                             #
#---------------------------------------------#
MODEL_PATH = "model.pth"  # Update path if needed
SEQ_LENGTH = 30
CONF_THRESHOLD = 0.8

with open("label_map.json", "r") as f:
    label_map = json.load(f)

LABEL_MAP = {int(v): k for k, v in label_map.items()}  # Ensure keys are ints



#---------------------------------------------#
# INITIALIZATION                              #
#                                             #
#---------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()


window = deque(maxlen=SEQ_LENGTH)
hand_tracker = HandTracker()
cap = cv2.VideoCapture(0)  # Use 1 if iPhone camera

# Prepare JSON log
log_file = "prediction_log.json"
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        prediction_log = json.load(f)
else:
    prediction_log = []

print("[INFO] Starting real-time sign detection...")
start_time = time.time()


#---------------------------------------------#
# MAIN LOOP                                   #
#                                             #
#---------------------------------------------#
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame capture failed.")
        break

    frame = cv2.flip(frame, 1)
    processed_frame, results = hand_tracker.process_frame(frame)
    hand_tracker.draw_landmarks(processed_frame, results)

    feature_vector = hand_tracker.create_landmark_array(results)
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
                # Log result
                log_entry = {
                    "timestamp": round(time.time() - start_time, 2),
                    "label": label,
                    "confidence": round(confidence, 3)
                }
                prediction_log.append(log_entry)
                print(f"[PREDICTED] {label} ({confidence:.2f})")

            else:
                label = "No sign detected"

    cv2.putText(processed_frame, label, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Real-Time Sign Detection", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#---------------------------------------------#
# CLEANUP                                     #
#                                             #
#---------------------------------------------#
with open(log_file, "w") as f:
    json.dump(prediction_log, f, indent=2)

cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Log saved to {log_file}")
