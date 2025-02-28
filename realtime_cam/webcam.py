import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort

# YOLO 모델 및 LSTM 모델 로드
yolo_model = YOLO("yolov8n-pose.pt")  # YOLO pose 모델 파일 경로
lstm_model = load_model("lastoflast.keras")  # 학습된 LSTM 모델 파일 경로

# DeepSORT 트래커 초기화
tracker = DeepSort(max_age=5)  # ID 유지를 위한 설정

# 웹캠 로드
cap = cv2.VideoCapture(0)  # 0번 카메라 (웹캠) 사용

# 행동 라벨 정의
action_labels = ['dribble', 'shooting', 'pass']
sequence_length = 30  # 시퀀스 길이 설정
person_sequences = {}  # ID별로 시퀀스 저장

def process_pose_keypoints(keypoints, prev_keypoints=None):
    feature_vector = []
    if hasattr(keypoints, 'xy'):
        keypoints_data = keypoints.xy
        if not isinstance(keypoints_data, np.ndarray):
            keypoints_data = keypoints_data.numpy()
    elif isinstance(keypoints, np.ndarray):
        keypoints_data = keypoints
    else:
        keypoints_data = np.zeros((14, 2))

    joint_pairs = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12)
    ]

    for (j1, j2) in joint_pairs:
        if j1 < keypoints_data.shape[0] and j2 < keypoints_data.shape[0]:
            v1 = keypoints_data[j1]
            v2 = keypoints_data[j2]
            angle = np.arctan2(v2[1] - v1[1], v2[0] - v1[0])
            feature_vector.append(np.degrees(angle))
        else:
            feature_vector.append(0)

    if prev_keypoints is not None and prev_keypoints.shape == keypoints_data.shape:
        if keypoints_data.shape[0] > 11:
            reference_length = np.linalg.norm(keypoints_data[5] - keypoints_data[11])
        else:
            reference_length = 1

        for (j1, j2) in joint_pairs:
            if j1 < keypoints_data.shape[0] and j2 < keypoints_data.shape[0]:
                dist = np.linalg.norm(keypoints_data[j1] - prev_keypoints[j1]) / reference_length
                feature_vector.append(dist)
            else:
                feature_vector.append(0)

        for (j1, j2) in joint_pairs:
            if j1 < keypoints_data.shape[0] and j2 < keypoints_data.shape[0]:
                v1 = keypoints_data[j1]
                v2 = keypoints_data[j2]
                prev_v1 = prev_keypoints[j1]
                prev_v2 = prev_keypoints[j2]
                angle = np.arctan2(v2[1] - v1[1], v2[0] - v1[0])
                prev_angle = np.arctan2(prev_v2[1] - prev_v1[1], prev_v2[0] - prev_v1[0])
                delta_angle = np.degrees(angle - prev_angle)
                feature_vector.append(delta_angle)
            else:
                feature_vector.append(0)

    if len(feature_vector) < 30:
        feature_vector.extend([0] * (30 - len(feature_vector)))

    return np.array(feature_vector)

person_prev_keypoints = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not open video.")
        break

    # YOLO 모델로 사람 탐지
    yolo_results = yolo_model(frame)

    detections = []
    for box in yolo_results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], box.conf[0], 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for i, track in enumerate(tracks):
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        track_id = track.track_id
        bbox = track.to_ltwh()
        x1, y1, w, h = bbox
        x2, y2 = int(x1 + w), int(y1 + h)

        if i < len(yolo_results[0].keypoints):
            keypoints = yolo_results[0].keypoints.xy[i].cpu().numpy()[:, :2]

            feature_vector = process_pose_keypoints(keypoints)

            if track_id not in person_sequences:
                person_sequences[track_id] = deque(maxlen=sequence_length)

            person_sequences[track_id].append(feature_vector)

            current_sequence = list(person_sequences[track_id])
            while len(current_sequence) < sequence_length:
                current_sequence.append(current_sequence[-1])

            sequence_input = np.array(current_sequence).reshape(1, sequence_length, -1)
            prediction = lstm_model.predict(sequence_input)
            action_probs = prediction[0]

            confidence_threshold = 0.45
            max_prob = max(action_probs[:3])
            action_label = action_labels[np.argmax(action_probs)] if max_prob >= confidence_threshold else 'None'

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}: {action_label}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            for j, label in enumerate(action_labels):
                prob_text = f"{label}: {action_probs[j]:.2f}"
                cv2.putText(frame, prob_text, (int(x2) + 10, int(y1) + j * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Webcam Action Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
