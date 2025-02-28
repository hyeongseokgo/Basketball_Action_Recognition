import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from datetime import datetime

# YOLO 및 LSTM 모델 로드
yolo_model = YOLO("yolov8n-pose.pt")  # YOLO pose 모델 파일 경로
lstm_model = load_model("lastoflast.keras")  # 학습된 LSTM 모델 파일 경로

# 행동 라벨 정의 및 파라미터 설정
action_labels = ['dribble', 'shooting', 'pass']
sequence_length = 30  # 시퀀스 길이
person_sequences = {}  # ID별로 시퀀스 저장
previous_actions = {}  # ID별로 이전 행동 저장

def process_pose_keypoints(keypoints, prev_keypoints=None):
    feature_vector = []

    # 키포인트가 'Keypoints' 객체인지, 아니면 이미 numpy 배열인지 확인 후 변환
    if hasattr(keypoints, 'xy'):
        keypoints_data = keypoints.xy
        if not isinstance(keypoints_data, np.ndarray):
            keypoints_data = keypoints_data.numpy()
    elif isinstance(keypoints, np.ndarray):
        keypoints_data = keypoints
    else:
        print("Warning: No valid keypoints detected. Using default values.")
        keypoints_data = np.zeros((14, 2))  # 기본값 (0,0)으로 채워서 사용

    # 필요한 관절이 누락된 경우 기본값으로 패딩
    joint_pairs = [
        (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12)
    ]

    # 관절 간 각도 계산
    for (j1, j2) in joint_pairs:
        if (
                j1 < keypoints_data.shape[0] and
                j2 < keypoints_data.shape[0] and
                not np.any(np.isin(keypoints_data[j1], [0, 1])) and
                not np.any(np.isin(keypoints_data[j2], [0, 1]))
        ):
            v1 = keypoints_data[j1]
            v2 = keypoints_data[j2]
            angle = np.arctan2(v2[1] - v1[1], v2[0] - v1[0])
            feature_vector.append(np.degrees(angle))
        else:
            feature_vector.append(0)

    # 이전 프레임과의 거리 및 각도 변화 계산
    if prev_keypoints is not None and prev_keypoints.shape == keypoints_data.shape:
        # 기준 길이 계산 (왼쪽 어깨(5)와 왼쪽 엉덩이(11) 간의 거리)
        reference_length = (
            np.linalg.norm(keypoints_data[5] - keypoints_data[11])
            if not np.any(np.isin([keypoints_data[5], keypoints_data[11]], [0, 1]))
            else 1
        )

        for (j1, j2) in joint_pairs:
            if (
                    j1 < keypoints_data.shape[0] and
                    not np.any(np.isin(keypoints_data[j1], [0, 1])) and
                    not np.any(np.isin(prev_keypoints[j1], [0, 1]))
            ):
                # 거리 계산 및 기준 길이로 정규화
                dist = np.linalg.norm(keypoints_data[j1] - prev_keypoints[j1]) / reference_length
                feature_vector.append(dist)
            else:
                feature_vector.append(0)

        for (j1, j2) in joint_pairs:
            if (
                    j1 < keypoints_data.shape[0] and
                    j2 < keypoints_data.shape[0] and
                    not np.any(np.isin(keypoints_data[j1], [0, 1])) and
                    not np.any(np.isin(keypoints_data[j2], [0, 1]))
            ):
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

    # feature_vector의 길이를 고정
    if len(feature_vector) < 30:
        feature_vector.extend([0] * (30 - len(feature_vector)))

    return np.array(feature_vector)

# Clear 버튼 동작 수정
def clear_log():
    log_text.delete('1.0', tk.END)  # 로그 창 비우기
    previous_actions.clear()  # 이전 행동 기록 초기화

# Tkinter 설정
root = tk.Tk()
root.title("실시간 웹캠 동작 분석")
ico = tk.PhotoImage(file='농구아이콘.png')
root.iconphoto(True, ico)

# 레이아웃 프레임 정의
video_frame = tk.Frame(root, width=400, height=400, bg="black")
video_frame.grid(row=0, column=0, padx=10, pady=10)

log_frame = tk.Frame(root)
log_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

button_frame = tk.Frame(root)
button_frame.grid(row=1, column=0, columnspan=2, pady=10)

# 웹캠 화면 표시
video_label = tk.Label(video_frame, bg="black")
video_label.pack(fill="both", expand=True)

# 로그 표시
log_text = scrolledtext.ScrolledText(log_frame, width=40, height=28, font=("KoPubWorld돋움체 Medium", 10))
log_text.pack(padx=10, pady=10)

# Clear 버튼
clear_button = tk.Button(log_frame, text="Clear", command=clear_log,
                         font=("KoPubWorld돋움체 Medium", 12), width=40)
clear_button.pack(pady=5)


# 종료 버튼
quit_button = tk.Button(button_frame, text="종료", command=lambda: quit_program(),
                        font=("KoPubWorld돋움체 Medium", 12), width=20)
quit_button.pack()

# 웹캠 설정
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


    # YOLO 모델로 사람 탐지
    yolo_results = yolo_model(frame)
    detections = []

    for box in yolo_results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], box.conf[0], 'person'))

    for i, box in enumerate(detections):
        if i < len(yolo_results[0].keypoints):
            keypoints = yolo_results[0].keypoints.xy[i].cpu().numpy()[:, :2]
            feature_vector = process_pose_keypoints(keypoints)

            track_id = i  # 트래킹 ID 대신 인덱스를 ID로 사용
            if track_id not in person_sequences:
                person_sequences[track_id] = deque(maxlen=sequence_length)

            person_sequences[track_id].append(feature_vector)
            current_sequence = list(person_sequences[track_id])
            while len(current_sequence) < sequence_length:
                current_sequence.append(current_sequence[-1])

            sequence_input = np.array(current_sequence).reshape(1, sequence_length, -1)
            prediction = lstm_model.predict(sequence_input)
            action_probs = prediction[0]

            confidence_threshold = 0.55
            max_prob = max(action_probs[:3])
            action_label = action_labels[np.argmax(action_probs)] if max_prob >= confidence_threshold else 'None'

            # 동일한 행동을 중복 기록하지 않도록 조정
            if track_id not in previous_actions:
                previous_actions[track_id] = None

            if action_label in ['shooting', 'pass']:
                if previous_actions[track_id] != action_label:  # 새로운 행동만 기록
                    current_time = datetime.now().strftime("%m-%d %H:%M")  # 현재 시간
                    log_text.insert(tk.END, f"{current_time} / ID {track_id}: {action_label}\n")
                    log_text.see(tk.END)
                    previous_actions[track_id] = action_label
            else:
                previous_actions[track_id] = None  # 행동이 None이면 초기화

            # 바운딩 박스 및 예측된 행동 표시
            x1, y1, w, h = box[0]
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}: {action_label}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # OpenCV 이미지를 PIL 이미지로 변환하여 Tkinter 레이블에 표시
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=pil_image)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)




def quit_program():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

update_frame()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
