import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from collections import deque
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLO 모델 및 LSTM 모델 로드
yolo_model = YOLO("yolov8n-pose.pt")
lstm_model = load_model("model/lastoflast.keras")
sequence_length = 30
action_labels = ['dribble', 'shooting', 'pass']
tracker = DeepSort(max_age=30, n_init=3)

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

    cap.release()


def process_video(video_path, roi_coords):
    """
    동영상을 입력받아 ROI로 초기 ID를 설정한 후 ID 기반으로 사람을 추적 및 행동 분석.

    :param video_path: 입력 동영상 경로
    :param roi_coords: ROI 좌표 (x, y, w, h)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    person_sequences = {}
    action_counts = {'shooting': 0, 'pass': 0}

    # 초기 상태
    current_frame_count = 0
    predicted_pose = "None"
    confidence_threshold = 0.5
    selected_id = None
    roi_processed = False  # ROI가 처리되었는지 여부

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO로 사람 탐지
        yolo_results = yolo_model(frame)
        detections = []

        for box in yolo_results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # 사람만 탐지
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])

                # 바운딩 박스 확장
                x1, y1, x2, y2 = expand_bounding_box(
                    x1, y1, x2, y2, scale_factor=1.2, image_width=frame.shape[1], image_height=frame.shape[0]
                )

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        # DeepSORT로 추적
        tracks = tracker.update_tracks(detections, frame=frame)

        # ROI로 초기 ID 설정
        if not roi_processed:  # ROI가 처리되지 않은 경우에만 실행
            roi_center = (roi_coords[0] + roi_coords[2] // 2, roi_coords[1] + roi_coords[3] // 2)
            min_distance = float("inf")

            for track in tracks:
                track_bbox = track.to_ltrb()
                track_center = (
                    (track_bbox[0] + track_bbox[2]) // 2,
                    (track_bbox[1] + track_bbox[3]) // 2
                )
                distance = np.linalg.norm(np.array(roi_center) - np.array(track_center))
                if distance < min_distance:
                    min_distance = distance
                    selected_id = int(track.track_id)  # 저장 시 정수형으로 변환

            print(f"ROI와 가장 가까운 ID: {selected_id}")
            roi_processed = True  # ROI가 처리되었음을 설정

        # 선택된 ID만 추적 및 행동 분석
        current_bbox = None
        # DeepSORT 트랙 ID와 YOLO 키포인트 매칭
        for track in tracks:
            if int(track.track_id) != int(selected_id):  # 선택된 ID만 처리
                continue

            # 현재 트랙의 바운딩 박스
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            track_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            current_bbox = (x1, y1, x2, y2)
            # 가장 가까운 YOLO 키포인트를 찾기
            closest_keypoint_index = -1
            min_distance = float("inf")
            for i, keypoints in enumerate(yolo_results[0].keypoints.xy):
                yolo_center = (
                    (yolo_results[0].boxes.xyxy[i][0] + yolo_results[0].boxes.xyxy[i][2]) // 2,
                    (yolo_results[0].boxes.xyxy[i][1] + yolo_results[0].boxes.xyxy[i][3]) // 2,
                )
                distance = np.linalg.norm(np.array(track_center) - np.array(yolo_center))
                if distance < min_distance:
                    min_distance = distance
                    closest_keypoint_index = i
            print('1')
            # 키포인트를 매칭하지 못한 경우 건너뛰기
            if closest_keypoint_index == -1:
                print(f"Track ID {track.track_id}에 대한 키포인트를 찾을 수 없습니다.")
                continue

            print(f"Track ID: {track.track_id}, Closest Keypoint Index: {closest_keypoint_index}")

            # 키포인트를 가져와 처리
            keypoints = yolo_results[0].keypoints.xy[closest_keypoint_index].cpu().numpy()[:, :2]

            if int(selected_id) not in person_sequences:
                person_sequences[int(selected_id)] = deque(maxlen=sequence_length)

            feature_vector = process_pose_keypoints(keypoints)

            person_sequences[int(selected_id)].append(feature_vector)

            # 시퀀스 길이가 충분할 때 행동 예측
            if len(person_sequences[int(selected_id)]) == sequence_length:
                sequence_input = np.array(person_sequences[int(selected_id)]).reshape(1, sequence_length, -1)
                prediction = lstm_model.predict(sequence_input)
                action_probs = prediction[0]
                max_prob = max(action_probs)
                print('3')
                if max_prob >= confidence_threshold:
                    action_label = action_labels[np.argmax(action_probs)]
                    predicted_pose = action_label
                    if action_label in action_counts:
                        action_counts[action_label] += 1
                else:
                    predicted_pose = "None"

        current_frame_count += 1
        yield action_counts, int((current_frame_count / total_frames) * 100), frame, predicted_pose, current_bbox

    cap.release()


def expand_bounding_box(x1, y1, x2, y2, scale_factor, image_width, image_height):
    # 기존 바운딩 박스 중심 계산
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = x2 - x1, y2 - y1

    # 새로운 크기 계산
    new_width = width * scale_factor
    new_height = height * scale_factor

    # 확장된 바운딩 박스 좌표 계산
    new_x1 = max(0, int(cx - new_width / 2))
    new_y1 = max(0, int(cy - new_height / 2))
    new_x2 = min(image_width, int(cx + new_width / 2))
    new_y2 = min(image_height, int(cy + new_height / 2))

    return new_x1, new_y1, new_x2, new_y2


