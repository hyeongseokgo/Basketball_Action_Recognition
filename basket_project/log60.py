import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
from yo60 import process_video  # 영상 처리 함수 임포트
from ultralytics import YOLO
import cv2
import threading

# 전역 변수 초기화
video_writer = None
stop_analysis = False  # 분석 중지 플래그

def select_roi(file_path):
    """ROI 선택 및 좌표 출력 (축소된 영상 기반으로 ROI 선택 후 원본 크기로 변환)"""
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "동영상을 로드할 수 없습니다.")
        return

    # 영상 크기 축소 비율 계산
    original_height, original_width = frame.shape[:2]
    scale_factor = min(800 / original_width, 800 / original_height)  # 최대 크기를 800px로 제한
    resized_width = int(original_width * scale_factor)
    resized_height = int(original_height * scale_factor)

    # 프레임 크기 축소
    resized_frame = cv2.resize(frame, (resized_width, resized_height))

    # ROI 선택 (축소된 영상 기반)
    selected_roi = cv2.selectROI("Select ROI (Resized)", resized_frame, False, False)
    print(selected_roi)
    try:
        cv2.destroyWindow("Select ROI (Resized)")
    except cv2.error:
        pass
    cap.release()

    # ROI 값 검증 및 원본 크기로 변환
    if selected_roi == (0, 0, 0, 0):
        messagebox.showerror("Error", "유효하지 않은 ROI입니다.")
        return None
    else:
        # 축소된 ROI를 원본 크기로 변환
        x, y, w, h = selected_roi
        x = int(x / scale_factor)
        y = int(y / scale_factor)
        w = int(w / scale_factor)
        h = int(h / scale_factor)
        original_roi = (x, y, w, h)

        print(f"축소된 ROI 좌표: {selected_roi}")
        print(f"원본 크기 기준 ROI 좌표: {original_roi}")
        return original_roi



def analyze_video_threaded():
    """비동기로 실행될 analyze_video 함수"""
    threading.Thread(target=analyze_video).start()

def analyze_video():
    global video_writer, stop_analysis  # 전역 변수 사용 선언
    stop_analysis = False  # 분석 시작 시 플래그 초기화
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if not file_path:
        return

    try:
        selected = select_roi(file_path)

        # 초기화
        shooting_count = 0
        pass_count = 0
        dribble_frame_count = 0  # 드리블 프레임 수
        previous_action = None
        progress_bar['value'] = 0

        # 로그 업데이트 (불러온 파일 경로)
        log_text.insert(tk.END, f"\n{'='*35}\n")
        log_text.insert(tk.END, f"불러오기 성공 : {file_path}\n")
        log_text.insert(tk.END, f"{'-'*40}\n")
        log_text.see(tk.END)

        # Current Frame 창 생성
        create_current_frame_window()

        # 동영상 저장 설정
        output_video_path = "current_frame_output.avi"  # 저장할 동영상 파일 경로
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (400, 300))

        # process_video 호출
        for action_counts, progress, frame, predicted_pose, current_bbox in process_video(file_path, selected):
            if stop_analysis:  # 분석 중지 요청 시 루프 종료
                break

            if current_bbox:
                x1, y1, x2, y2 = current_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # 진행률 표시 업데이트
            update_progress_bar(progress)

            # 현재 프레임 및 포즈 업데이트
            update_frame_display(frame, predicted_pose)

            # 행동 카운트 업데이트
            if predicted_pose == "dribble":
                dribble_frame_count += 1  # 드리블은 매 프레임 증가
            elif predicted_pose in ["shooting", "pass"] and predicted_pose != previous_action:
                if predicted_pose == "shooting":
                    shooting_count += 1
                elif predicted_pose == "pass":
                    pass_count += 1

            previous_action = predicted_pose  # 이전 액션 업데이트

        # 드리블 시간 계산
        dribble_time = round(dribble_frame_count / 30, 2)  # 초 단위로 변환

        # 최종 결과 출력
        log_text.insert(tk.END, "분석 결과:\n")
        log_text.insert(tk.END, f"    Shooting: {shooting_count}\n")
        log_text.insert(tk.END, f"    Pass: {pass_count}\n")
        log_text.insert(tk.END, f"    Dribble Time: {dribble_time} seconds\n")
        log_text.insert(tk.END, f"{'='*35}\n")
        log_text.see(tk.END)

        # 비디오 저장 종료
        video_writer.release()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during analysis: {str(e)}")

def stop_analysis_action():
    """분석 중지 버튼 동작"""
    global stop_analysis
    stop_analysis = True  # 분석 중지 플래그 설정

def update_progress_bar(progress):
    """진행률 바 업데이트"""
    progress_bar['value'] = progress
    root.update_idletasks()

def update_frame_display(frame, predicted_pose):
    """현재 프레임과 포즈 텍스트를 업데이트"""
    global video_writer

    if frame.shape[0] > frame.shape[1]:
        resized_frame = cv2.resize(frame, (300, 500))
    else:
        resized_frame = cv2.resize(frame, (500, 300))
    # 고정된 크기로 리사이즈
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=pil_image)
    current_frame_label.imgtk = imgtk
    current_frame_label.configure(image=imgtk)

    # 현재 프레임을 동영상으로 저장
    video_writer.write(resized_frame)

    # 포즈 텍스트 업데이트
    pose_label.config(text=f"현재 동작: {predicted_pose}")
    current_frame_window.update_idletasks()

def create_current_frame_window():
    """현재 프레임을 표시할 별도의 창 생성"""
    global current_frame_window, current_frame_label, pose_label
    current_frame_window = tk.Toplevel(root)  # 새로운 창 생성
    current_frame_window.title("현재 프레임")
    current_frame_window.geometry("600x600")  # 창 크기 설정

    current_frame_label = tk.Label(current_frame_window)
    current_frame_label.pack(padx=10, pady=10)

    # 현재 포즈 라벨 추가
    pose_label = tk.Label(current_frame_window, text="현재 동작:", font=("KoPubWorld돋움체 Medium", 12))
    pose_label.pack(pady=5)

    # 정지 버튼 추가
    stop_button = tk.Button(current_frame_window, text="정지", command=stop_analysis_action, font=("KoPubWorld돋움체 Medium", 12))
    stop_button.pack(pady=10)

# Tkinter 설정
root = tk.Tk()
ico = tk.PhotoImage(file='농구아이콘.png')
root.iconphoto(True, ico)
root.title("농구 동작 분류 프로그램")
root.geometry("600x500")  # 창 크기 설정

# UI 요소들
button_frame = tk.Frame(root)
button_frame.grid(row=0, column=0, padx=10, pady=10)

select_button = tk.Button(button_frame, text="동영상 불러오기", command=analyze_video_threaded, font=("KoPubWorld돋움체 Medium", 12))
select_button.grid(row=0, column=0, padx=5)

quit_button = tk.Button(button_frame, text="QUIT", command=root.destroy, font=("KoPubWorld돋움체 Medium", 12))
quit_button.grid(row=0, column=1, padx=5)

log_text = scrolledtext.ScrolledText(root, width=60, height=10, font=("KoPubWorld돋움체 Medium", 10))
log_text.grid(row=1, column=0, padx=10, pady=10)

progress_bar = Progressbar(root, length=600, mode='determinate')
progress_bar.grid(row=2, column=0, padx=10, pady=10)

root.mainloop()
