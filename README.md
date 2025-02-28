# 농구경기 행동분석모델
[Basketball Game Action Recognition Model]

본 프로젝트는 2024년도 서경대학교 컴퓨터공학과 졸업작품으로 진행된 프로젝트입니다.
- 프로젝트 이름 : 농구경기 행동분석모델
- 목표 : 농구경기 기록지 자동화
- 기간 : 2024.05 ~ 2024.11

---

### **Overview**

농구 경기에 관련된 행동들을 YOLOv8 Pose를 통해 관절 정보를 예측하고 이를 LSTM 분류 모델을 통해 지정한 행동을 학습시킨다. 이 모델을 통해 실시간 농구 경기 영상을 입력하고 ROI 지정을 하여 선택된 사람의 행동을 추적하고 예측한다. 행동 예측이 마무리 되면 총 행동 횟수를 출력하는 프로그램을 제작한다.

실시간 농구 경기를 불러오는 것은 무리가 있다고 판단하여 두 과정으로 나눠서 프로젝트를 진행했다. 첫번째는 실시간 웹캠 영상을 입력하여 사람을 인식하고 행동을 판단하여 로그를 남기는 프로그램을 제작하였고, 두번째는 실제 농구 경기 동영상을 인풋하면 ROI로 지정하고 행동 분석 후 행동 횟수를 출력하는 프로그램을 제작하였다.

---

### System Pipeline

![Image](https://github.com/user-attachments/assets/791f8886-eaec-450a-a7da-62745721f1a5)
---

### **Example**
1. 실시간 웹캠 행동 분석 프로그램
   ![Image](https://github.com/user-attachments/assets/734606cc-a67e-4f62-b214-9b4ec84318af)

2. 농구 동영상 행동 분석 프로그램
   ![Image](https://github.com/user-attachments/assets/82a44163-c270-4677-946a-6d090da53823)

---

### Reference
- OpenCV: https://github.com/opencv/opencv
- Ultralytics YOLO Pose Estimation: https://docs.ultralytics.com/ko/tasks/pose/
- aihub: https://www.aihub.or.kr/
- Fight_detection: https://github.com/imsoo/fight_detection
