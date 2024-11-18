# demo.py
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator


@st.cache_resource
def load_model():
    return YOLO("yolov8m-pose.pt")


def predict(frame, iou=0.7, conf=0.25):
    results = model(
        source=frame,
        device="0" if torch.cuda.is_available() else "cpu",
        iou=0.7,
        conf=0.25,
        verbose=False,
    )
    result = results[0]
    return result


def draw_boxes(result, frame):
    for boxes in result.boxes:
        x1, y1, x2, y2, score, classes = boxes.data.squeeze().cpu().numpy()
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
    return frame


def draw_keypoints(result, frame):
    annotator = Annotator(frame, line_width=1)
    for kps in result.keypoints:
        annotator.kpts(kps)

        for idx, kp in enumerate(kps):
            x, y, score = kp.data.squeeze().cpu().numpy()
            
            if score > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    
    return frame


model = load_model()
uploaded_file = st.file_uploader("파일 선택", type=["PNG", "JPG", "JPEG"])
if uploaded_file is not None:
    if "image" in uploaded_file.type:
        with st.spinner(text="포즈 정보 추출 중.."):
            pil_image = Image.open(uploaded_file).convert("RGB")
            np_image = np.asarray(pil_image)
            cv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            
            result = predict(cv_image)
            image = draw_boxes(result, cv_image)
            image = draw_keypoints(result, image)
            st.image(image, channels="BGR")