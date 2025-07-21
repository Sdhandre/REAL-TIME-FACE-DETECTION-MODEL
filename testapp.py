import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import tensorflow as tf
import traceback
import logging
import os

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)

# --- UI/UX Branding ---
st.set_page_config(page_title="Live Face Detection", page_icon="🤖", layout="wide")
st.markdown("<h1 style='text-align: center; color: #E36209;'>🤖 Live Face Detection Demo</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Webcam-based face detection powered by your custom ML model!</p>",
    unsafe_allow_html=True
)
st.sidebar.header("📷 Video Controls")

# --- Keras H5 Model (cached load from local file) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cpu_model2.h5')
model = load_model()

# --- App state for controlling stream ---
if 'run_face_stream' not in st.session_state:
    st.session_state['run_face_stream'] = False

# --- Sidebar Controls ---
start = st.sidebar.button("▶️ Start Camera", key="start")
stop = st.sidebar.button("⏹️ Stop Camera", key="stop")

if start:
    st.session_state['run_face_stream'] = True
if stop:
    st.session_state['run_face_stream'] = False

# --- Live Status Banner ---
if st.session_state['run_face_stream']:
    st.success("Live detection running. Click Stop Camera in the sidebar to end.")
else:
    st.info("Press Start Camera in the sidebar to begin live detection.")

# --- Main Processor Class (Using User's Original Logic with Cloud Fixes) ---
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self, keras_model) -> None:
        self.model = keras_model
        # Get input shape from the model
        self.input_height = 120
        self.input_width = 120

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")

            # --- Preprocessing from user's original script ---
            crop = img[50:500, 50:500, :]
            
            # Convert color from BGR to RGB for the model
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # *** THE FIX IS HERE: Use cv2.resize for consistency with local environment ***
            resized = cv2.resize(rgb_crop, (self.input_width, self.input_height))
            
            # Normalize and expand dimensions
            normalized_resized = resized / 255.0
            input_data = np.expand_dims(normalized_resized, axis=0)

            # --- Inference ---
            yhat = self.model.predict(input_data, verbose=0)
            
            confidence = yhat[0][0][0] 
            sample_coords = yhat[1][0]

            out_img = img.copy()
            if confidence > 0.5:
                # --- Coordinate Transformation ---
                # Scale coords to the 450x450 crop size
                x1_scaled = int(sample_coords[1] * 450)
                y1_scaled = int(sample_coords[0] * 450)
                x2_scaled = int(sample_coords[3] * 450)
                y2_scaled = int(sample_coords[2] * 450)
                
                # Get the absolute coordinates on the original image by adding the crop offset
                x1 = x1_scaled + 50
                y1 = y1_scaled + 50
                x2 = x2_scaled + 50
                y2 = y2_scaled + 50
                
                # --- Drawing logic from user's local script ---
                # Controls the main rectangle
                cv2.rectangle(out_img, (x1, y1), (x2, y2), (255,0,0), 2)
                
                # Controls the label rectangle
                cv2.rectangle(out_img, (x1, y1-30), (x1+80, y1), (255,0,0), -1)
                
                # Controls the text rendered
                cv2.putText(out_img, 'face', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

            # The final frame sent to the browser MUST be in RGB format.
            rgb_out = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(rgb_out, format="rgb24")

        except Exception as e:
            logging.error(f"Error in video processing: {e}")
            traceback.print_exc()
            return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")

def processor_factory():
    if model is None:
        st.error("Model is not loaded. Cannot start video stream.")
        return None
    return FaceDetectionProcessor(keras_model=model)

# --- Start/Stop Stream Based on State ---
if st.session_state['run_face_stream']:
    webrtc_streamer(
        key="face-detect-stream",
        mode=WebRtcMode.SENDRECV,
        # Add STUN/TURN servers for robust cloud connection
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": ["turn:numb.viagenie.ca"],
                    "username": "webrtc@live.com",
                    "credential": "muazkh",
                },
            ]
        },
        video_processor_factory=processor_factory,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with st.sidebar.expander("ℹ️ How To Use", expanded=True):
    st.markdown("""
    **1. Click “Start Camera”** to begin live detection.
    **2. Allow browser permission** for webcam access.
    **3. Click “Stop Camera”** to end the stream safely.
    """)

st.sidebar.markdown("*Built with ❤️ and Streamlit*")
