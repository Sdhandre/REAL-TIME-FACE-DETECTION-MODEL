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
def load_h5_model():
    MODEL_PATH = "facetracker.h5"
    st.write("Attempting to load Keras (.h5) model from local repository...")
    try:
        # Load the model and manually compile it for inference
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile()
        st.write("✅ Keras Model loaded and compiled successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model from '{MODEL_PATH}'. Make sure it's in your GitHub repo. Error: {e}")
        return None

model = load_h5_model()

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
            
            # Resize the RGB cropped image for the model
            resized = tf.image.resize(rgb_crop, (self.input_height, self.input_width))
            
            # Normalize and expand dimensions
            normalized_resized = resized / 255.0
            input_data = np.expand_dims(normalized_resized, axis=0)

            # --- Inference ---
            yhat = self.model.predict(input_data, verbose=0)
            
            confidence = yhat[0][0][0]
            sample_coords = yhat[1][0]

            out_img = img.copy()
            if confidence > 0.5:
                # --- Coordinate Transformation from user's original script ---
                start_pt = (int(sample_coords[1]*450) + 50, int(sample_coords[0]*450) + 50)
                end_pt = (int(sample_coords[3]*450) + 50, int(sample_coords[2]*450) + 50)
                
                # Draw the final bounding box and text
                cv2.rectangle(out_img, start_pt, end_pt, (50, 205, 50), 2) # Green box
                cv2.putText(out_img, 'face', (start_pt[0], start_pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # IMPORTANT FIX: The final frame sent back must NOT be converted to RGB again.
            # It should be in the BGR format that OpenCV uses for drawing.
            return av.VideoFrame.from_ndarray(out_img, format="bgr24")

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
