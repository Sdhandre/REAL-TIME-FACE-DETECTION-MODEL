import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import tensorflow as tf
import traceback
import logging
import gdown
import os

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)

# --- UI/UX Branding ---
st.set_page_config(page_title="Live Face Detection", page_icon="🤖", layout="wide")
st.markdown("<h1 style='text-align: center; color: #E36209;'>🤖 Live Face Detection Demo</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Webcam-based face detection powered by a Keras (.h5) model!</p>",
    unsafe_allow_html=True
)
st.sidebar.header("📷 Video Controls")

# --- Keras H5 Model (cached load from Google Drive) ---
@st.cache_resource
def load_h5_model():
    FILE_ID = "1OiakFnWq3_WJqfSJPyFbqnLpMOQYpxhy"
    MODEL_PATH = "facetracker.h5"

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading H5 model from Google Drive..."):
            try:
                gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None

    st.write("Attempting to load Keras (.h5) model...")
    try:
        # Load the model without its training configuration
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
        # Manually compile the model for inference
        model.compile()
        
        st.write("✅ Keras Model loaded and compiled successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model from '{MODEL_PATH}'. Error: {e}")
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

# --- Main Processor Class (Using User's Original Logic with Fixes) ---
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self, keras_model) -> None:
        self.model = keras_model
        self.input_height = self.model.input_shape[1]
        self.input_width = self.model.input_shape[2]

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")

            # --- Preprocessing from user's original script ---
            crop = img[50:500, 50:500, :]
            
            # *** THE FIX IS HERE: CONVERT COLOR FROM BGR TO RGB ***
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Resize the RGB cropped image for the model
            resized = cv2.resize(rgb_crop, (self.input_width, self.input_height))
            
            # Normalize and expand dimensions
            normalized_resized = resized / 255.0
            input_data = np.expand_dims(normalized_resized, axis=0)

            # --- Inference ---
            yhat = self.model.predict(input_data, verbose=0)
            
            confidence = yhat[0][0][0]
            sample_coords = yhat[1][0]

            out_img = img.copy()
            
            # --- VISUAL DEBUGGING ---
            box_color = (50, 205, 50) if confidence > 0.5 else (0, 0, 255) # Green or Red

            # Coordinate Transformation
            start_pt = (int(sample_coords[1]*450) + 50, int(sample_coords[0]*450) + 50)
            end_pt = (int(sample_coords[3]*450) + 50, int(sample_coords[2]*450) + 50)
            
            # Always draw the box and confidence text
            cv2.rectangle(out_img, start_pt, end_pt, box_color, 2)
            cv2.putText(out_img, f'{round(confidence*100, 1)}%', (start_pt[0], start_pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            # ------------------------

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

if st.session_state['run_face_stream']:
    webrtc_streamer(
        key="face-detect-stream",
        mode=WebRtcMode.SENDRECV,
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
