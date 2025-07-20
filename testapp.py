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
        
        # *** THE FIX IS HERE: Manually compile the model for inference ***
        # This addresses the warning and ensures .predict() works reliably.
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

# --- Main Processor Class with Visual Debugging ---
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self, keras_model) -> None:
        self.model = keras_model
        self.input_height = self.model.input_shape[1]
        self.input_width = self.model.input_shape[2]

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            frame_height, frame_width, _ = img.shape

            # Preprocessing
            new_dim = max(frame_height, frame_width)
            padded_img = np.zeros((new_dim, new_dim, 3), dtype=np.uint8)
            pad_h = (new_dim - frame_height) // 2
            pad_w = (new_dim - frame_width) // 2
            padded_img[pad_h:pad_h+frame_height, pad_w:pad_w+frame_width] = img
            resized = cv2.resize(padded_img, (self.input_width, self.input_height))
            normalized_resized = resized / 255.0
            input_data = np.expand_dims(normalized_resized, axis=0)

            # Inference
            yhat = self.model.predict(input_data, verbose=0)
            
            # Extract the scalar confidence value
            confidence = yhat[0][0][0]
            coords = yhat[1][0]

            out_img = img.copy()

            # --- VISUAL DEBUGGING ---
            # Always draw the box, but change color based on confidence
            box_color = (50, 205, 50) if confidence > 0.5 else (0, 0, 255) # Green or Red

            # Coordinate Transformation
            box_on_padded_x1 = int(coords[1] * new_dim)
            box_on_padded_y1 = int(coords[0] * new_dim)
            box_on_padded_x2 = int(coords[3] * new_dim)
            box_on_padded_y2 = int(coords[2] * new_dim)
            
            x1 = box_on_padded_x1 - pad_w
            y1 = box_on_padded_y1 - pad_h
            x2 = box_on_padded_x2 - pad_w
            y2 = box_on_padded_y2 - pad_h

            # Draw the box and confidence text on every frame
            cv2.rectangle(out_img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(out_img, f'{round(confidence*100, 1)}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
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
