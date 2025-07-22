import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage for TensorFlow
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import tensorflow as tf

# --- Streamlit Page Config ---
st.set_page_config(page_title="Live Face Detection!", page_icon=":smiley:", layout="wide")
st.markdown("<h1 style='text-align: center; color: #E36209;'>🤖 Live Face Detection Demo</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Webcam-based face detection with custom ML model</p>", unsafe_allow_html=True)
st.sidebar.header("📷 Video Controls")

# --- Model Loader (CPU Only) ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('facetrackercpu.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# --- App State & Sidebar Controls ---
if "run_face_stream" not in st.session_state:
    st.session_state["run_face_stream"] = False

start = st.sidebar.button("▶️ Start Camera", key="start")
stop = st.sidebar.button("⏹️ Stop Camera", key="stop")
if start:
    st.session_state["run_face_stream"] = True
if stop:
    st.session_state["run_face_stream"] = False

# --- Status Banner ---
if st.session_state["run_face_stream"]:
    st.success("Live detection running. Click Stop Camera in the sidebar to end.")
else:
    st.info("Press Start Camera in the sidebar to begin live detection.")

# --- WebRTC ICE Configuration (STUN + TURN) ---
# Replace values with your actual TURN credentials or subscribe to a reliable service (e.g., Xirsys, Twilio, etc)!
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},  # Free public STUN
        # Example TURN server (replace with your credentials):
        # {"urls": ["turn:turn.example.com:3478"], "username": "your_user", "credential": "your_pass"},
    ]
}

# --- Face Detection Processor ---
class FaceDetectionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        out_img = img.copy()
        try:
            crop = img[50:500, 50:500, :]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb_crop, (120, 120))
            input_arr = np.expand_dims(resized / 255.0, 0)
            yhat = model.predict(input_arr)
            sample_coords = yhat[1][0]
            if yhat[0][0] > 0.5:
                start_pt = (int(sample_coords[0]*450)+50, int(sample_coords[1]*450)+50)
                end_pt = (int(sample_coords[2]*450)+50, int(sample_coords[3]*450)+50)
                cv2.rectangle(out_img, start_pt, end_pt, (255, 0, 0), 2)
                cv2.putText(out_img, 'face', (start_pt[0], max(start_pt[1]-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        except Exception as e:
            pass  # Optional: log error or show feedback
        rgb_out = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(rgb_out, format="rgb24")

# --- Webcam Stream ---
if st.session_state["run_face_stream"] and model is not None:
    webrtc_streamer(
        key="face-detect-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceDetectionProcessor,
        rtc_configuration=rtc_configuration,  # <-- ENABLES TURN/STUN
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --- Sidebar Help ---
with st.sidebar.expander("ℹ️ How To Use", expanded=True):
    st.markdown("""
    **1. Click “Start Camera”** to begin live detection.  
    **2. Allow browser permission** for webcam access.  
    **3. Click “Stop Camera”** to end the stream safely.

    **Note:** In the cloud, TURN/STUN is needed for connection.  
    Your video stays local—no images are sent to any server but you!
    """)

st.sidebar.markdown("*Built with ❤️ using Streamlit & TensorFlow*")
