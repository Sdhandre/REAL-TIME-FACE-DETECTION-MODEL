import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import tensorflow as tf

#--- UI/UX Branding ---
st.set_page_config(page_title="Live Face Detection", page_icon=":smiley:", layout="wide")
st.markdown("<h1 style='text-align: center; color: #E36209;'>🤖 Live Face Detection Demo</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Webcam-based face detection powered by your custom ML model!</p>",
    unsafe_allow_html=True
)
st.sidebar.header("📷 Video Controls")

#--- Model (cached load) ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cpu_model2.h5')
model = load_model()

#--- App state for controlling stream ---
if 'run_face_stream' not in st.session_state:
    st.session_state['run_face_stream'] = False

#--- Sidebar Controls ---
start = st.sidebar.button("▶️ Start Camera", key="start")
stop = st.sidebar.button("⏹️ Stop Camera", key="stop")

if start:
    st.session_state['run_face_stream'] = True
if stop:
    st.session_state['run_face_stream'] = False

#--- Live Status Banner ---
if st.session_state['run_face_stream']:
    st.success("Live detection running. Click Stop Camera in the sidebar to end.")
else:
    st.info("Press Start Camera in the sidebar to begin live detection.")

#--- Main Processor Class ---
class FaceDetectionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # For model input: crop or resize as needed
        crop = img[50:500, 50:500, :]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb_crop, (120, 120))
        yhat = model.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]
        # For output: draw box on the original frame
        out_img = img.copy()
        if yhat[0] > 0.5:
            # Scale bbox to match crop and then map to the original frame
            start_pt = (int(sample_coords[0]*450) + 50, int(sample_coords[1]*450) + 50)
            end_pt = (int(sample_coords[2]*450) + 50, int(sample_coords[3]*450) + 50)
            cv2.rectangle(out_img, start_pt, end_pt, (255,0,0), 2)
            cv2.putText(out_img, 'face', (start_pt[0], start_pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        rgb_out = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        return av.VideoFrame.from_ndarray(rgb_out, format="rgb24")


#--- Start/Stop Stream Based on State ---
if st.session_state['run_face_stream']:
    webrtc_streamer(
        key="face-detect-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceDetectionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

#--- Help and Info ---
with st.sidebar.expander("ℹ️ How To Use", expanded=True):
    st.markdown("""
    **1. Click “Start Camera”** to begin live detection.
    **2. Allow browser permission** for webcam access.
    **3. Click “Stop Camera”** to end the stream safely.

    **Note:** Your video stays local—no images are sent to any server.
    """)

st.sidebar.markdown("*Built with ❤️ and Streamlit*")
