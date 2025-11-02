import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO
import numpy as np
from collections import deque
import time

# Page config
st.set_page_config(page_title="People Counter", layout="centered")

# Initialize YOLO model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.fuse()
    return model

model = load_model()

# Audio file mapping
AUDIO_FILES = {
    1: "1_person.mp3",
    2: "2_people.mp3",
    3: "3_people.mp3",
    4: "4_people.mp3",
    5: "5_people.mp3"
}

# Helper function to safely play audio
def play_audio(file_path, placeholder, key_prefix="audio"):
    """Play an audio file using a unique Streamlit key each time."""
    try:
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        placeholder.audio(
            audio_bytes,
            format='audio/mp3',
            autoplay=True,
            key=f"{key_prefix}_{time.time()}"
        )
    except Exception as e:
        st.error(f"❌ Error playing {file_path}: {e}")

# Video processor class
class PersonDetector(VideoProcessorBase):
    def __init__(self):
        self.person_count = 0
        self.frame_count = 0
        self.detection_history = deque(maxlen=3)  # Smooth short-term variation
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Process every 2nd frame for faster inference
        if self.frame_count % 2 == 0:
            results = model(
                img,
                verbose=False,
                imgsz=640,
                conf=0.4,
                device='cpu'
            )
            
            # Count "person" detections
            count = 0
            for box in results[0].boxes:
                if int(box.cls[0]) == 0:  # class 0 = person
                    count += 1
            
            count = min(count, 5)
            self.detection_history.append(count)
            if len(self.detection_history) > 0:
                self.person_count = int(np.median(self.detection_history))
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit UI
st.title("👥 People Counter")
st.markdown("### Background detection with audio alerts")

# WebRTC streamer (video capture, no display)
ctx = webrtc_streamer(
    key="people-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PersonDetector,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480}
        },
        "audio": False
    },
    async_processing=True,
)

st.markdown("---")

# Status area
if ctx.video_processor:
    count_placeholder = st.empty()
    audio_placeholder = st.empty()
    status_placeholder = st.empty()
    
    last_played_count = 0
    
    # Live update loop
    while ctx.state.playing:
        if hasattr(ctx.video_processor, 'person_count'):
            current_count = ctx.video_processor.person_count
            
            # Display the live count
            count_placeholder.metric(
                label="People Detected",
                value=current_count
            )
            
            # Handle audio playback logic
            if current_count != last_played_count and current_count > 0:
                if current_count in AUDIO_FILES:
                    audio_file = AUDIO_FILES[current_count]
                    audio_placeholder.empty()
                    play_audio(audio_file, audio_placeholder)
                    last_played_count = current_count
                    status_placeholder.success(
                        f"🔊 Playing audio for {current_count} "
                        f"{'person' if current_count == 1 else 'people'}"
                    )
            elif current_count == 0 and last_played_count != 0:
                last_played_count = 0
                audio_placeholder.empty()
                status_placeholder.info("👀 Waiting for people...")
        
        time.sleep(0.3)
else:
    st.info("👆 Click **START** to begin detection")
    st.markdown("""
    **How it works:**
    - Uses your device camera (no video shown)
    - Detects how many people are visible
    - Plays audio alert when count changes
    """)

st.markdown("---")
st.caption("*Powered by YOLOv8 and Streamlit WebRTC*")
