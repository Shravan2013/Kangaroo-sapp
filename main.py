import os
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
from ultralytics import YOLO
import numpy as np
from collections import deque
import time
import base64

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title=" ", layout="centered")

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.fuse()
    return model

model = load_model()

# ----------------------------
# AUDIO FILES - Preload as base64
# ----------------------------
@st.cache_data
def load_audio_base64():
    audio_data = {}
    for count, file_path in AUDIO_FILES.items():
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            audio_data[count] = base64.b64encode(audio_bytes).decode()
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    return audio_data

AUDIO_FILES = {
    1: "1_person.mp3",
    2: "2_people.mp3", 
    3: "3_people.mp3",
    4: "4_people.mp3",
    5: "5_people.mp3"
}

audio_base64 = load_audio_base64()

# ----------------------------
# AUDIO PLAYER COMPONENT
# ----------------------------
def audio_player(audio_key=None):
    """Creates an audio player that autoplays and auto-cleans up"""
    if audio_key and audio_key in audio_base64:
        # Create HTML that will autoplay and auto-remove the audio
        audio_html = f"""
        <audio id="peopleCounterAudio" autoplay onended="this.remove()">
            <source src="data:audio/mp3;base64,{audio_base64[audio_key]}" type="audio/mp3">
        </audio>
        <script>
            // Stop any other audio elements
            var allAudio = document.querySelectorAll('audio');
            allAudio.forEach(function(audio) {{
                if (audio.id !== 'peopleCounterAudio') {{
                    audio.pause();
                    audio.currentTime = 0;
                    audio.remove();
                }}
            }});
            
            // Auto-remove after 5 seconds max (safety)
            setTimeout(function() {{
                var audioElem = document.getElementById('peopleCounterAudio');
                if (audioElem) {{
                    audioElem.remove();
                }}
            }}, 5000);
        </script>
        """
        st.components.v1.html(audio_html, height=0)

# ----------------------------
# YOLO PERSON DETECTOR
# ----------------------------
class PersonDetector(VideoProcessorBase):
    def __init__(self):
        self.person_count = 0
        self.frame_count = 0
        self.detection_history = deque(maxlen=5)
        self.last_announced_count = 0
        self.cooldown_frames = 0
        self.cooldown_length = 15  # ~5 seconds at 3 FPS
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # Process every 2nd frame for performance
        if self.frame_count % 2 == 0:
            results = model(
                img,
                verbose=False,
                imgsz=640,
                conf=0.4,
                device='cpu'
            )
            
            # Count people (class 0 in COCO dataset)
            count = 0
            if results[0].boxes is not None:
                count = sum(int(box.cls[0]) == 0 for box in results[0].boxes)
            count = min(count, 5)
            
            self.detection_history.append(count)
            if self.detection_history:
                self.person_count = int(np.median(self.detection_history))
            
            # Update cooldown
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def can_announce(self, current_count):
        """Check if we can announce this count (not in cooldown and count changed)"""
        if (current_count != self.last_announced_count and 
            current_count > 0 and 
            self.cooldown_frames == 0):
            self.last_announced_count = current_count
            self.cooldown_frames = self.cooldown_length
            return True
        return False

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title(" ")
st.markdown(" ")

# Initialize session state
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = True
if 'last_ui_count' not in st.session_state:
    st.session_state.last_ui_count = 0

# Audio toggle
st.session_state.audio_enabled = st.checkbox(
    "Enable Audio Alerts", 
    value=st.session_state.audio_enabled,
    help="Play sound when people count changes"
)

ctx = webrtc_streamer(
    key="people-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PersonDetector,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False
    },
    async_processing=True,
)

st.markdown("---")

# Main processing loop
if ctx.video_processor:
    count_placeholder = st.empty()
    status_placeholder = st.empty()
    audio_placeholder = st.empty()
    
    while ctx.state.playing:
        if hasattr(ctx.video_processor, 'person_count'):
            current_count = ctx.video_processor.person_count
            
            # Update display
            count_placeholder.metric("People Detected", current_count)
            
            # Check if we should play audio
            should_play = (ctx.video_processor.can_announce(current_count) and 
                          st.session_state.audio_enabled)
            
            if should_play and current_count in audio_base64:
                # Use the audio player component
                with audio_placeholder:
                    audio_player(current_count)
                status_placeholder.success(
                    f"🔊 Announced: {current_count} "
                    f"{'person' if current_count == 1 else 'people'}"
                )
                st.session_state.last_ui_count = current_count
            elif current_count == 0 and st.session_state.last_ui_count != 0:
                st.session_state.last_ui_count = 0
                status_placeholder.info("👀 Waiting for people...")
            elif current_count == st.session_state.last_ui_count and current_count > 0:
                status_placeholder.info(f"✅ Tracking {current_count} {'person' if current_count == 1 else 'people'}")
        
        time.sleep(0.3)
else:
    st.info("👆 Click **START** to activate camera and audio")

st.markdown("---")
st.caption(" ")
