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
st.set_page_config(page_title="People Counter", layout="centered")

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
# AUDIO FILES
# ----------------------------
AUDIO_FILES = {
    1: "1_person.mp3",
    2: "2_people.mp3",
    3: "3_people.mp3",
    4: "4_people.mp3",
    5: "5_people.mp3"
}

# ----------------------------
# HTML AUDIO CONTROLLER (persistent)
# ----------------------------
def audio_controller():
    """Injects persistent JS controller once."""
    js_code = """
    <script>
    window.peopleAudio = window.peopleAudio || {
        el: null,
        currentSrc: null,
        playAudio(b64data) {
            if (!this.el) {
                this.el = document.createElement('audio');
                this.el.id = 'people_audio';
                this.el.autoplay = true;
                this.el.volume = 1.0;
                document.body.appendChild(this.el);
            }
            this.el.src = "data:audio/mp3;base64," + b64data;
            this.el.play().catch(e => console.warn("Autoplay blocked:", e));
            this.currentSrc = b64data;
        },
        stopAudio() {
            if (this.el) {
                this.el.pause();
                this.el.currentTime = 0;
                this.el.src = "";
                console.log("Audio stopped");
            }
        }
    };
    </script>
    """
    st.components.v1.html(js_code, height=0)

def play_audio(file_path):
    """Use existing JS audio controller to play sound."""
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()
    st.components.v1.html(f"<script>window.peopleAudio.playAudio('{b64}');</script>", height=0)

def stop_audio():
    """Stop currently playing sound."""
    st.components.v1.html("<script>window.peopleAudio.stopAudio();</script>", height=0)

# ----------------------------
# YOLO PERSON DETECTOR
# ----------------------------
class PersonDetector(VideoProcessorBase):
    def __init__(self):
        self.person_count = 0
        self.frame_count = 0
        self.detection_history = deque(maxlen=3)
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.frame_count % 2 == 0:
            results = model(
                img,
                verbose=False,
                imgsz=640,
                conf=0.4,
                device='cpu'
            )
            
            count = sum(int(box.cls[0]) == 0 for box in results[0].boxes)
            count = min(count, 5)
            
            self.detection_history.append(count)
            if self.detection_history:
                self.person_count = int(np.median(self.detection_history))
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("👥 People Counter")
st.markdown("### Detects people and plays sound alerts instantly")

# Load audio controller JS once
audio_controller()

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

if ctx.video_processor:
    count_placeholder = st.empty()
    status_placeholder = st.empty()
    last_played_count = 0
    
    while ctx.state.playing:
        if hasattr(ctx.video_processor, 'person_count'):
            current_count = ctx.video_processor.person_count
            
            count_placeholder.metric("People Detected", current_count)
            
            if current_count != last_played_count:
                if current_count > 0 and current_count in AUDIO_FILES:
                    play_audio(AUDIO_FILES[current_count])
                    status_placeholder.success(
                        f"🔊 Playing audio for {current_count} "
                        f"{'person' if current_count == 1 else 'people'}"
                    )
                else:
                    stop_audio()
                    status_placeholder.info("👀 Waiting for people...")
                last_played_count = current_count
        
        time.sleep(0.3)
else:
    st.info("👆 Click **START** to activate camera and audio")

st.markdown("---")
st.caption("Built with YOLOv8 + Streamlit + Persistent JS Audio 🎧")
