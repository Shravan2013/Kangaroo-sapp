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

st.set_page_config(page_title="People Counter", layout="centered")

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    model.fuse()
    return model

model = load_model()

AUDIO_FILES = {
    1: "1_person.mp3",
    2: "2_people.mp3",
    3: "3_people.mp3",
    4: "4_people.mp3",
    5: "5_people.mp3"
}

# -----------------------------------------------------------------
# 🔊 Inject global JS controller ONCE at page load
# -----------------------------------------------------------------
st.markdown("""
<script>
window.peopleAudio = new Audio();
window.peopleAudio.volume = 1.0;
window.playPeopleAudio = function(base64data) {
    try {
        window.peopleAudio.pause();
        window.peopleAudio.src = "data:audio/mp3;base64," + base64data;
        window.peopleAudio.currentTime = 0;
        window.peopleAudio.play().catch(e => console.warn("Autoplay blocked", e));
    } catch(e) { console.error("Play error", e); }
};
window.stopPeopleAudio = function() {
    try {
        window.peopleAudio.pause();
        window.peopleAudio.currentTime = 0;
    } catch(e) { console.error("Stop error", e); }
};
</script>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------
# Helper to call the JS functions
# -----------------------------------------------------------------
def play_audio(file_path):
    """Convert audio to base64 and trigger playback via JS."""
    try:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        js = f"<script>window.playPeopleAudio('{b64}');</script>"
        st.components.v1.html(js, height=0)
    except Exception as e:
        st.error(f"❌ Play error: {e}")

def stop_audio():
    """Stops the global audio."""
    st.components.v1.html("<script>window.stopPeopleAudio();</script>", height=0)

# -----------------------------------------------------------------
# YOLO detector
# -----------------------------------------------------------------
class PersonDetector(VideoProcessorBase):
    def __init__(self):
        self.person_count = 0
        self.frame_count = 0
        self.detection_history = deque(maxlen=3)
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            results = model(img, verbose=False, imgsz=640, conf=0.4, device="cpu")
            count = sum(int(box.cls[0]) == 0 for box in results[0].boxes)
            count = min(count, 5)
            self.detection_history.append(count)
            if self.detection_history:
                self.person_count = int(np.median(self.detection_history))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------
st.title("👥 People Counter")
st.markdown("### Detects people and auto-plays corresponding sounds")

st.sidebar.header("⚙️ Detection Settings")
stability_threshold = st.sidebar.slider("Stability Threshold", 1, 10, 5)
cooldown = st.sidebar.slider("Cooldown (seconds)", 0.5, 5.0, 1.0, step=0.1)

ctx = webrtc_streamer(
    key="people-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PersonDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")

if ctx.video_processor:
    count_placeholder = st.empty()
    status_placeholder = st.empty()

    stable_count = None
    candidate_count = None
    stability_counter = 0
    last_audio_time = 0

    while ctx.state.playing:
        if hasattr(ctx.video_processor, "person_count"):
            current_count = ctx.video_processor.person_count
            count_placeholder.metric("People Detected", current_count)

            # stabilization
            if candidate_count != current_count:
                candidate_count = current_count
                stability_counter = 1
            else:
                stability_counter += 1

            if stability_counter >= stability_threshold and stable_count != candidate_count:
                stable_count = candidate_count
                stability_counter = 0
                now = time.time()

                if now - last_audio_time >= cooldown:
                    last_audio_time = now
                    if stable_count > 0 and stable_count in AUDIO_FILES:
                        play_audio(AUDIO_FILES[stable_count])
                        status_placeholder.success(
                            f"🔊 {stable_count} {'person' if stable_count == 1 else 'people'} detected"
                        )
                    else:
                        stop_audio()
                        status_placeholder.info("👀 No people detected")
        time.sleep(0.3)
else:
    st.info("👆 Click **START** to activate camera")

st.markdown("---")
st.caption("💡 Stable autoplay + stop audio fixed version")
