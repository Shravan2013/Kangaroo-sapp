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
# JS AUTOPLAY INJECTOR
# ----------------------------
def js_play_audio(file_path):
    """Injects JS to force-play an audio clip (bypasses browser autoplay blocking)."""
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        unique = str(time.time()).replace(".", "")
        js_code = f"""
        <script>
        (async () => {{
            const existing = document.getElementById("audio_{unique}");
            if (existing) existing.remove();

            const audio = document.createElement('audio');
            audio.id = "audio_{unique}";
            audio.src = "data:audio/mp3;base64,{b64}";
            audio.autoplay = true;
            audio.volume = 1.0;
            document.body.appendChild(audio);
            try {{
                await audio.play();
                console.log("Audio played ✅");
            }} catch (e) {{
                console.warn("Autoplay blocked ❌", e);
            }}
        }})();
        </script>
        """
        st.components.v1.html(js_code, height=0)
    except Exception as e:
        st.error(f"❌ Error playing {file_path}: {e}")

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
st.markdown("### Detects people and plays sound alerts intelligently")

# --- Config Controls ---
st.sidebar.header("⚙️ Detection Settings")
stability_threshold = st.sidebar.slider("Stability Threshold (frames)", 1, 10, 5)
cooldown = st.sidebar.slider("Audio Cooldown (seconds)", 0.5, 5.0, 1.0, step=0.1)

st.sidebar.info(
    "➡️ Higher *stability threshold* = fewer false audio changes.\n"
    "➡️ Longer *cooldown* = less frequent replays."
)

# --- WebRTC setup ---
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
    stability_placeholder = st.empty()
    status_placeholder = st.empty()

    # --- Stability variables ---
    stable_count = None
    candidate_count = None
    stability_counter = 0

    last_audio_time = 0

    while ctx.state.playing:
        if hasattr(ctx.video_processor, 'person_count'):
            current_count = ctx.video_processor.person_count
            count_placeholder.metric("People Detected", current_count)

            # --- Stabilization logic ---
            if candidate_count != current_count:
                candidate_count = current_count
                stability_counter = 1
            else:
                stability_counter += 1

            stability_placeholder.text(
                f"Stability: {stability_counter}/{stability_threshold} (Target: {candidate_count})"
            )

            # --- Only trigger if stable ---
            if stability_counter >= stability_threshold and stable_count != candidate_count:
                stable_count = candidate_count
                stability_counter = 0

                now = time.time()
                if now - last_audio_time >= cooldown:
                    last_audio_time = now

                    if stable_count > 0 and stable_count in AUDIO_FILES:
                        js_play_audio(AUDIO_FILES[stable_count])
                        status_placeholder.success(
                            f"🔊 Stable count: {stable_count} "
                            f"{'person' if stable_count == 1 else 'people'}"
                        )
                    else:
                        status_placeholder.info("👀 Waiting for people...")

        time.sleep(0.3)

else:
    st.info("👆 Click **START** to activate camera and audio")

st.markdown("---")
st.caption("Built with YOLOv8 + Streamlit + Stable Audio Logic 🎧")
