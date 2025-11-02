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
# JS AUDIO MANAGER
# ----------------------------
class AudioManager:
    def __init__(self):
        self.current_audio_id = None
        
    def play_audio(self, file_path):
        """Injects JS to play audio and stops any currently playing audio"""
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()
            b64 = base64.b64encode(audio_bytes).decode()
            unique_id = str(time.time()).replace(".", "")
            
            # Stop previous audio first
            if self.current_audio_id:
                stop_js = f"""
                <script>
                const prevAudio = document.getElementById("audio_{self.current_audio_id}");
                if (prevAudio) {{
                    prevAudio.pause();
                    prevAudio.currentTime = 0;
                    prevAudio.remove();
                }}
                </script>
                """
                st.components.v1.html(stop_js, height=0)
            
            # Play new audio
            play_js = f"""
            <script>
            (async () => {{
                const audio = document.createElement('audio');
                audio.id = "audio_{unique_id}";
                audio.src = "data:audio/mp3;base64,{b64}";
                audio.autoplay = true;
                audio.volume = 1.0;
                
                // Auto-remove when finished playing
                audio.onended = function() {{
                    audio.remove();
                }};
                
                document.body.appendChild(audio);
                try {{
                    await audio.play();
                    console.log("Audio played ✅");
                }} catch (e) {{
                    console.warn("Autoplay blocked ❌", e);
                    audio.remove();
                }}
            }})();
            </script>
            """
            st.components.v1.html(play_js, height=0)
            self.current_audio_id = unique_id
            
        except Exception as e:
            st.error(f"❌ Error playing {file_path}: {e}")

# Initialize audio manager
audio_manager = AudioManager()

# ----------------------------
# YOLO PERSON DETECTOR
# ----------------------------
class PersonDetector(VideoProcessorBase):
    def __init__(self):
        self.person_count = 0
        self.frame_count = 0
        self.detection_history = deque(maxlen=5)  # Increased for better stability
        self.last_played_count = 0
        self.last_play_time = 0
        self.play_cooldown = 2.0  # Minimum seconds between audio plays
        
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
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("👥 People Counter")
st.markdown("### Detects people and plays sound alerts instantly")

# Initialize session state for persistence
if 'last_displayed_count' not in st.session_state:
    st.session_state.last_displayed_count = 0
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = True

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

if ctx.video_processor:
    count_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Use session state to persist across reruns
    last_played_count = st.session_state.last_displayed_count
    
    while ctx.state.playing:
        if hasattr(ctx.video_processor, 'person_count'):
            current_count = ctx.video_processor.person_count
            
            # Update display
            count_placeholder.metric("People Detected", current_count)
            
            # Check if count changed and cooldown period passed
            current_time = time.time()
            should_play_audio = (
                current_count != last_played_count and 
                current_count > 0 and
                st.session_state.audio_enabled and
                (current_time - ctx.video_processor.last_play_time) > ctx.video_processor.play_cooldown
            )
            
            if should_play_audio:
                if current_count in AUDIO_FILES:
                    try:
                        audio_manager.play_audio(AUDIO_FILES[current_count])
                        status_placeholder.success(
                            f"🔊 Played audio for {current_count} "
                            f"{'person' if current_count == 1 else 'people'}"
                        )
                        ctx.video_processor.last_play_time = current_time
                        last_played_count = current_count
                        st.session_state.last_displayed_count = current_count
                    except Exception as e:
                        status_placeholder.error(f"❌ Audio error: {e}")
            elif current_count == 0 and last_played_count != 0:
                last_played_count = 0
                st.session_state.last_displayed_count = 0
                status_placeholder.info("👀 Waiting for people...")
            elif current_count == last_played_count and current_count > 0:
                status_placeholder.info(f"✅ Tracking {current_count} {'person' if current_count == 1 else 'people'}")
        
        time.sleep(0.3)
else:
    st.info("👆 Click **START** to activate camera and audio")

st.markdown("---")
st.caption("Built with YOLOv8 + Streamlit + JS Audio Hack 💪")
