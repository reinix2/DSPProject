import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from math import asin, degrees
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import sounddevice as sd
st.write(sd.query_devices())

# Constants
MIC_DISTANCE = 0.2  # meters between microphones
SPEED_OF_SOUND = 343  # m/s
SAMPLE_RATE = 48000  # typical webrtc sample rate

st.title("🎤 Real-Time Direction-of-Arrival Estimation")

class DoAProcessor(AudioProcessorBase):
    def __init__(self):
        self.sample_rate = SAMPLE_RATE

    def recv(self, frame):
        audio = frame.to_ndarray(format="flt32")
        # audio shape: (samples, channels), expect stereo: channels=2
        if audio.shape[1] < 2:
            return frame  # need 2 channels
        
        left = audio[:, 0]
        right = audio[:, 1]

        # Normalize
        left = left / np.max(np.abs(left)) if np.max(np.abs(left)) != 0 else left
        right = right / np.max(np.abs(right)) if np.max(np.abs(right)) != 0 else right

        # Cross-correlation
        corr = correlate(left, right, mode='full')
        lags = np.arange(-len(left) + 1, len(left))
        lag = lags[np.argmax(corr)]
        tdoa = lag / self.sample_rate

        # Calculate angle if valid
        try:
            val = tdoa * SPEED_OF_SOUND / MIC_DISTANCE
            if abs(val) <= 1:
                angle_rad = asin(val)
                angle_deg = degrees(angle_rad)
            else:
                angle_deg = None
        except Exception:
            angle_deg = None

        # Save angle for Streamlit display
        st.session_state.angle = angle_deg
        st.session_state.tdoa = tdoa

        return frame

webrtc_streamer(key="doa", audio_processor_factory=DoAProcessor,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True)

# Display results
if "angle" in st.session_state:
    tdoa = st.session_state.tdoa
    angle = st.session_state.angle

    st.write(f"🕒 **TDOA**: {tdoa*1e6:.1f} microseconds")
    if angle is not None:
        st.success(f"📐 Estimated angle of arrival: {angle:.1f}°")

        # Polar plot
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location('front')
        ax.set_theta_direction(-1)
        ax.plot([0, np.deg2rad(angle)], [0, 1], color='magenta', linewidth=3)
        ax.set_yticklabels([])
        ax.set_title("Estimated Sound Direction")
        st.pyplot(fig)
    else:
        st.warning("Angle estimation out of range or invalid.")
else:
    st.info("Make a sound near your stereo microphone (e.g., AirPods).")
