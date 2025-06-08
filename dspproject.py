import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from doa_utils import record_audio, calculate_tdoa, calculate_angle

st.set_page_config(page_title="Real-Time DoA Estimation", layout="centered")

st.title("🎧 Real-Time Direction-of-Arrival (DoA) Estimation")
st.markdown("""
Estimate the direction of a sound source using two microphones.

📌 **Instructions:**
- Ensure your input device provides **stereo** audio.
- Place two microphones at a fixed known distance (e.g., 0.2 meters).
- Click **Record**, make a **clap or sound**, and see the estimated direction.
""")

mic_distance = st.slider("Microphone Distance (meters)", 0.01, 1.0, 0.2, 0.01)
duration = st.slider("Recording Duration (seconds)", 0.5, 3.0, 1.0, 0.1)

if st.button("🎙️ Record & Estimate Direction"):
    try:
        signal, fs = record_audio(duration=duration, channels=2)
        tdoa = calculate_tdoa(signal, fs)
        angle = calculate_angle(tdoa, mic_distance)

        if angle is None:
            st.error("Invalid TDOA. Try again with cleaner input.")
        else:
            st.success(f"Estimated Angle: **{angle:.2f}°**")

            # Polar Plot
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            theta = np.deg2rad(angle)
            ax.plot([theta, theta], [0, 1], color='r', linewidth=3)
            ax.set_ylim(0, 1)
            ax.set_title("Direction of Arrival", va='bottom')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during recording or processing: {str(e)}")
