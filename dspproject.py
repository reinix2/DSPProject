import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Constants
SOUND_SPEED = 343  # m/s
MIC_DISTANCE = 0.2  # meters
DURATION = 1  # seconds
SAMPLERATE = 44100  # Hz

st.title("🔊 Direction-of-Arrival Estimation Using Microphone Array")
st.markdown("""
**Objective**: Estimate the direction from which a sound is coming using time delays between microphones.

**Instructions**:
- Set up two microphones spaced apart.
- Record a clap or sound from various angles.
- Use cross-correlation to calculate TDOA.
- Calculate and display the angle of arrival and direction.
""")

# Get all input devices (including mono)
all_devices = sd.query_devices()
input_devices = [
    {"name": d["name"], "index": i, "channels": d["max_input_channels"]}
    for i, d in enumerate(all_devices)
    if d["max_input_channels"] >= 1
]

device_labels = [f'{d["index"]}: {d["name"]} ({d["channels"]} channels)' for d in input_devices]
selected_label = st.selectbox("🎚️ Select Input Device", device_labels)
selected_device = input_devices[device_labels.index(selected_label)]

if st.button("🎙️ Record Sound"):
    st.info(f"Recording from: {selected_device['name']}...")

    try:
        channels = 2 if selected_device["channels"] >= 2 else 1
        recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE,
                           channels=channels, device=selected_device["index"])
        sd.wait()

        if channels == 2:
            mic1 = recording[:, 0]
            mic2 = recording[:, 1]

            # Cross-correlation for TDOA
            correlation = correlate(mic1, mic2, mode='full')
            lag = np.argmax(correlation) - len(mic1) + 1
            tdoa = lag / SAMPLERATE

            # Estimate direction
            if lag > 0:
                direction = "RIGHT"
            elif lag < 0:
                direction = "LEFT"
            else:
                direction = "CENTER"

            # Calculate angle
            try:
                sin_theta = (tdoa * SOUND_SPEED) / MIC_DISTANCE
                sin_theta = np.clip(sin_theta, -1.0, 1.0)
                angle = np.degrees(np.arcsin(sin_theta))
            except ValueError:
                angle = None

            # Output
            st.write(f"🕒 TDOA: `{tdoa:.6f}` seconds")
            st.write(f"📍 Sound Source Direction: **{direction}**")
            if angle is not None:
                st.success(f"Estimated Angle of Arrival: `{angle:.2f}°`")
            else:
                st.error("Could not calculate angle due to invalid TDOA.")

            # Polar plot
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            if angle is not None:
                angle_rad = np.radians(angle)
                ax.plot([angle_rad, angle_rad], [0, 1], linewidth=3)
            ax.set_title("Estimated Direction")
            st.pyplot(fig)

        else:
            st.warning("Only one channel detected. Direction-of-arrival estimation requires stereo input.")
            st.line_chart(recording)

    except Exception as e:
        st.error(f"Recording error: {e}")
