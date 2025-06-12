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

st.title("🔊 Direction-of-Arrival Estimation")
st.write("Estimate the angle of arrival of a sound using two microphones.")

# Get stereo input devices (at least 2 channels)
all_devices = sd.query_devices()
input_devices = [
    {"name": d["name"], "index": i}
    for i, d in enumerate(all_devices)
    if d["max_input_channels"] >= 2
]

# Dropdown list
device_names = [f'{d["index"]}: {d["name"]}' for d in input_devices]
selected_device_label = st.selectbox("🎚️ Select Stereo Microphone Device", device_names)

# Extract actual device index from selection
selected_device_index = int(selected_device_label.split(":")[0])

if st.button("🎙️ Record Sound"):
    st.info("Recording from selected stereo microphone for 1 second...")

    try:
        # Record from selected device
        recording = sd.rec(int(DURATION * SAMPLERATE),
                           samplerate=SAMPLERATE,
                           channels=2,
                           device=selected_device_index)
        sd.wait()

        # Separate channels
        mic1 = recording[:, 0]
        mic2 = recording[:, 1]

        # Cross-correlation to estimate TDOA
        correlation = correlate(mic1, mic2, mode='full')
        lag = np.argmax(correlation) - len(mic1) + 1
        tdoa = lag / SAMPLERATE

        # Calculate angle
        try:
            sin_theta = (tdoa * SOUND_SPEED) / MIC_DISTANCE
            sin_theta = np.clip(sin_theta, -1.0, 1.0)
            angle = np.degrees(np.arcsin(sin_theta))
        except ValueError:
            angle = None

        # Display results
        st.write(f"Estimated TDOA: {tdoa:.6f} s")
        if angle is not None:
            st.success(f"Estimated Angle of Arrival: {angle:.2f}°")
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

    except Exception as e:
        st.error(f"Error recording audio: {e}")
