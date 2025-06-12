import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Constants
SOUND_SPEED = 343  # speed of sound in m/s
SAMPLERATE = 44100  # audio sampling rate in Hz

st.title("🔊 Sound Direction Detection Using Two Microphones")

st.markdown("""
This app uses **Time Difference of Arrival (TDOA)** between two microphones to estimate the **angle** and **direction** of any sound source.

1. Set up two microphones spaced apart. 
2. Record a clap or sound from various angles. 
3. Use cross-correlation to calculate time difference of arrival (TDOA). 
4. Calculate the angle of arrival based on microphone distance and TDOA. 
5. Display results in degrees or on a polar plot. 
""")

# Adjustable parameters
DURATION = st.slider("🎧 Recording Duration (seconds)", 0.5, 5.0, 1.0, step=0.1)
MIC_DISTANCE = st.slider("📏 Distance Between Microphones (meters)", 0.5, 5.0, 1.0, step=0.1)

# List available devices
devices = sd.query_devices()
input_devices = [
    {"name": d["name"], "index": i, "channels": d["max_input_channels"]}
    for i, d in enumerate(devices)
    if d["max_input_channels"] >= 1
]
device_labels = [f'{d["index"]}: {d["name"]} ({d["channels"]} channels)' for d in input_devices]
selected_label = st.selectbox("🎚️ Select Input Device", device_labels)
selected_device = input_devices[device_labels.index(selected_label)]

# Start recording
if st.button("🎙️ Record and Detect Sound"):
    st.info(f"Recording from `{selected_device['name']}` for {DURATION} seconds...")

    try:
        channels = 2 if selected_device["channels"] >= 2 else 1
        recording = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE,
                           channels=channels, device=selected_device["index"])
        sd.wait()

        if channels == 2:
            mic1 = recording[:, 0]
            mic2 = recording[:, 1]

            # Cross-correlation
            correlation = correlate(mic1, mic2, mode='full')
            lag = np.argmax(correlation) - len(mic1) + 1
            tdoa = lag / SAMPLERATE

            # Estimate direction
            direction = "RIGHT" if lag > 0 else "LEFT" if lag < 0 else "CENTER"

            # Calculate angle
            try:
                sin_theta = (tdoa * SOUND_SPEED) / MIC_DISTANCE
                sin_theta = np.clip(sin_theta, -1.0, 1.0)
                angle = np.degrees(np.arcsin(sin_theta))
            except ValueError:
                angle = None

            # Output
            st.write(f"🕒 TDOA: `{tdoa:.6f}` seconds")
            st.write(f"📍 Estimated Direction: **{direction}**")
            if angle is not None:
                st.success(f"Estimated Angle: `{angle:.2f}°`")
            else:
                st.error("Angle could not be calculated.")

            # Polar plot
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            if angle is not None:
                angle_rad = np.radians(angle)
                ax.plot([angle_rad, angle_rad], [0, 1], linewidth=3)
            ax.set_title("Estimated Sound Direction")
            st.pyplot(fig)

        else:
            st.warning("Only one input channel detected. Stereo input required.")
            st.line_chart(recording)

    except Exception as e:
        st.error(f"Error during recording: {e}")
