import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Constants
SOUND_SPEED = 343  # m/s
SAMPLERATE = 44100  # Hz

st.set_page_config(page_title="Sound Direction Detection", layout="centered")
st.title("🔊 Sound Direction Detection Using Two Microphones")

st.markdown("""
This app uses **Time Difference of Arrival (TDOA)** between two microphones to estimate the **angle** and **direction** of a sound source.

1. Requires a **2-channel** input device (stereo microphones).
2. Uses **cross-correlation** to calculate time delay.
3. Calculates **direction and angle of arrival**.
4. Plots the **sound waveform** and **polar direction**.
""")

# Adjustable parameters
DURATION = st.slider("🎧 Recording Duration (seconds)", 0.5, 5.0, 1.0, step=0.1)
MIC_DISTANCE = st.slider("📏 Distance Between Microphones (meters)", 0.5, 5.0, 1.0, step=0.1)

# List input devices
devices = sd.query_devices()
input_devices = [
    {"name": d["name"], "index": i, "channels": d["max_input_channels"]}
    for i, d in enumerate(devices)
    if d["max_input_channels"] >= 1
]
device_labels = [f'{d["index"]}: {d["name"]} ({d["channels"]} channels)' for d in input_devices]
selected_label = st.selectbox("🎚️ Select Input Device", device_labels)
selected_device = input_devices[device_labels.index(selected_label)]

# Record and process
if st.button("🎙️ Record and Detect Sound"):
    st.info(f"Recording from `{selected_device['name']}` for {DURATION:.1f} seconds...")

    try:
        channels = 2 if selected_device["channels"] >= 2 else 1
        recording = sd.rec(int(DURATION * SAMPLERATE),
                           samplerate=SAMPLERATE,
                           channels=channels,
                           device=selected_device["index"])
        sd.wait()

        if channels == 2:
            mic1 = recording[:, 0]
            mic2 = recording[:, 1]

            # Waveform plot
            st.subheader("🎵 Recorded Waveforms")
            time_axis = np.linspace(0, DURATION, len(mic1))

            fig_wave, ax_wave = plt.subplots()
            ax_wave.plot(time_axis, mic1, label="Mic 1", alpha=0.7)
            ax_wave.plot(time_axis, mic2, label="Mic 2", alpha=0.7)
            ax_wave.set_xlabel("Time (s)")
            ax_wave.set_ylabel("Amplitude")
            ax_wave.set_title("Recorded Sound Waveforms")
            ax_wave.legend()
            st.pyplot(fig_wave)

            # Cross-correlation
            correlation = correlate(mic1, mic2, mode='full')
            lag = np.argmax(correlation) - len(mic1) + 1
            tdoa = lag / SAMPLERATE
            direction = "RIGHT" if lag > 0 else "LEFT" if lag < 0 else "CENTER"

            # Angle
            try:
                sin_theta = (tdoa * SOUND_SPEED) / MIC_DISTANCE
                sin_theta = np.clip(sin_theta, -1.0, 1.0)
                angle = np.degrees(np.arcsin(sin_theta))
            except ValueError:
                angle = None

            # Output
            st.subheader("📈 Direction Estimation")
            st.write(f"🕒 TDOA: `{tdoa:.6f}` seconds")
            st.write(f"📍 Estimated Direction: **{direction}**")
            if angle is not None:
                st.success(f"Estimated Angle: `{angle:.2f}°`")
            else:
                st.error("Could not compute angle — invalid TDOA.")

            # Polar plot
            fig_polar, ax_polar = plt.subplots(subplot_kw={'projection': 'polar'})
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)
            if angle is not None:
                angle_rad = np.radians(angle)
                ax_polar.plot([angle_rad, angle_rad], [0, 1], linewidth=3)
            ax_polar.set_title("Estimated Sound Direction")
            st.pyplot(fig_polar)

        else:
            st.warning("Only one channel detected. This app needs stereo (2-channel) input.")
            st.line_chart(recording)

    except Exception as e:
        st.error(f"Error during recording: {e}")
