import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

st.title("🎧 Real-Time Audio Visualizer (Single Microphone)")

duration = st.slider("Recording duration (seconds)", 0.5, 3.0, 1.0, 0.1)
fs = 44100

def record_audio(duration, fs):
    st.info("Recording audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    st.success("Recording complete!")
    return audio.flatten()

if st.button("Record & Visualize"):
    audio = record_audio(duration, fs)
    
    # Volume (RMS)
    rms = np.sqrt(np.mean(audio**2))
    st.write(f"RMS Volume: {rms:.4f}")
    
    # Plot waveform
    fig_wf, ax_wf = plt.subplots()
    ax_wf.plot(np.linspace(0, duration, len(audio)), audio)
    ax_wf.set_title("Waveform")
    ax_wf.set_xlabel("Time [s]")
    ax_wf.set_ylabel("Amplitude")
    st.pyplot(fig_wf)
    
    # Spectrogram
    f, t, Sxx = spectrogram(audio, fs)
    fig_sp, ax_sp = plt.subplots()
    ax_sp.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    ax_sp.set_ylabel('Frequency [Hz]')
    ax_sp.set_xlabel('Time [s]')
    ax_sp.set_title("Spectrogram")
    st.pyplot(fig_sp)
