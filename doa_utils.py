import numpy as np
from scipy.signal import correlate
import sounddevice as sd

def record_audio(duration=1, fs=44100, channels=2):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    print("Recording done.")
    return audio, fs

def calculate_tdoa(signal, fs):
    sig1 = signal[:, 0]
    sig2 = signal[:, 1]
    
    corr = correlate(sig1, sig2, mode='full')
    lag = np.argmax(corr) - (len(sig1) - 1)
    tdoa = lag / fs
    return tdoa

def calculate_angle(tdoa, mic_distance, speed_of_sound=343.0):
    max_tdoa = mic_distance / speed_of_sound
    if abs(tdoa) > max_tdoa:
        return None  # physically impossible delay
    angle_rad = np.arcsin(tdoa * speed_of_sound / mic_distance)
    angle_deg = np.degrees(angle_rad)
    return angle_deg
