import sounddevice as sd
import numpy as np
import aubio

# Configuration
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNELS = 1

# Initialize pitch detection
pitch_detector = aubio.pitch("default", BUFFER_SIZE, BUFFER_SIZE // 2, SAMPLE_RATE)
pitch_detector.set_unit("Hz")
pitch_detector.set_tolerance(0.8)

# Desired scale for autotuning (example: A major scale frequencies)
scale = [440.0, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99]

def quantize_pitch(pitch):
    """Quantize pitch to nearest frequency in the scale."""
    return min(scale, key=lambda x: abs(x - pitch))

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    audio_data = indata[:, 0]  # Mono input
    detected_pitch = pitch_detector(audio_data)[0]
    
    if detected_pitch > 0:  # Valid pitch detected
        corrected_pitch = quantize_pitch(detected_pitch)
        print(f"Detected: {detected_pitch:.2f}, Corrected: {corrected_pitch:.2f}")
        # Apply pitch correction (here you would apply DSP, e.g., time-stretching)
        outdata[:] = indata  # For simplicity, pass input to output directly
    else:
        outdata[:] = indata

# Open real-time audio stream
with sd.Stream(samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=BUFFER_SIZE,
               callback=audio_callback):
    print("Real-time autotune is running... Press Ctrl+C to stop.")
    sd.sleep(30000)  # Run for 30 seconds

