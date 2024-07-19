import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Settings
duration = 5  # seconds
sample_rate = 44100  # Hz

print("Recording...")
# Record audio
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
sd.wait()  # Wait until the recording is finished
print("Recording finished.")

# Save the audio as a WAV file
write('output.wav', sample_rate, audio)

print("Audio saved as 'output.wav'.")
