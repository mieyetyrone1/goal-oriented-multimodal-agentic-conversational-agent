import sounddevice as sd
import numpy as np

def record_until_enter(
    samplerate: int = 16000,
    channels: int = 1,
):
    print("Recording... press ENTER to stop")

    frames = []

    def callback(indata, frames_count, time, status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=callback,
    ):
        input()  # waits for ENTER

    audio = np.concatenate(frames, axis=0)
    return audio
