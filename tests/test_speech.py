"""
Use the command below to test
py -m tests.test_speech
"""
from src.speech.audio_controller import AudioController
from src.speech.mic_capture import record_until_enter

audio = AudioController()

while True:
    input("Press ENTER to start talking (Ctrl+C to quit)")
    audio.begin_listening()

    audio_data = record_until_enter()

    audio.end_listening()

    print(f"Captured audio shape: {audio_data.shape}")
