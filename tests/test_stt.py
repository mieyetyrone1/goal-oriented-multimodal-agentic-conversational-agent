"""
Use the command below to test
py -m tests.test_stt
"""
from src.speech.audio_controller import AudioController
from src.speech.mic_capture import record_until_enter
from src.speech.stt import WhisperSTT


audio = AudioController()
stt = WhisperSTT(model_size="base")

input("Press ENTER to start talking")
audio.begin_listening()

audio_data = record_until_enter()

audio.end_listening()

text = stt.transcribe(audio_data)
print("Transcription:", text)
