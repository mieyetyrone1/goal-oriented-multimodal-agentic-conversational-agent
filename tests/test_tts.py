"""
Use the command below to test
py -m tests.test_tts
"""
from src.speech.audio_controller import AudioController
from src.speech.tts import WindowsTTS

audio = AudioController()
tts = WindowsTTS(rate=150)

text = "Hello! This is a test of the Windows text to speech system."

audio.begin_speaking()
tts.speak(text)
audio.end_speaking()
