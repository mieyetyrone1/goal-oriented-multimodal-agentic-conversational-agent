import pyttsx3
from typing import Optional


class WindowsTTS:
    """
    Text-to-speech using Windows pyttsx3 voices.
    Works with AudioController to ensure half-duplex.
    """

    def __init__(self, voice_name: Optional[str] = None, rate: int = 150, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        if voice_name:
            voices = self.engine.getProperty("voices")
            for v in voices:
                if voice_name.lower() in v.name.lower():
                    self.engine.setProperty("voice", v.id)
                    break

    def speak(self, text: str):
        """
        Speak the given text (blocking).
        """
        if not text.strip():
            return

        self.engine.say(text)
        self.engine.runAndWait()
