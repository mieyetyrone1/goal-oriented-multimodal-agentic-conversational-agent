import numpy as np
from faster_whisper import WhisperModel


class WhisperSTT:
    """
    Speech-to-text using faster-whisper.
    Designed for short, push-to-talk utterances.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000,
    ):
        self.sample_rate = sample_rate
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """
        audio: np.ndarray of shape (num_samples, 1) or (num_samples,)
        returns: transcribed text
        """

        if audio.ndim == 2:
            audio = audio.squeeze(axis=1)

        # Ensure float32 in [-1, 1]
        audio = audio.astype(np.float32)

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True,
        )

        text_parts = [seg.text.strip() for seg in segments]
        return " ".join(text_parts)
