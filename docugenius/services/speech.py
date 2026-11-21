"""Speech-to-text using the ``SpeechRecognition`` library (Google Web Speech API).

WAV files are decoded natively. Other formats (MP3, M4A, OGG, WEBM, ...) are
converted with ``pydub``, which needs FFmpeg on the ``PATH``; the Docker image
ships with it.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from io import BytesIO
from typing import Any

from docugenius.errors import TranscriptionError

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 5 * 60


class Transcriber:
    def __init__(self, recognizer: Any | None = None) -> None:
        self._recognizer = recognizer

    @property
    def recognizer(self) -> Any:
        if self._recognizer is None:
            import speech_recognition as sr

            self._recognizer = sr.Recognizer()
        return self._recognizer

    @staticmethod
    def ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    def transcribe(self, data: bytes, language: str = "en-US") -> str:
        import speech_recognition as sr

        wav = self._to_wav(data)
        try:
            with sr.AudioFile(BytesIO(wav)) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio, language=language)
        except sr.UnknownValueError as exc:
            raise TranscriptionError("No intelligible speech was detected in this recording.") from exc
        except sr.RequestError as exc:
            logger.warning("Speech recognition service error: %s", exc)
            raise TranscriptionError(
                "The speech recognition service could not be reached. Please try again later.",
                status_code=502,
            ) from exc
        return str(text).strip()

    def _to_wav(self, data: bytes) -> bytes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # pydub warns when FFmpeg is absent
            from pydub import AudioSegment
            from pydub.exceptions import CouldntDecodeError

        is_wav = data[:4] == b"RIFF" and data[8:12] == b"WAVE"
        if not is_wav and not self.ffmpeg_available():
            raise TranscriptionError(
                "Only WAV audio can be processed because FFmpeg is not installed on the server. "
                "Install FFmpeg or upload a .wav file.",
                status_code=415,
            )
        try:
            segment = AudioSegment.from_file(BytesIO(data), format="wav" if is_wav else None)
        except (CouldntDecodeError, IndexError, OSError) as exc:
            raise TranscriptionError("This audio file could not be decoded. Try WAV, MP3 or M4A.") from exc

        if segment.duration_seconds > MAX_DURATION_SECONDS:
            raise TranscriptionError(
                f"Audio longer than {MAX_DURATION_SECONDS // 60} minutes is not supported.", status_code=413
            )
        buffer = BytesIO()
        segment.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1).set_sample_width(2).export(buffer, format="wav")
        return buffer.getvalue()
