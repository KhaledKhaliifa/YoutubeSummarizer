import whisper
from ..utils.logger import logger

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise e
