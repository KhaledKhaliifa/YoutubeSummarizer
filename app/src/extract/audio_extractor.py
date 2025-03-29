import ffmpeg
from ..utils.logger import logger

def extract_audio(video_path, audio_path):
    try:
        ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
        logger.info(f"Audio extracted to {audio_path}")
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        raise e
