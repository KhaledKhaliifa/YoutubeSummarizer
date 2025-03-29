from pytubefix import YouTube
from ..utils.logger import logger
# where to save 
SAVE_PATH = "./data/raw_videos"

def download_video(url, filename):
    try: 
        # object creation using YouTube 
        yt = YouTube(url) 
    except Exception as e: 
        #to handle exception 
        logger.error("Connection Error")
        logger.error(e) 

    # Get all streams and filter for mp4 files
    mp4_streams = yt.streams.filter(progressive=True, res='360p').first()

    try: 
        # downloading the video 
        mp4_streams.download(output_path=SAVE_PATH, filename=f"{filename}.mp4")
        logger.info('Video downloaded successfully!')
        return f"{SAVE_PATH}/{filename}.mp4"
    except: 
        logger.error("Error downloading video.")
        