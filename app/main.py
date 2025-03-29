import streamlit as st
import os
from src.extract.video_downloader import download_video
from src.extract.audio_extractor import extract_audio
from src.extract.speech_to_text import transcribe_audio
from src.generate.summarizer import Summarizer
from src.utils.logger import logger

# Set page config
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("YouTube Video Summarizer")
st.markdown("""
This application takes a YouTube video URL, extracts its audio, converts speech to text, 
and generates a concise summary of the content.
""")

# Create necessary directories if they don't exist
os.makedirs("./data/raw_videos", exist_ok=True)
os.makedirs("./data/audio", exist_ok=True)

def process_video(url):
    try:
        # Generate a unique filename based on video ID
        video_id = url.split("v=")[1].split("&")[0]
        video_path = f"./data/raw_videos/{video_id}.mp4"
        audio_path = f"./data/audio/{video_id}.mp3"
        
        # Download video
        with st.spinner("Downloading video..."):
            download_video(url, video_id)
            st.success("Video downloaded successfully!")
        
        # Extract audio
        with st.spinner("Extracting audio..."):
            extract_audio(video_path, audio_path)
            st.success("Audio extracted successfully!")
        
        # Transcribe audio
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(audio_path)
            st.success("Audio transcribed successfully!")
        
        # Generate summary
        with st.spinner("Generating summary..."):
            summarizer = Summarizer("t5-base")
            summary = summarizer.summarize(transcription)
            
            # Display summary in bullet points
            st.markdown("### Summary")
            # Split summary into points and display
            points = summary.split('.')
            for point in points:
                if point.strip():
                    st.markdown(f"• {point.strip()}.")
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        st.error(f"An error occurred: {str(e)}")

# URL input
url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")

if url:
    if "youtube.com" in url or "youtu.be" in url:
        process_video(url)
    else:
        st.error("Please enter a valid YouTube URL") 