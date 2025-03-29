import logging
import os
from datetime import datetime

class Logger:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not Logger._initialized:
            # Create logs directory if it doesn't exist
            self.log_dir = "logs"
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            # Create a unique log file for each day
            current_date = datetime.now().strftime("%Y-%m-%d")
            self.log_file = os.path.join(self.log_dir, f"app_{current_date}.log")

            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
                handlers=[
                    logging.FileHandler(self.log_file),
                    logging.StreamHandler()
                ]
            )

            # Create logger instance
            self.logger = logging.getLogger("video_summarizer")
            Logger._initialized = True

    def get_logger(self):
        return self.logger

# Create a global logger instance
logger = Logger().get_logger()
