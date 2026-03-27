import logging
from src import config 

def build_logger(name):
    log_dir = config.LOG_DIR 
    log_file = log_dir/"tweet.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(filename=log_file, mode="a", encoding="utf-8")
    formatting = logging.Formatter("%(asctime)s -> %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)

    file_handler.setFormatter(formatting)
    console_handler.setFormatter(formatting)
    
    return logger, file_handler, console_handler