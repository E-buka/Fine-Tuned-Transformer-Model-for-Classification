from src.tweet_logger import build_logger 
import os 

filename = os.path.basename(__file__)
logger, file_handler, console_handler = build_logger(filename)

if __name__ == "__main__":
    logger.addHandler(file_handler)
    now =5
    then = "this"
    
    logger.info(f"last run {now} and {then}")
    logger.removeHandler(file_handler)