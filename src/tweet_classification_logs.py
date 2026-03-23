import logging
import config 

log_dir = config.LOG_DIR 
log_file = log_dir/"tweet.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
 