import logging 

LOG_FILE_NAME = "test.log"


# get a logger object
logger = logging.getLogger(__name__)
# set handler to tell where to write logs
handler = logging.FileHandler(LOG_FILE_NAME)
# set the format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# set formatter to handler
handler.setFormatter(formatter)
# add handler to logger
logger.addHandler(handler)

# set the level
logger.setLevel(logging.INFO)

# log in code
logger.info("custom test logger is working")