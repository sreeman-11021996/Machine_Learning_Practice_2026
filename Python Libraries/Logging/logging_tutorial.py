import logging

# logging.debug
# logging.info
# -----------from this level log to console--------------
# logging.warning 
# logging.error
# logging.critical

logging.basicConfig(level=logging.INFO, filename="log.log",filemode='w',
                    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# this basicConfig is to be run once for the project only

try:
    1/0
except Exception as e:
    # logging.error("ZeroDivisionError",exc_info=True)
    logging.exception("test")