import logging 

logging.basicConfig(
    format='[%(levelname)s] %(filename)s:%(lineno)d: %(message)s',
    level=logging.INFO
)

logger = logging.getLogger("default")
