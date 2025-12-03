import logging


def create_logger(logger_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(logger_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger