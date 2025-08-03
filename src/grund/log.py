import logging


# implement singleton pattern for logger
ROOT_LOGGER_SET = False
STDOUT_FORMAT = "%(asctime)s %(name)-4s %(levelname)-4s %(funcName)-8s %(message)s"


def set_root_logger(verbose: bool = False):
    global ROOT_LOGGER_SET
    if not ROOT_LOGGER_SET:
        ROOT_LOGGER_SET = True
        log_level = logging.DEBUG if verbose else logging.INFO
        logger = logging.getLogger()
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        logger.addHandler(handler)
        formatter = logging.Formatter(STDOUT_FORMAT)
        handler.setFormatter(formatter)