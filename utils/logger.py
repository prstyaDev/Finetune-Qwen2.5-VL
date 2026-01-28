import logging
import colorlog


def get_logger(level=logging.DEBUG):
    logger = logging.getLogger("VLM Finetune Experiment")
    logger.setLevel(level)

    log_colors = {
        "DEBUG": "purple",
        "INFO": "yellow",
        "WARNING": "light_white",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

    format = colorlog.ColoredFormatter(
        fmt="%(log_color)s[%(asctime)-15s] [%(levelname)8s]%(reset)s: %(message)s",
        log_colors=log_colors,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(format)
    logger.addHandler(handler)
    return logger


logger = get_logger()
