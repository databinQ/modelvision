# coding: utf-8

import os
import logging

from constants import LOG_PATH


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_dir = log_dir if log_dir else LOG_PATH
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    std_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, name + ".log"), encoding="utf-8")

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]%(message)s")
    std_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(std_handler)
    logger.addHandler(file_handler)
    return logger


base_logger = get_logger("base")
