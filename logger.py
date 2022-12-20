import logging
import os

from global_settings import LOGS_DIR


def create_logger():
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(fmt="%(name)s - %(levelname)s : %(message)s")

    main_file_handler = logging.FileHandler(LOGS_DIR + "mainLog.log", mode='at')
    main_file_handler.setFormatter(formatter)
    main_file_handler.setLevel("INFO")

    debug_file_handler = logging.FileHandler(LOGS_DIR + "last_run.log", mode='wt')
    debug_file_handler.setFormatter(formatter)
    debug_file_handler.setLevel(10)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(13)

    log = logging.Logger(f"{os.path.basename(__file__)}", level=10)
    # logging.addLevelName(11, "Debug11")
    logging.addLevelName(10, "  DebugL   ")
    logging.addLevelName(11, "  DebugHi  ")
    logging.addLevelName(12, "  DebugTop ")
    logging.addLevelName(13, "DebugResult")
    logging.addLevelName(14, " DebugWarn ")
    logging.addLevelName(15, "DebugError")

    log.propagate = True
    log.addHandler(main_file_handler)
    log.addHandler(debug_file_handler)
    log.addHandler(console_handler)

    return log


def create_stream_debug_logger():

    formatter = logging.Formatter(fmt="%(name)s - %(levelname)s : %(message)s")

    # main_file_handler = logging.FileHandler(LOGS_DIR + "mainLog.log", mode='at')
    # main_file_handler.setFormatter(formatter)
    # main_file_handler.setLevel("INFO")

    debug_file_handler = logging.FileHandler(LOGS_DIR + "last_run.log", mode='wt')
    debug_file_handler.setFormatter(formatter)
    debug_file_handler.setLevel(10)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(13)

    log = logging.Logger(f"TEST-LOGS", level=10)

    logging.addLevelName(10, "  DebugL   ")
    logging.addLevelName(11, "  DebugHi  ")
    logging.addLevelName(12, "  DebugTop ")
    logging.addLevelName(13, "DebugResult")
    logging.addLevelName(14, " DebugWarn ")
    logging.addLevelName(15, "DebugError")

    log.propagate = True
    # log.addHandler(main_file_handler)
    log.addHandler(debug_file_handler)
    log.addHandler(console_handler)

    return log


_logger = []


def initialize_logger(debug=False):
    if debug:
        log = create_stream_debug_logger()
        # log = logging.getLogger("Empty")
        # log.disabled = True
    else:
        log = create_logger()

    _logger.append(log)


def get_logger_instance():
    if len(_logger) < 1:
        initialize_logger()

    return _logger[0]
