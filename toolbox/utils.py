import enum
import logging
import os
import sys
from pathlib import Path

LOG_DIR = Path("./LOG_DIR")
NEW_LOG_DIR = None


def get_next_version(root_dir):
    try:
        listdir_info = os.listdir(root_dir)
    except OSError:
        print(f"Missing logger folder: {root_dir}")
        return 0

    existing_versions = []
    for d in listdir_info:
        bn = os.path.basename(d)
        if os.path.isdir(os.path.join(root_dir, d)) and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            if not dir_ver.isnumeric():
                continue
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions) + 1


def get_module_logger(name):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        return logger
    # create console handler and set level to debug
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    # fileHandler = logging.FileHandler(get_log_path() / f"logs.log")
    # fileHandler.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)
    # fileHandler.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    # logger.addHandler(fileHandler)
    return logger


def get_log_path():
    # Feels a bit hacky, but works so let's go with that for now.
    global NEW_LOG_DIR
    if NEW_LOG_DIR is None:
        NEW_LOG_DIR = LOG_DIR / f"version_{get_next_version(LOG_DIR)}"
        NEW_LOG_DIR.mkdir(exist_ok=True, parents=True)
    return NEW_LOG_DIR


class StrEnum(str, enum.Enum):
    """
    Enum where members are also (and must be) strings
    This is the exact same implementation introduced in python 3.11
    """

    def __new__(cls, *values):
        "values must already be of type `str`"
        if len(values) > 3:
            raise TypeError("too many arguments for str(): %r" % (values,))
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                raise TypeError("%r is not a string" % (values[0],))
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                raise TypeError("encoding must be a string, not %r" % (values[1],))
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    def _generate_next_value_(name, start, count, last_values):
        """
        Return the lower-cased version of the member name.
        """
        return name.lower()
