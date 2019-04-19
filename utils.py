import os
import re
import json
import logging


def data_path(*path_segments):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(
        os.path.join(root_dir, 'data', *path_segments))


def output_path(*path_segments):
    root_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(
        os.path.join(root_dir, 'output', *path_segments))


def set_logger(log_path):
    """Set the logger to log info in terminal and file 'log_path'.
    In general, it is useful to have a logger so that every output
    to the terminal is saved in a permanent file.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json_file(d, file_path):
    with open(file_path, "w") as f:
        json.dump(d, f, indent=4)


def load_dict_from_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def print_table(myDict, colList=None):
    """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
    If column names (colList) aren't specified, they will show in random order.
    Author: Thierry Husson - Use it as you want but don't blame me.
    """
    if not colList:
        colList = list(myDict[0].keys() if myDict else [])
    myList = [colList]  # 1st row = header
    for item in myDict:
        myList.append([str(item[col]) or '' for col in colList])
    colSize = [max(map(len, map(str, col))) for col in zip(*myList)]
    formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
    myList.insert(1, ['-' * i for i in colSize])  # Seperating line
    for item in myList:
        print(formatStr.format(*item))
