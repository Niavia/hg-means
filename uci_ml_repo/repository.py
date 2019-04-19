import utils
import numpy as np


def dataset(dataset_name):
    datapoints_path = utils.data_path(f"{dataset_name}.txt")
    labels_path = utils.data_path(f"{dataset_name}.label")

    try:
        with open(datapoints_path, "r") as datapoints_fp:
            file_iterator = iter(datapoints_fp)
            _ = next(file_iterator)  # discard first line
            datapoints = [list(map(float, line.split()))
                          for line in file_iterator]
            datapoints = np.array(datapoints, dtype=np.float)
    except IOError:
        datapoints = None

    try:
        with open(labels_path, "r") as labels_fp:
            labels = [int(line.strip()) for line in labels_fp]
            labels = np.array(labels, dtype=np.int)
    except IOError:
        labels = None

    return datapoints, labels
