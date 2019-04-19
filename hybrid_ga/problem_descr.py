import numpy as np


class ProblemDescription(object):
    def __init__(self, dataset, labels, num_clusters,
                 population_param, terminating_param):
        self.dataset = np.array(dataset)
        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = None
        self.num_clusters = num_clusters

        min_population_size, max_population_size = population_param
        self.min_population_size = np.abs(min_population_size)
        self.max_population_size = np.abs(max_population_size)

        patience, max_num_iterations = terminating_param
        self.patience = np.abs(patience)
        self.max_num_iterations = np.abs(max_num_iterations)

        if self.labels is not None:
            if num_clusters != len(np.unique(self.labels)):
                self.do_eval = False
                self.labels = None
            elif len(self.labels) != len(self.dataset):
                raise ValueError(
                    "Given labels and dataset are of unequal lengths.")

        if len(self.dataset) < num_clusters:
            raise ValueError(
                "Number of clusters is more than available data points.")

        if self.min_population_size > self.max_population_size:
            self.max_population_size = 2*self.min_population_size
