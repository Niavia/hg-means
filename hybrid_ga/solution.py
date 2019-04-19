import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2


class Solution(object):
    MUTATION_RATE = 0.2

    def __init__(self, problem_description, mutation_param,
                 membership=None, centroids=None):
        self.problem_description = problem_description
        self.mutation_param = mutation_param
        self.membership_chromosome = None
        self.coordinate_chromosome = None
        self._clusters_cardinality = None
        self._cost = None

        if membership is not None:
            self.__membership_to_coordinate(membership)
        elif centroids is not None:
            self.__coordinate_to_membership(centroids)

    def __membership_to_coordinate(self, membership):
        dataset = self.problem_description.dataset
        num_clusters = self.problem_description.num_clusters
        num_points, num_features = dataset.shape
        membership = np.array(membership)
        cluster_ids, membership_chromosome = np.unique(membership,
                                                       return_inverse=True)

        if len(membership) < num_points:
            raise ValueError("Membership chromosome is too short.")
        elif len(membership) > num_points:
            raise ValueError("Membership chromosome is too long.")
        if len(cluster_ids) < num_clusters:
            raise ValueError("Not enough clusters in membership chromosome.")
        elif len(cluster_ids) > num_clusters:
            raise ValueError("Too many clusters in membership chromosome.")

        coordinate_chromosome = np.zeros((num_clusters, num_features))

        for index, cid in enumerate(cluster_ids):
            rows = np.where(membership == cid)
            coordinate_chromosome[index] = dataset[rows].mean(axis=0)

        self.membership_chromosome = membership_chromosome
        self.coordinate_chromosome = coordinate_chromosome

    def __coordinate_to_membership(self, centroids):
        dataset = self.problem_description.dataset
        num_clusters = self.problem_description.num_clusters
        num_features = dataset.shape[1]
        coordinate_chromosome = np.array(centroids)
        num_clusters_in_cc, num_features_in_cc = coordinate_chromosome.shape

        if num_features_in_cc < num_features:
            raise ValueError("Not enough features in coordinate chromosome.")
        elif num_features_in_cc > num_features:
            raise ValueError("Too many features coordinate chromosome.")
        if num_clusters_in_cc < num_clusters:
            raise ValueError("Not enough clusters in coordinate chromosome.")
        elif num_clusters_in_cc > num_clusters:
            raise ValueError("Too many clusters in coordinate chromosome.")

        membership_chromosome = cdist(dataset,
                                      coordinate_chromosome,
                                      "euclidean"
                                      ).argmin(axis=1)

        self.membership_chromosome = membership_chromosome
        self.coordinate_chromosome = coordinate_chromosome

    def __compute_cost(self):
        dataset = self.problem_description.dataset
        membership = self.membership_chromosome
        centroids = self.coordinate_chromosome
        deviations = dataset - centroids[membership]
        return np.linalg.norm(deviations, axis=1).sum()

    def __compute_cluster_cardinality(self):
        _, counts = np.unique(self.membership_chromosome,
                              return_counts=True)
        return np.sort(counts).tolist()

    def mutate(self, random_state):
        self.mutation_param += random_state.uniform(-Solution.MUTATION_RATE,
                                                    Solution.MUTATION_RATE)
        self.mutation_param = max(0, min(1, self.mutation_param))
        alpha = self.mutation_param
        dataset = self.problem_description.dataset
        num_points = dataset.shape[0]
        num_clusters = self.problem_description.num_clusters
        index_of_centroid_to_delete = random_state.randint(0, num_clusters)
        temp_centroids = np.delete(self.coordinate_chromosome,
                                   index_of_centroid_to_delete, axis=0)

        dist_to_nearest_centroid = cdist(dataset,
                                         temp_centroids,
                                         "euclidean"
                                         ).min(axis=1)
        norm = dist_to_nearest_centroid.sum()
        selection_probabilities = (np.full((num_points,),
                                           (1 - alpha) / num_points) +
                                   (alpha * dist_to_nearest_centroid / norm))

        new_centroid_index = random_state.choice(num_points,
                                                 p=selection_probabilities)
        new_centroid = dataset[new_centroid_index]
        centroids = np.append(temp_centroids, [new_centroid], axis=0)
        self.__coordinate_to_membership(centroids)
        self.repair(random_state)

    def repair(self, random_state):
        num_required_clusters = self.problem_description.num_clusters
        dataset = self.problem_description.dataset
        num_points = dataset.shape[0]

        cluster_ids, clusters_cardinality = np.unique(
            self.membership_chromosome,
            return_counts=True)
        num_empty_clusters = num_required_clusters - len(cluster_ids)
        if num_empty_clusters == 0:
            return

        dist_to_nearest_centroid = cdist(dataset,
                                         self.coordinate_chromosome,
                                         "euclidean"
                                         ).min(axis=1)

        empty_cluster_id = cluster_ids.max() + 1
        membership = self.membership_chromosome
        clusters_cardinality = clusters_cardinality.tolist()
        while num_empty_clusters > 0:
            # select a datapoint el at random with preference given to points
            # far away from their centroids
            norm = dist_to_nearest_centroid.sum()
            selection_probabilities = dist_to_nearest_centroid / norm
            rand_el_index = random_state.choice(num_points,
                                                p=selection_probabilities)
            el_centroid_index = membership[rand_el_index]
            # if the selected element isn't the only one in its cluster,
            if clusters_cardinality[el_centroid_index] > 1:
                # move it to a new cluster
                clusters_cardinality.append(1)
                clusters_cardinality[el_centroid_index] -= 1
                membership[rand_el_index] = empty_cluster_id
                # prevent the element from being selected again
                dist_to_nearest_centroid[rand_el_index] = 0
                # update loop control and other variables
                num_empty_clusters -= 1
                empty_cluster_id += 1

        self.__membership_to_coordinate(membership)

    def improve_by_local_search(self):
        centroids, labels = kmeans2(
            data=self.problem_description.dataset,
            k=self.coordinate_chromosome,
            minit="matrix"
        )
        self.membership_chromosome = labels
        self.coordinate_chromosome = centroids

    def evaluate(self):
        return {
            "cost": self.cost
        }

    @property
    def cost(self):
        if self._cost is None:
            self._cost = self.__compute_cost()
        return self._cost

    @property
    def clusters_cardinality(self):
        if self._clusters_cardinality is None:
            self._clusters_cardinality = self.__compute_cluster_cardinality()
        return self._clusters_cardinality

    def get_state(self):
        return {
            "cost": self.cost,
            "mutation_param": self.mutation_param,
            "membership_chromosome": self.membership_chromosome.tolist()
        }

    @classmethod
    def from_state(cls, problem_description, state_dict):
        return cls(
            problem_description,
            mutation_param=state_dict["mutation_param"],
            membership=np.array(
                state_dict["membership_chromosome"], dtype=np.int)
        )
