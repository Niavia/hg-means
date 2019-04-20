import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2, ClusterError
import utils


class Solution(object):
    MUTATION_RATE = 0.2

    def __init__(self, problem_description, random_gen,
                 mutation_param, membership=None, centroids=None):
        self.problem_description = problem_description
        self.mutation_param = mutation_param
        self.random_gen = random_gen
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
        # the returned inverses assigned to membership_chromosome are indices
        # into the cluster_ids. If the cluster_ids form an
        # index set with no gap, i.e, all elements in cluster_ids
        # are from 0 to len(cluster_ids) - 1, then the passed in membership
        # equals the membership_chromosome. But we can't rely on the user
        # passing in a gap-free index set.

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
        centroids = np.array(centroids)
        num_clusters_in_cc, num_features_in_cc = centroids.shape

        if num_features_in_cc < num_features:
            raise ValueError("Not enough features in coordinate chromosome.")
        elif num_features_in_cc > num_features:
            raise ValueError("Too many features coordinate chromosome.")
        if num_clusters_in_cc < num_clusters:
            raise ValueError("Not enough clusters in coordinate chromosome.")
        elif num_clusters_in_cc > num_clusters:
            raise ValueError("Too many clusters in coordinate chromosome.")

        membership = cdist(dataset,
                           centroids,
                           "euclidean"
                           ).argmin(axis=1)

        # If there is a tie in the euclidean distance calculated by cdist,
        # argmin always returns the index of the first centroid in the tie.
        # Therefore, we may have a case where a centroid isn't assigned any
        # data points because all data points are closer or equally close to
        # some preceding centroid. Consequently, there will be gaps
        # in membership.

        # detect and fix gap in membership
        cluster_ids, membership_chromosome = np.unique(membership,
                                                       return_inverse=True)
        has_gap = len(cluster_ids) != cluster_ids[-1] + 1
        if has_gap:
            used_centroids = [
                centroids[i]
                for i in cluster_ids
            ]
            self.membership_chromosome = membership_chromosome
            self.coordinate_chromosome = np.vstack(used_centroids)
            # a gap will lead to a decrease in number of centroids that
            # must be repaired
            self.repair()
        else:
            self.membership_chromosome = membership_chromosome
            self.coordinate_chromosome = centroids

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

    def sum_square_error(self):
        dataset = self.problem_description.dataset
        membership = self.membership_chromosome
        centroids = self.coordinate_chromosome
        deviations = dataset - centroids[membership]
        return (deviations * deviations).sum()

    def mutate(self):
        self.mutation_param += self.random_gen.uniform(
            -Solution.MUTATION_RATE, Solution.MUTATION_RATE)
        self.mutation_param = max(0, min(1, self.mutation_param))
        alpha = self.mutation_param
        dataset = self.problem_description.dataset
        num_points = dataset.shape[0]
        index_of_centroid_to_delete = self.random_gen.randint(
            0, self.coordinate_chromosome.shape[0])
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

        new_centroid_index = self.random_gen.choice(
            num_points,
            p=selection_probabilities)
        new_centroid = dataset[new_centroid_index]
        centroids = np.append(temp_centroids, [new_centroid], axis=0)
        self.__coordinate_to_membership(centroids)
        self.repair()

    def repair(self):
        """ It modifies the chromosomes to ensure the required number of
            clusters can be formed from them.
        """
        # precondition: membership_chromosome has no gap.
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
        membership = np.copy(self.membership_chromosome)
        clusters_cardinality = clusters_cardinality.tolist()
        while num_empty_clusters > 0:
            # select a datapoint el at random with preference given to points
            # far away from their centroids
            norm = dist_to_nearest_centroid.sum()
            selection_probabilities = dist_to_nearest_centroid / norm
            rand_el_index = self.random_gen.choice(num_points,
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
                # update the loop control and other variables
                num_empty_clusters -= 1
                empty_cluster_id += 1

        self.__membership_to_coordinate(membership)

    def improve_by_local_search(self):
        while True:
            try:
                centroids, labels = kmeans2(
                    data=self.problem_description.dataset,
                    k=self.coordinate_chromosome,
                    minit="matrix",
                    missing="raise",
                )
                self.membership_chromosome = labels
                self.coordinate_chromosome = centroids
                return
            except ClusterError:
                self.repair()

    def evaluate(self):
        return {
            "cost": self.cost,
            "sse": self.sum_square_error()
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
    def from_state(cls, problem_description, random_gen, state_dict):
        return cls(
            problem_description,
            random_gen,
            mutation_param=state_dict["mutation_param"],
            membership=np.array(
                state_dict["membership_chromosome"], dtype=np.int)
        )
