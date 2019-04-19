import numpy as np
import heapq
from collections import defaultdict
from operator import itemgetter
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment as hungarian
from .solution import Solution

_LOG_BASE10_OF_2 = 3.010299956639811952137388947244930267681898814621085413104274611e-1  # noqa


def create_population(problem_description, random_state):
    dataset = problem_description.dataset
    num_points = len(dataset)
    num_clusters = problem_description.num_clusters
    min_population_size = problem_description.min_population_size
    population = []
    for _ in range(min_population_size):
        centroids_indices = random_state.choice(
            num_points,
            num_clusters,
            replace=False)
        centroids = dataset[centroids_indices]
        mutation_param = random_state.uniform()
        solution = Solution(problem_description,
                            mutation_param, centroids=centroids)
        solution.repair(random_state)
        solution.improve_by_local_search()
        population.append(solution)
    return population


def tournament_selection(population, k, random_state):
    best = None
    for _ in range(k):
        ind = random_state.randint(len(population))
        if best is None or population[ind].cost < best.cost:
            best = population[ind]
    return best


def cross_over(parent1, parent2, random_state):
    distance_matrix = cdist(parent1.coordinate_chromosome,
                            parent2.coordinate_chromosome,
                            "euclidean")
    matching = hungarian(distance_matrix)
    child_centroids = [
        _choose(a, b, random_state)
        for a, b in zip(parent1.coordinate_chromosome[matching[0]],
                        parent2.coordinate_chromosome[matching[1]])
    ]
    child_centroids = np.vstack(child_centroids)
    child_mutation_param = (parent1.mutation_param +
                            parent2.mutation_param) / 2
    child = Solution(parent1.problem_description,
                     child_mutation_param,
                     centroids=child_centroids)
    child.repair(random_state)
    return child


def select_survivors(population, size, random_state):
    population_as_set = set(population)
    groups = defaultdict(list)
    for solution in population:
        key_builder = [_frac_part_to_sigfig(solution.cost, 6)]
        key_builder.extend(solution.clusters_cardinality)
        key = tuple(key_builder)
        groups[key].append(solution)

    clone_keys = [k for k, v in groups.items() if len(v) > 1]
    for key in sorted(clone_keys, key=itemgetter(0), reverse=True):
        clones = groups[key]
        to_save = random_state.randint(0, len(clones))
        clones[0], clones[to_save] = clones[to_save], clones[0]
        to_be_removed = clones[1:len(population_as_set) - size + 1]
        population_as_set.difference_update(to_be_removed)
        if len(population_as_set) <= size:
            break

    if len(population_as_set) > size:
        return heapq.nsmallest(size, population_as_set, lambda s: s.cost)
    return list(population_as_set)


def _choose(a, b, random_state):
    if random_state.uniform() > 0.5:
        return a
    return b


def _frac_part_to_sigfig(x, sigfig):
    integ = np.trunc(x)
    frac = x - integ
    return integ + _round_to_sigfig(frac, sigfig)


def _round_to_sigfig(x, sig_figures):
    x_sgn = np.sign(x)
    abs_x = x_sgn * x
    binary_mantissa, binary_exponent = np.frexp(abs_x)

    decimal_magnitude = _LOG_BASE10_OF_2 * binary_exponent
    decimal_exponent = np.floor(decimal_magnitude)
    decimal_mantissa = (binary_mantissa *
                        10.0 ** (decimal_magnitude - decimal_exponent))

    if decimal_mantissa < 1.0:
        decimal_mantissa *= 10.0
        decimal_exponent -= 1.0

    return (x_sgn * 10.0 ** decimal_exponent *
            np.around(decimal_mantissa, decimals=sig_figures - 1))
