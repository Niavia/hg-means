from hybrid_ga import HGMeans, ProblemDescription
from storage import HGMeansStore
import uci_ml_repo as repo
from tqdm import tqdm
import numpy as np
import logging
import utils
import os

SEED = 250
DESCRIPTION_FMT = """
Task name: {0}
    number of clusters: {1}
    population size parameter: {2}
    max number of generations: {3}
    terminates if no improvement after {4} generations.
"""


def save_checkpoint_fn_factory(instance_name, store, pbar):
    def save_checkpoint(hgmeans_obj):
        nonlocal instance_name, store, pbar
        state = hgmeans_obj.get_state()
        store.save_checkpoint(instance_name, state)
        pbar.update(1)
    return save_checkpoint


def run_experiment(task_name, dataset, labels, num_clusters,
                   population_param, terminating_param, store, seed=None):
    instance_name = f"{task_name}-cz-{num_clusters}-tp-{terminating_param}-pp-{population_param}"  # noqa

    logging.info(DESCRIPTION_FMT.format(task_name,
                                        num_clusters,
                                        population_param,
                                        terminating_param[1],
                                        terminating_param[0]).strip())

    problem_descriptor = ProblemDescription(dataset, labels,
                                            num_clusters,
                                            population_param,
                                            terminating_param)

    random_gen = np.random.RandomState(seed)
    state = store.load_checkpoint(instance_name)
    if state:
        hg = HGMeans.from_state(problem_descriptor, random_gen, state)
        current_step = state["current_step"]
    else:
        hg = HGMeans(problem_descriptor, random_gen)
        current_step = 0

    with tqdm(total=terminating_param[1],
              initial=current_step, unit="generation") as pbar:
        callback = save_checkpoint_fn_factory(instance_name, store, pbar)
        best_solution = hg.run(callback)
        if pbar.n != terminating_param[1]:
            pbar.set_description("Terminated")
        evaluation = best_solution.evaluate()
        store.save_evaluation(instance_name, evaluation)


if __name__ == "__main__":
    os.makedirs(utils.output_path(), exist_ok=True)
    utils.set_logger(utils.output_path("train.log"))

    store = HGMeansStore(utils.output_path("checkpoints"),
                         utils.output_path("evaluations"))

    terminating_param = (500, 5000)
    population_param = (10, 20)
    a1_cluster_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    a2_cluster_sizes = [2, 5, 10, 15, 20, 25, 30, 40, 50]
    tasks_dict = {
        "Bavaria 1": ("bavaria1", a1_cluster_sizes),
        "Fisher Iris": ("iris", a1_cluster_sizes),
        "Liver Disorders": ("liver", a2_cluster_sizes),
        "Congressional Voting": ("congress", a2_cluster_sizes)
    }

    for task_name, task in tasks_dict.items():
        dataset_name, cluster_sizes = task
        dataset, labels = repo.dataset(dataset_name)
        if dataset is None:
            continue
        for num_clusters in cluster_sizes:
            run_experiment(task_name, dataset, labels, num_clusters,
                           population_param, terminating_param, store, SEED)
