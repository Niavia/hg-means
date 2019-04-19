from . import genetic_operations as go
from .solution import Solution
import numpy as np


class HGMeans(object):
    def __init__(self, problem_description,
                 population=None, best_solution=None,
                 current_step=0, last_improvement_step=0,
                 random_state=None):
        self.population = population
        self.current_step = current_step
        self.best_solution = best_solution
        self.problem_description = problem_description
        self.random_state = random_state or np.random.RandomState()
        self.last_improvement_step = last_improvement_step

    def run(self, iteration_done_fn=None):
        patience = self.problem_description.patience
        max_steps = self.problem_description.max_num_iterations
        min_population_size = self.problem_description.min_population_size
        max_population_size = self.problem_description.max_population_size

        if self.population is None:
            self.population = go.create_population(self.problem_description,
                                                   self.random_state)
            self.best_solution = min(self.population, key=lambda s: s.cost)

        while (self.current_step < max_steps and
               (self.current_step - self.last_improvement_step) < patience):
            parent1 = go.tournament_selection(
                self.population, 2, self.random_state)
            parent2 = go.tournament_selection(
                self.population, 2, self.random_state)
            child = go.cross_over(parent1, parent2, self.random_state)
            child.mutate(self.random_state)
            child.improve_by_local_search()
            self.population.append(child)
            if child.cost < self.best_solution.cost:
                self.best_solution = child
                self.last_improvement_step = self.current_step
            if len(self.population) > max_population_size:
                self.population = go.select_survivors(self.population,
                                                      min_population_size,
                                                      self.random_state)
            self.current_step += 1
            if iteration_done_fn is not None:
                iteration_done_fn(self)

        return self.best_solution

    def get_state(self):
        return {
            "population": [x.get_state() for x in self.population],
            "best": self.best_solution.get_state(),
            "current_step": self.current_step,
            "last_improvement_step": self.last_improvement_step,
        }

    @classmethod
    def from_state(cls, problem_description, state_dict):
        population = [Solution.from_state(problem_description, s)
                      for s in state_dict["population"]
                      ]
        best_solution = Solution.from_state(problem_description,
                                            state_dict["best"])
        current_step = state_dict["current_step"]
        last_improvement_step = state_dict["last_improvement_step"]
        return cls(problem_description,
                   population=population,
                   best_solution=best_solution,
                   current_step=current_step,
                   last_improvement_step=last_improvement_step
                   )
