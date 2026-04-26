from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from nsga2.core.individual import Individual
from nsga2.core.nondominated_sort import fast_non_dominated_sort
from nsga2.core.crowding_distance import crowding_distance_assignment
from nsga2.core.selection import tournament_selection
from nsga2.core.crossover import uniform_crossover
from nsga2.core.mutation import discrete_mutation


class NSGA2:
    """NSGA-II multi-objective optimiser for discrete variables.

    Args:
        problem: A problem object.
        pop_size: Population size N.
        n_var: Number of decision variables.
        n_obj: Number of objectives.
        num_choices: Array or int defining valid options per gene.
        seed: Random seed for reproducibility.
        prob_crossover: Crossover probability
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_var: int = 30,
        n_obj: int = 2,
        num_choices: Optional[np.ndarray] = None, 
        seed: Optional[int] = None,
        prob_crossover: float = 0.9,
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.n_var = n_var
        self.n_obj = n_obj
        
        if num_choices is None:
            raise ValueError("num_choices must be provided for discrete optimization")
        self.num_choices = np.array(num_choices)

        self.prob_crossover = prob_crossover
        self.prob_mutation = 1.0 / n_var

        self.rng = np.random.RandomState(seed)
        
        self.history: List[Dict] = []

    def _create_individual(self) -> Individual:
        """Create a random individual with integer decision variables."""
        vars_ = np.array([
            self.rng.randint(0, k) for k in self.num_choices
        ])
        return Individual(vars_, n_objectives=self.n_obj)

    def _init_population(self) -> List[Individual]:
        """Generate and evaluate the initial population."""
        population = [self._create_individual() for _ in range(self.pop_size)]
        for ind in population:
            ind.evaluate(self.problem)
        return population

    def _create_offspring(self, population: List[Individual]) -> List[Individual]:
        """Create N offspring via selection, crossover, and mutation."""
        offspring: List[Individual] = []
        while len(offspring) < self.pop_size:
            parent1 = tournament_selection(population, rng=self.rng)
            parent2 = tournament_selection(population, rng=self.rng)

            c1_vars, c2_vars = uniform_crossover(
                parent1.decision_vars, parent2.decision_vars,
                prob_crossover=self.prob_crossover,
                rng=self.rng,
            )
            
            discrete_mutation(
                c1_vars, self.prob_mutation, self.num_choices, rng=self.rng
            )
            discrete_mutation(
                c2_vars, self.prob_mutation, self.num_choices, rng=self.rng
            )

            c1 = Individual(c1_vars, n_objectives=self.n_obj)
            c2 = Individual(c2_vars, n_objectives=self.n_obj)
            c1.evaluate(self.problem)
            c2.evaluate(self.problem)

            offspring.append(c1)
            if len(offspring) < self.pop_size:
                offspring.append(c2)
        return offspring

    def run(
        self, 
        generations: int = 250, 
        verbose: bool = True, 
        callback: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute the NSGA-II algorithm.
        
        Args:
            generations: Maximum number of generations.
            verbose: Print progress every 50 generations.
            callback: Optional function called every generation with (gen, population).
        """
        if verbose:
            print(f"NSGA-II (Discrete) | pop_size={self.pop_size}, gens={generations}")

        population = self._init_population()

        for gen in range(generations):
            # 1. Create Offspring
            offspring = self._create_offspring(population)

            # 2. Combine Populations (Elitism)
            combined = population + offspring

            # 3. Fast Non-dominated Sort
            fronts = fast_non_dominated_sort(combined)

            # 4. Fill New Population
            new_population: List[Individual] = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                else:
                    crowding_distance_assignment(front)
                    front.sort(key=lambda ind: -ind.crowding_distance)
                    remaining = self.pop_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break

            # 5. Update Population
            population = new_population

            # History Tracking
            current_fronts = fast_non_dominated_sort(population)
            best_front = current_fronts[0]
            
            self.history.append({
                "generation": gen + 1,
                "n_fronts": len(current_fronts),
                "front_size": len(best_front),
                "objectives": np.array([ind.objectives for ind in best_front])
            })

            # Logging
            if verbose and (gen + 1) % 50 == 0:
                print(f"  Gen {gen + 1:>4d}/{generations} | Front 0 size: {len(best_front)}")

            if callback is not None:
                callback(gen, population)

        # Extract Final Results
        final_fronts = fast_non_dominated_sort(population)
        pareto_set = np.array([ind.decision_vars for ind in final_fronts[0]])
        pareto_front = np.array([ind.objectives for ind in final_fronts[0]])

        return pareto_set, pareto_front