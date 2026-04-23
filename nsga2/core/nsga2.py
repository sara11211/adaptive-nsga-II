"""
Main NSGA-II algorithm loop.

Implements the complete evolutionary cycle from Section III-C of the paper:

  1. Generate initial population P_0 of size N, evaluate.
  2. For each generation t:
     a. Create offspring Q_t of size N via selection, crossover, mutation.
     b. Form combined population R_t = P_t ∪ Q_t (size 2N).
     c. Fast nondominated sort R_t.
     d. Fill new population P_{t+1} front by front.
     e. If last front exceeds remaining slots, use crowding distance to prune.
     f. Compute crowding distance for every front in P_{t+1}.
"""

from __future__ import annotations
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from nsga2.core.individual import Individual
from nsga2.core.nondominated_sort import fast_non_dominated_sort
from nsga2.core.crowding_distance import crowding_distance_assignment
from nsga2.core.selection import tournament_selection
from nsga2.core.crossover import sbx_crossover
from nsga2.core.mutation import polynomial_mutation


class NSGA2:
    """NSGA-II multi-objective optimiser.

    Args:
        problem: A problem object implementing ``evaluate(x) -> (objectives, constraints)``.
        pop_size: Population size N (default 100).
        n_var: Number of decision variables.
        n_obj: Number of objectives.
        n_constr: Number of constraints (default 0).
        bounds: (n_var, 2) array of [lower, upper] bounds.
        seed: Random seed for reproducibility (default None).
        prob_crossover: SBX crossover probability (default 0.9).
        eta_c: SBX distribution index (default 20.0).
        eta_m: Polynomial mutation distribution index (default 20.0).

    Example::

        problem = ZDT1()
        optimizer = NSGA2(problem, pop_size=100, n_var=30, n_obj=2,
                          bounds=problem.bounds)
        pareto_set, pareto_front = optimizer.run(generations=250)
    """

    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_var: int = 30,
        n_obj: int = 2,
        n_constr: int = 0,
        bounds: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        prob_crossover: float = 0.9,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.bounds = np.array(bounds, dtype=np.float64) if bounds is not None else None
        self.prob_crossover = prob_crossover
        self.eta_c = eta_c
        self.eta_m = eta_m

        # Per-variable mutation probability: 1/n_var (recommended in the paper)
        self.prob_mutation = 1.0 / n_var

        self.np_rng = np.random.RandomState(seed)
        self.rng = random.Random(seed)
        # Also seed Python's global random module (used by tournament_selection)
        if seed is not None:
            random.seed(seed)

        # History tracking
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _create_individual(self) -> Individual:
        """Create a random individual within the variable bounds."""
        if self.bounds is not None:
            vars_ = np.array([
                self.rng.uniform(lo, hi) for lo, hi in self.bounds
            ])
        else:
            vars_ = self.np_rng.rand(self.n_var)
        return Individual(vars_, n_objectives=self.n_obj, n_constraints=self.n_constr)

    def _init_population(self) -> List[Individual]:
        """Generate and evaluate the initial population."""
        population = [self._create_individual() for _ in range(self.pop_size)]
        for ind in population:
            ind.evaluate(self.problem)
        return population

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def _create_offspring(self, population: List[Individual]) -> List[Individual]:
        """Create N offspring via tournament selection, SBX crossover,
        and polynomial mutation."""
        offspring: List[Individual] = []
        while len(offspring) < self.pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            c1_vars, c2_vars = sbx_crossover(
                parent1.decision_vars, parent2.decision_vars,
                prob_crossover=self.prob_crossover,
                eta_c=self.eta_c,
                bounds=self.bounds,
                rng=self.np_rng,
            )
            # Mutate
            polynomial_mutation(c1_vars, self.prob_mutation, self.eta_m, self.bounds,
                                rng=self.np_rng)
            polynomial_mutation(c2_vars, self.prob_mutation, self.eta_m, self.bounds,
                                rng=self.np_rng)

            c1 = Individual(c1_vars, n_objectives=self.n_obj, n_constraints=self.n_constr)
            c2 = Individual(c2_vars, n_objectives=self.n_obj, n_constraints=self.n_constr)
            c1.evaluate(self.problem)
            c2.evaluate(self.problem)

            offspring.append(c1)
            if len(offspring) < self.pop_size:
                offspring.append(c2)
        return offspring

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, generations: int = 250, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Execute the NSGA-II algorithm.

        Args:
            generations: Maximum number of generations (default 250).
            verbose: Print progress every 50 generations (default True).

        Returns:
            (pareto_set, pareto_front) where:
            - pareto_set is an (N, n_var) array of decision variables for
              the final nondominated front.
            - pareto_front is an (N, n_obj) array of objective values.
        """
        if verbose:
            print(f"NSGA-II | pop_size={self.pop_size}, gens={generations}")

        population = self._init_population()

        for gen in range(generations):
            # Create offspring
            offspring = self._create_offspring(population)

            # Combine parent + offspring
            combined = population + offspring

            # Nondominated sorting
            fronts = fast_non_dominated_sort(combined)

            # Fill new population front by front
            new_population: List[Individual] = []
            for front in fronts:
                # If the whole front fits, add it
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                else:
                    # Sort front by crowding distance and take the best
                    crowding_distance_assignment(front)
                    front.sort(key=lambda ind: -ind.crowding_distance)
                    remaining = self.pop_size - len(new_population)
                    new_population.extend(front[:remaining])
                    break

            # Assign crowding distance to every front in the new population
            final_fronts = fast_non_dominated_sort(new_population)
            for front in final_fronts:
                crowding_distance_assignment(front)

            population = new_population

            # Record history
            best_front = final_fronts[0]
            objs = np.array([ind.objectives for ind in best_front])
            self.history.append({
                "generation": gen + 1,
                "n_fronts": len(final_fronts),
                "front_size": len(best_front),
                "objectives": objs.copy(),
            })

            if verbose and (gen + 1) % 50 == 0:
                print(f"  Gen {gen + 1:>4d}/{generations} | "
                      f"fronts={len(final_fronts)}, "
                      f"front0_size={len(best_front)}")

        # Extract the final Pareto front
        final_fronts = fast_non_dominated_sort(population)
        pareto_set = np.array([ind.decision_vars for ind in final_fronts[0]])
        pareto_front = np.array([ind.objectives for ind in final_fronts[0]])

        if verbose:
            print(f"Done. Final Pareto front size: {len(pareto_front)}")

        return pareto_set, pareto_front
