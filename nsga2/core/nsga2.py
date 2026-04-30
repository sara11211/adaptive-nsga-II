from __future__ import annotations
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from nsga2.core.individual import Individual
from nsga2.core.nondominated_sort import fast_non_dominated_sort
from nsga2.core.crowding_distance import crowding_distance_assignment
from nsga2.core.selection import tournament_selection
from nsga2.core.crossover import uniform_crossover
from nsga2.core.mutation import discrete_mutation
from nsga2.core.adaptive_mutation import AdaptiveMutationPool


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
        adaptive_mutation: Optional[dict] = None,
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

        # ── Adaptive mutation pool (SaMuNet) ──────────────────────────────
        self._adaptive_mutation_config = adaptive_mutation
        self._adaptive_pool: Optional[AdaptiveMutationPool] = None
        self._feedback_queue: List[dict] = []
        if adaptive_mutation is not None:
            self._adaptive_pool = AdaptiveMutationPool(
                num_choices=self.num_choices,
                initial_probs=adaptive_mutation.get("initial_probs"),
                update_period=adaptive_mutation.get("update_period", 5),
            )
            print(f"[NSGA-II] Adaptive mutation ENABLED  "
                  f"(period={self._adaptive_pool.update_period}, "
                  f"init_probs={self._adaptive_pool.probs.tolist()})")

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
        offspring = []
        use_adaptive = self._adaptive_pool is not None

        while len(offspring) < self.pop_size:
            p1 = tournament_selection(population, self.rng)
            p2 = tournament_selection(population, self.rng)
            c1_vars, c2_vars = uniform_crossover(
                p1.decision_vars, p2.decision_vars,
                self.prob_crossover, self.rng,
            )

            if use_adaptive:
                # ── Adaptive mutation: child 1 ────────────────────
                strat_idx = self._adaptive_pool.select_strategy(self.rng)

                c1 = Individual(c1_vars, n_objectives=self.n_obj)
                c1.mutation_type = strat_idx
                c1.evaluate(self.problem)
                self._adaptive_pool.apply_strategy(c1.decision_vars, strat_idx, self.rng)
                c1.evaluate(self.problem)

                # DEFERRED feedback — save parent rank for later comparison
                self._feedback_queue.append({
                    "child": c1,
                    "strategy_idx": strat_idx,
                    "parent_rank": p1.rank,
                })
            else:
                discrete_mutation(c1_vars, self.prob_mutation, self.num_choices, self.rng)
                c1 = Individual(c1_vars, n_objectives=self.n_obj)
                c1.evaluate(self.problem)

            offspring.append(c1)

            if len(offspring) < self.pop_size:
                if use_adaptive:
                    # ── Adaptive mutation: child 2 ────────────────
                    strat_idx = self._adaptive_pool.select_strategy(self.rng)

                    c2 = Individual(c2_vars, n_objectives=self.n_obj)
                    c2.mutation_type = strat_idx
                    c2.evaluate(self.problem)
                    self._adaptive_pool.apply_strategy(c2.decision_vars, strat_idx, self.rng)
                    c2.evaluate(self.problem)

                    self._feedback_queue.append({
                        "child": c2,
                        "strategy_idx": strat_idx,
                        "parent_rank": p2.rank,
                    })
                else:
                    discrete_mutation(c2_vars, self.prob_mutation, self.num_choices, self.rng)
                    c2 = Individual(c2_vars, n_objectives=self.n_obj)
                    c2.evaluate(self.problem)

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

        # ── Deferred adaptive mutation feedback ─────────────────────
            if self._adaptive_pool is not None and self._feedback_queue:
                # Build rank lookup from the combined sort
                rank_lookup = {}
                for rank, front in enumerate(fronts):
                    for ind in front:
                        rank_lookup[id(ind)] = rank

                for entry in self._feedback_queue:
                    child_rank = rank_lookup.get(id(entry["child"]), 999)
                    parent_rank = entry["parent_rank"]
                    strat = entry["strategy_idx"]

                    if child_rank < parent_rank:
                        # Better front → SUCCESS
                        self._adaptive_pool.record_outcome(strat, True)
                    elif child_rank > parent_rank:
                        # Worse front → FAILURE
                        self._adaptive_pool.record_outcome(strat, False)
                    else:
                        # Same rank → NEUTRAL (could add CD tie-breaker here)
                        pass

                self._feedback_queue.clear()

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

            # ── Adaptive mutation: update probabilities ────────────────
            if self._adaptive_pool is not None:
                updated = self._adaptive_pool.step()
                if updated and verbose:
                    print(f"  Gen {gen+1} | Adaptive mutation probs: "
                          f"{self._adaptive_pool.summary}")

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

        if verbose:
            print(f"Done. Final Pareto front size: {len(pareto_front)}")
            if self._adaptive_pool is not None:
                print(f"  Final adaptive mutation probs: {self._adaptive_pool.summary}")
        
        return pareto_set, pareto_front