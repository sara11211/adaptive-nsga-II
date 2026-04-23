"""
NSGA-II: A Fast and Elitist Multiobjective Genetic Algorithm.

Implementation based on the paper by K. Deb et al. (2002):
"A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
IEEE Transactions on Evolutionary Computation, Vol. 6, No. 2, April 2002.
"""

from nsga2.core.nsga2 import NSGA2
from nsga2.core.individual import Individual

__version__ = "1.0.0"
__all__ = ["NSGA2", "Individual"]
