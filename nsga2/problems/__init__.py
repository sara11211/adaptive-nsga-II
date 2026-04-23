"""Benchmark multi-objective problems for testing NSGA-II."""

from nsga2.problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from nsga2.problems.constrained import CONSTR, SRN, TNK

__all__ = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6", "CONSTR", "SRN", "TNK"]
