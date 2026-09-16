import logging
from fractions import Fraction
from typing import override

from z3 import z3

from satisfactorysolver.optimal_chain_finder import OptimalChainFinder

logger = logging.getLogger(__name__)


class Z3OptimalChainFinder(OptimalChainFinder[z3.ArithRef]):
    def add_optimization_constraints(self, outputs_to_maximize):
        if outputs_to_maximize:
            self.objectives.append(self.solver_model.maximize(sum(outputs_to_maximize)))
        self.objectives.append(self.solver_model.minimize(self.resources_scaled))

    def add_constraint_to_model(self, constraint, name=""):
        self.solver_model.add(constraint)
        return constraint

    def get_num_scopes(self):
        return self.num_scopes

    def push(self):
        self.solver_model.push()
        self._has_solution = False
        self.num_scopes += 1

    def pop(self, num=1):
        while num > 0:
            self.solver_model.pop()
            self.num_scopes -= 1
            num -= 1
        self._has_solution = False

    @override
    def get_fraction_from_val(self, val) -> Fraction:
        return val.as_fraction()

    @override
    def solve(self) -> bool:
        self._has_solution = False
        logger.debug("Model:\n%s", self.solver_model)
        if self.solver_model.check() != z3.sat:
            return False
        # Optimize reports sat for unbounded objectives too. Only finite optima
        # can be displayed as an optimal production chain.
        if not all(
            z3.is_rational_value(bound) or z3.is_int_value(bound)
            for objective in self.objectives
            for bound in (objective.lower(), objective.upper())
        ):
            return False
        self._has_solution = True
        return True

    def __init__(self, recipe_data):
        super().__init__(recipe_data)
        self.reset_solver_model()

    @override
    def reset_solver_model(self):
        self.solver_model = z3.Optimize()
        self.num_scopes = 0
        self.objectives: list[z3.OptimizeObjective] = []

    @override
    def create_real_var(self, name: str):
        new_var = z3.Real(name)
        self.add_constraint_to_model(new_var >= 0)
        return new_var
