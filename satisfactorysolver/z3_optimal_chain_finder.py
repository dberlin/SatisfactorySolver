import logging
import time
from typing import override

import rich.pretty
from z3 import z3, z3num

from satisfactorysolver.optimal_chain_finder import OptimalChainFinder

logger = logging.getLogger(__name__)


class Z3OptimalChainFinder(OptimalChainFinder[z3.ArithRef]):
    def add_optimization_constraints(self, outputs_to_maximize):
        exprs = [self.resources_scaled, -(sum(outputs_to_maximize) * 99999)]
        # exprs.append(self.items_used * 0.4)
        expr = sum(exprs)
        self.solver_model.minimize(expr)

    def add_constraint_to_model(self, constraint, name=""):
        self.solver_model.add(constraint)
        return constraint

    def get_num_scopes(self):
        return self.num_scopes

    def push(self):
        self.solver_model.push()
        self.num_scopes += 1

    def pop(self, num=1):
        while num > 0:
            self.solver_model.pop()
            self.num_scopes -= 1
            num -= 1

    @override
    def get_fraction_from_val(self, val):
        numeral = z3num.Numeral(val)
        fraction = numeral.approx(3).as_fraction()
        return fraction

    @override
    def solve(self):
        logger.debug(f"Model:\n{self.solver_model}")
        start_time = time.perf_counter()
        logger.debug(f"Starting finding function max, time is {start_time}")
        result = self.solver_model.check()
        end_time = time.perf_counter()
        logger.debug(f"Ending finding function max, time is {end_time}")
        logger.debug(f"Elapsed time is {end_time - start_time} seconds")
        # If it can't be satisfied at all, give up early
        if result != z3.sat:
            return None
        if logger.level <= logging.DEBUG:
            rich.pretty.pprint(self.solver_model.model())
            self.print_inputs_outputs()
        return None

    def __init__(self, recipe_data):
        super().__init__(recipe_data)
        self.solver_model = z3.Optimize()
        z3.set_param('parallel.enable', True)
        self.num_scopes = 0

    @override
    def create_real_var(self, name: str):
        new_var = z3.Real(name)
        self.add_constraint_to_model(new_var >= 0)
        return new_var
