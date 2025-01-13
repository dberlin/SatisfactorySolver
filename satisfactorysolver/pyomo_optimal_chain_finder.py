import fractions
import logging
import time
from typing import override

from pyomo import environ as pyo
from pyomo.contrib import appsi

from satisfactorysolver.optimal_chain_finder import OptimalChainFinder
from satisfactorysolver.pyomo_model import defractionize

logger = logging.getLogger(__name__)


class PyomoOptimalChainFinder(OptimalChainFinder[pyo.Var]):
    def add_optimization_constraints(self, outputs_to_maximize):
        exprs = [self.resources_scaled, -(sum(outputs_to_maximize) * 99999)]
        # exprs.append(self.items_used * 0.4)
        expr = sum(exprs)
        objective = pyo.Objective(expr=expr, sense=pyo.minimize)

        self.solver_model.add_component(str(next(self.count)), objective)

    def add_constraint_to_model(self, constraint, name=""):
        constraint = pyo.Constraint(expr=defractionize(constraint))
        name = str(next(self.count))
        self.solver_model.add_component(name, constraint)
        return constraint

    def create_real_var(self, name: str):
        var = pyo.Var(bounds=(0, None), name=name, domain=pyo.NonNegativeReals)
        self.solver_model.add_component(name, var)
        return var

    def get_fraction_from_val(self, val):
        return fractions.Fraction(val)

    def solve(self):
        logger.debug(f"Model:\n")
        if logger.level <= logging.DEBUG:
            self.solver_model.pprint()
        start_time = time.perf_counter()
        logger.debug(f"Starting finding function max, time is {start_time}")
        result = self.opt.solve(self.solver_model)
        end_time = time.perf_counter()
        logger.debug(f"Ending finding function max, time is {end_time}")
        logger.debug(f"Elapsed time is {end_time - start_time} seconds")
        # If it can't be satisfied at all, give up early
        if result.termination_condition != appsi.base.TerminationCondition.optimal:
            return None
        if logger.level <= logging.DEBUG:
            logger.debug(f"Model after finding function max:\n")
            self.solver_model.pprint()
            self.print_inputs_outputs()
        return None

    def __init__(self, recipe_data):
        super().__init__(recipe_data)
        self.solver_model = pyo.ConcreteModel()
        self.opt = appsi.solvers.Highs()
        self.opt.config.stream_solver = True

    @override
    def get_model_result_by_var(self, var):
        return var.value
