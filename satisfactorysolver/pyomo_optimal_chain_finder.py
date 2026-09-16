import fractions
import logging
from typing import override

from pyomo import environ as pyo
from pyomo.common.numeric_types import RegisterNumericType
from pyomo.contrib import appsi

from satisfactorysolver.optimal_chain_finder import OptimalChainFinder
from satisfactorysolver.pyomo_model import defractionize

# Model construction uses exact rates before defractionize lowers them for HiGHS.
RegisterNumericType(fractions.Fraction)

logger = logging.getLogger(__name__)


class PyomoOptimalChainFinder(OptimalChainFinder[pyo.Var]):
    def add_optimization_constraints(self, outputs_to_maximize):
        self.output_total = sum(outputs_to_maximize) if outputs_to_maximize else None
        if self.output_total is not None:
            self.solver_model.objective = pyo.Objective(
                expr=self.output_total, sense=pyo.maximize
            )
        else:
            self.solver_model.objective = pyo.Objective(
                expr=self.resources_scaled, sense=pyo.minimize
            )

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

    def solve(self) -> bool:
        self._has_solution = False
        self.opt.config.stream_solver = logger.isEnabledFor(logging.DEBUG)
        result = self.opt.solve(self.solver_model)
        if result.termination_condition != appsi.base.TerminationCondition.optimal:
            return False
        result.solution_loader.load_vars()
        if self.output_total is not None:
            # Fix the first optimum before minimizing resource use: a weighted
            # sum can sacrifice output, and is not a lexicographic objective.
            self.solver_model.maximum_output = pyo.Constraint(
                expr=self.output_total == pyo.value(self.output_total)
            )
            self.solver_model.objective.set_value(self.resources_scaled)
            self.solver_model.objective.sense = pyo.minimize
            self.output_total = None
            result = self.opt.solve(self.solver_model)
            if result.termination_condition != appsi.base.TerminationCondition.optimal:
                return False
            result.solution_loader.load_vars()
        self._has_solution = True
        return True

    def __init__(self, recipe_data):
        super().__init__(recipe_data)
        self.reset_solver_model()

    @override
    def reset_solver_model(self):
        self.solver_model = pyo.ConcreteModel()
        self.opt = appsi.solvers.Highs()
        self.opt.config.load_solution = False
        self.output_total = None

    @override
    def get_model_result_by_var(self, var):
        if not self._has_solution:
            raise RuntimeError("No optimal solution is available")
        return var.value
