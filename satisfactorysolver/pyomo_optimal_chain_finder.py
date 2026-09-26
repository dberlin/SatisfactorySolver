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
        # Lexicographic objectives, solved in order. Each optimum is fixed
        # before the next is solved: a weighted sum could trade them off.
        self.objective_stages = []
        if outputs_to_maximize:
            self.objective_stages.append((sum(outputs_to_maximize), pyo.maximize))
        self.objective_stages.append((self.resources_scaled, pyo.minimize))
        if self.tie_break is not None:
            self.objective_stages.append((self.tie_break, pyo.minimize))

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
        stages = self.objective_stages
        self.objective_stages = []
        for index, (expr, sense) in enumerate(stages):
            if index == 0:
                self.solver_model.objective = pyo.Objective(expr=expr, sense=sense)
            else:
                prev_expr = stages[index - 1][0]
                self.solver_model.add_component(
                    f"fixed_objective_{index - 1}",
                    pyo.Constraint(expr=prev_expr == pyo.value(prev_expr)),
                )
                self.solver_model.objective.set_value(expr)
                self.solver_model.objective.sense = sense
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
        self.objective_stages = []

    @override
    def get_model_result_by_var(self, var):
        if not self._has_solution:
            raise RuntimeError("No optimal solution is available")
        return var.value
