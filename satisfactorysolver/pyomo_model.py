# Copyright (c) 2024 Daniel Berlin and others
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import fractions
import itertools
from collections import defaultdict

from pyomo import environ as pyo
from pyomo.common.numeric_types import nonpyomo_leaf_types
from pyomo.core.expr import ExpressionReplacementVisitor

from satisfactorysolver.solver_model import SolverModel


class PyomoModel(SolverModel):
    def ToInt(self, expr):
        pass

    def __init__(self, model_data, condition):
        super().__init__(model_data)
        self.model = pyo.ConcreteModel()
        self.node_inputs = {}
        self.node_outputs = {}
        self.model_data = model_data
        self.condition = condition
        self.g = None
        self.all_vars = []
        self.count = itertools.count()
        self.g = self.build_model()
        self.try_maximize_output()

    def add_constraint_to_model(self, constraint, name=""):
        constraint = pyo.Constraint(expr=defractionize(constraint))
        name = str(next(self.count))
        self.model.add_component(name, constraint)
        return constraint

    def create_real_var(self, name: str):
        var = pyo.Var(bounds=(0, None), name=name, domain=pyo.NonNegativeReals)
        self.model.add_component(name, var)
        self.all_vars.append(var)
        return var

    def build_model(self):
        self.create_modeler_node_outputs()
        self.create_modeler_node_inputs()
        g = self.create_modeler_node_graph()
        self.create_edge_constraints(g)
        self.create_produced_resource_constraints()
        for node in self.model_data.Nodes:
            if node.Recipe is None:
                continue
            self.constrain_output_amount_to_input_amount(node, node.Recipe)
            self.constrain_io_to_same_batch_count(node, node.Recipe)
            self.constrain_output_to_node_max(node, node.Recipe)
        return g

    def try_maximize_output(self):
        USE_LINEAR_FORMULATION = False

        @self.model.Objective(sense=pyo.maximize)
        def maximize_output(m):
            if USE_LINEAR_FORMULATION:
                self.objective_var = self.create_real_var(name="Objective variable")
                self.penalty_var = self.create_real_var(name="Penalty variable")
                prod_exprs = []
                penalty_exprs = []
                edge_by_node_and_part = {graph_node[0]: defaultdict(list) for graph_node in self.g.nodes(data=True)}
                for edge in self.g.out_edges(data=True):
                    part_name = edge[2]["part_name"]
                    edge_var = edge[2]["edge_var"]
                    edge_by_node_and_part[edge[0]][part_name].append(edge_var)
                for part_list in edge_by_node_and_part.values():
                    # Penalize non-equal outputs
                    for var_list in part_list.values():
                        for pair in itertools.combinations(var_list, 2):
                            penalty_exprs.append(self.absolute_diff(pair[0], pair[1]))
                        prod_exprs.extend(var_list)
                self.add_constraint_to_model(self.objective_var == sum(prod_exprs))
                self.add_constraint_to_model(self.penalty_var == sum(penalty_exprs))

                # Don't let objective fall to zero or else the  constraints are trivially satisfiable
                self.add_constraint_to_model(self.objective_var >= 0.01)
                if self.condition == 'balanced':
                    self.solver_model.maximize(self.objective_var)
                    self.solver_model.minimize(self.penalty_var)
            else:
                # Sort edges by node and part
                prod_exprs = []
                edge_by_node_and_part = {}
                for graph_node in self.g.nodes(data="node_model"):
                    edge_by_node_and_part[graph_node[0]] = defaultdict(list)
                for edge in self.g.out_edges(data=True):
                    part_name = edge[2]["part_name"]
                    edge_var = edge[2]["edge_var"]
                    edge_by_node_and_part[edge[0]][part_name].append(edge_var)
                # Multiply edges to generate optimality with balance (non-linear)
                for part_list in edge_by_node_and_part.values():
                    for var_list in part_list.values():
                        prod_exprs.append(pyo.prod(var_list))

                return sum(prod_exprs)

    def pprint(self):
        self.model.pprint()

    def print_inputs_outputs(self):
        # Print the resulting values as two tables
        from rich import table
        from rich import console
        input_table = table.Table(title="Node inputs")
        input_table.add_column("Node name", justify="left", no_wrap=True)
        input_table.add_column("Value", justify="left", no_wrap=True)
        for node_input in self.node_inputs:
            for input_var in self.node_inputs[node_input].values():
                input_table.add_row(input_var.name, str(input_var.value))
        console = console.Console()
        console.print(input_table)
        output_table = table.Table(title="Node outputs")
        output_table.add_column("Node name", justify="left", no_wrap=True)
        output_table.add_column("Value", justify="left", no_wrap=True)
        for node_output in self.node_outputs:
            for output_var in self.node_outputs[node_output].values():
                output_table.add_row(output_var.name, str(output_var.value))
        console.print(output_table)


class ReplaceFractions(ExpressionReplacementVisitor):
    """Replace fractions that occur in expressions with their float value"""

    def __init__(self):
        super().__init__()

    def beforeChild(self, node, child, child_idx):
        if isinstance(child, fractions.Fraction):
            return False, float(child)
        if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
            return False, child
        return True, None


class FindFractions(ExpressionReplacementVisitor):
    def __init__(self):
        super().__init__()

    def beforeChild(self, node, child, child_idx):
        if isinstance(child, fractions.Fraction):
            assert "Should not have found a fraction"
        if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
            return False, child
        return True, None


def defractionize(expr: pyo.Expression) -> pyo.Expression:
    result = ReplaceFractions().walk_expression(expr)
    FindFractions().walk_expression(result)
    return result
