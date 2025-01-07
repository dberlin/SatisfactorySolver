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
import itertools
from collections import defaultdict

import z3

from satisfactorysolver.solver_helpers import collect_vars
from satisfactorysolver.smt_model import SolverModel


class Z3Model(SolverModel):
    solver_model: z3.Optimize | z3.Solver

    def __init__(self, model_data, condition):
        super().__init__(model_data)
        self.count = itertools.count()
        self.condition = condition
        if condition == 'balanced':
            self.solver_model = z3.Optimize()
        else:
            self.solver_model = z3.Solver()
        self.objective_var = self.create_real_var(name="Objective variable")
        self.g = self.build_model()
        self.maximize_output_soft()

    def create_real_var(self, name):
        new_var = z3.Real(name)
        self.solver_model.add(new_var >= 0)
        return new_var

    @staticmethod
    def z3_equals(xs):
        if len(xs) == 1:
            return None
        constr = xs[0] == xs[0]
        first = xs[0]
        for x in xs[1:]:
            constr = z3.And(constr, first == x)
        return constr

    def maximize_output_soft(self):
        _, _, producer_output_vars = collect_vars(self.node_inputs, self.node_outputs, self.model_data.Nodes)
        prod_exprs = []
        edge_by_node_and_part = {graph_node[0]: defaultdict(list) for graph_node in self.g.nodes(data=True)}
        for edge in self.g.out_edges(data=True):
            part_name = edge[2]["part_name"]
            edge_var = edge[2]["edge_var"]
            edge_by_node_and_part[edge[0]][part_name].append(edge_var)
        for part_list in edge_by_node_and_part.values():
            for var_list in part_list.values():
                if self.condition == 'balanced':
                    all_equal = self.z3_equals(var_list)
                    if all_equal is not None:
                        self.solver_model.add_soft(all_equal)
                prod_exprs.extend(var_list)
        self.solver_model.add(self.objective_var == sum(prod_exprs))
        # Don't let objective fall to zero or else the soft constraint is trivially satisfiable
        self.solver_model.add(self.objective_var > 0)
        if self.condition == 'balanced':
            self.solver_model.maximize(self.objective_var)

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
            if self.condition == "practical":
                self.constraint_input_to_practical_division(node, node.Recipe)
        return g

    def absolute_diff(self, a, b):
        # This is faster than using if clauses
        tempvar = self.create_real_var(name=f"tempvar {next(self.count)} for absolute difference")
        self.solver_model.add((a - b) <= tempvar)
        self.solver_model.add(-(a - b) <= tempvar)
        return tempvar

    def ToInt(self, expr):
        return z3.ToInt(expr)
