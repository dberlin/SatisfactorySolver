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

from satisfactorysolver.solver_model import SolverModel


class Z3Model(SolverModel):
    def add_constraint_to_model(self, constraint, name=""):
        self.solver_model.add(constraint)
        return constraint

    solver_model: z3.Optimize | z3.Solver

    def __init__(self, model_data, condition):
        super().__init__(model_data)
        self.condition = condition
        if condition == 'balanced':
            self.solver_model = z3.Optimize()
        else:
            self.solver_model = z3.Solver()
        self.objective_var = self.create_real_var(name="Objective variable")
        self.penalty_var = self.create_real_var(name="Penalty variable")
        self.g = self.build_model()
        self.maximize_output()

    def create_real_var(self, name):
        new_var = z3.Real(name)
        self.add_constraint_to_model(new_var >= 0)
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

    def maximize_output(self):
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

        # Don't let objective fall to zero or else the other constraints are trivially satisfiable
        self.add_constraint_to_model(self.objective_var > 0)
        if self.condition == 'balanced':
            self.solver_model.maximize(self.objective_var)
            self.solver_model.minimize(self.penalty_var)

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
