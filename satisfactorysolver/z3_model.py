import itertools
from collections import defaultdict

import z3

from satisfactorysolver.solver_helpers import collect_vars
from satisfactorysolver.solver_model import SolverModel


class Z3Model(SolverModel):
    solver_model: z3.Optimize | z3.Solver

    def __init__(self, model_data, use_optimizer=False):
        super().__init__(model_data)
        self.count = itertools.count()
        if use_optimizer:
            self.solver_model = z3.Optimize()
        else:
            self.solver_model = z3.Solver()
        self.objective_var = self.create_real_var(name="Objective variable")
        self.g = self.build_model()
        self.maximize_output_minimize_producers_soft(use_optimizer)

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

    def maximize_output_minimize_producers_soft(self, use_optimizer=True):
        _, _, producer_output_vars = collect_vars(self.node_inputs, self.node_outputs, self.model_data.Nodes)
        prod_exprs = []
        edge_by_node_and_part = {graph_node[0]: defaultdict(list) for graph_node in self.g.nodes(data=True)}
        for edge in self.g.out_edges(data=True):
            part_name = edge[2]["part_name"]
            edge_var = edge[2]["edge_var"]
            edge_by_node_and_part[edge[0]][part_name].append(edge_var)
        for part_list in edge_by_node_and_part.values():
            for var_list in part_list.values():
                if use_optimizer:
                    all_equal = self.z3_equals(var_list)
                    if all_equal is not None:
                        self.solver_model.add_soft(all_equal)
                prod_exprs.extend(var_list)
        self.solver_model.add(self.objective_var == sum(prod_exprs))
        # Don't let objective fall to zero or else the soft constraint is trivially satisfiable
        self.solver_model.add(self.objective_var > 0)
        if use_optimizer:
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
        return g

    def absolute_diff(self, a, b):
        # This is faster than using if clauses
        tempvar = self.create_real_var(name=f"tempvar {next(self.count)} for absolute difference")
        self.solver_model.add((a - b) <= tempvar)
        self.solver_model.add(-(a - b) <= tempvar)
        return tempvar
