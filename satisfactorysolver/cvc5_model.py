import itertools
import operator
from collections import defaultdict
from functools import reduce

from cvc5 import pythonic as cvc5

from satisfactorysolver.solver_helpers import collect_vars
from satisfactorysolver.solver_model import SolverModel


class CVC5Model(SolverModel):
    def __init__(self, model_data, condition):
        super().__init__(model_data)
        self.count = itertools.count()
        # self.cvc5_model = cvc5.SolverFor(logic="QF_NRA")
        self.solver_model = cvc5.Solver()
        self.solver_model.setOption("produce-models", "true")
        # self.cvc5_model.setOption("output", "post-asserts")
        # self.cvc5_model.setOption("verbosity", "5")
        # self.cvc5_model.setOption("stats-every-query", "true")
        self.objective_var = self.create_real_var(name="Objective variable")
        self.g = self.build_model()
        self.maximize_output_minimize_producers(condition)

    def create_real_var(self, name):
        new_var = cvc5.Real(name)
        self.solver_model.add(new_var >= 0)
        return new_var

    def absolute_diff(self, a, b):
        # This is faster than using if clauses
        tempvar = self.create_real_var(name=f"tempvar {next(self.count)} for absolute difference")
        self.solver_model.add((a - b) <= tempvar)
        self.solver_model.add(-(a - b) <= tempvar)
        return tempvar  # return cvc5.If(a - b >= 0, a - b, b - a)

    def maximize_output_minimize_producers(self, condition):
        _, _, producer_output_vars = collect_vars(self.node_inputs, self.node_outputs, self.model_data.Nodes)
        sum_exprs = []
        penalty_exprs = []
        edge_by_node_and_part = {graph_node[0]: defaultdict(list) for graph_node in self.g.nodes(data=True)}
        for edge in self.g.out_edges(data=True):
            part_name = edge[2]["part_name"]
            edge_var = edge[2]["edge_var"]
            edge_by_node_and_part[edge[0]][part_name].append(edge_var)
        for part_list in edge_by_node_and_part.values():
            for var_list in part_list.values():
                sum_exprs.append(reduce(operator.add, var_list))
            if condition == 'balanced':
                # Penalize non-equal outputs
                for var_list in part_list.values():
                    for pair in itertools.combinations(var_list, 2):
                        penalty_exprs.append(self.absolute_diff(pair[0], pair[1]))
            else:
                penalty_exprs = [0]
        self.solver_model.add(self.objective_var == sum(sum_exprs) - sum(penalty_exprs))
        # Don't let objective fall to zero or else most constranits are satisfiable
        self.solver_model.add(self.objective_var > 0)

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
