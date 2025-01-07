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

import networkx as nx
from pyomo import environ as pyo
from pyomo.common.numeric_types import nonpyomo_leaf_types
from pyomo.core.expr import ExpressionReplacementVisitor

from satisfactorysolver.solver_helpers import collect_vars, ResourceLimits


class PyomoModel:
    def __init__(self, model_data, condition):
        self.model = pyo.ConcreteModel()
        self.node_inputs = {}
        self.node_outputs = {}
        self.model_data = model_data
        self.condition = condition
        self.g = None
        self.build_model()

    def build_model(self):

        self.create_modeler_node_outputs()
        self.create_modeler_node_inputs()

        self.g = self.create_modeler_node_graph()
        self.create_edge_constraints()
        self.create_produced_resource_constraints()

        for node in self.model_data.Nodes:
            recipe = node.Recipe
            # Some nodes have no recipes because they are outposts or ...
            # Ignore them
            if recipe is None:
                continue

            self.constrain_output_amount_to_input_amount(node, recipe)
            self.constrain_io_to_same_batch_count(node, recipe)

            self.constrain_output_to_node_max(node, recipe)

        # model = try_maximize_input_minimize_producers(model, model_data, g)
        self.try_maximize_output()

    def calculate_sum_of_differences(self):
        difference_exprs = []
        # Minimize difference between outgoing edges of a given node that all have the same part.
        for graph_node in self.g.nodes(data="node_model"):
            node = graph_node[1]
            dest_vars_by_part = defaultdict(list)
            src_var_by_part = {}
            for (_, _, edge_data) in self.g.out_edges(node.Id, data=True):
                dest_vars_by_part[edge_data["part_name"]].append(edge_data["dest_var"])
                src_var_by_part[edge_data["part_name"]] = edge_data["src_var"]
            for (part_name, var_list) in dest_vars_by_part.items():
                # No point if one or no edge
                if len(var_list) <= 1:
                    continue
                src_var_name = src_var_by_part[part_name].name
                # normally we would generate the straight sum of absolute differences between the edges here. You can
                # do that if you want to use a non-linear solver, and ignore all of this.  In fact, you  can just use
                # product of outputs as you can see in the non-linear formulation, and the product will be maximized
                # by making the outputs as even as possible - there is a quadratic relationship between the distance
                # of two numbers and their product. So for an output of say 780 split two ways, 390*390 is greater
                # than the product of any other valid numbers.
                #
                # For linear formulations, it is trickier - because even if you use sum of differences, absolute
                # value is non-linear on its own. But we can convert it into a variable plus constraints that give us
                # the same thing and are linear.
                #
                for (current_var, other_var) in itertools.combinations(var_list, 2):
                    abs_diff_helper = self.linear_absolute_diff(current_var, other_var, f"edge abs helper for source {src_var_name} edge {current_var.name} + {other_var.name}")
                    difference_exprs.append(abs_diff_helper)
        sum_of_differences = sum(difference_exprs)
        return sum_of_differences

    def linear_absolute_diff(self, a, b, name):
        """"
        This works by converting abs(X) into a variable X' and the constraints
        X <= X'
        -X <= X'
        If X > 0, then -X <= X' is always true. Because we use X' in the subtraction below, we will find a value for X' that is as small as possible
        while still meeting the "X <= X'" constraint.  This will make it have the value of X
        If X < 0, then the opposite holds.
        if X == 0, both constraints are satisfied already, and making X as small as possible will ensure it stays at zero.
        so if you imagine a=10, b=5
        we want to simulate abs(10-5)
        -(10-5) <= X' (second constraint) is already true, as it's -5, and remain true for any positive value of X'
        (10-5) <= X' (first constraint) will become true when X' >=5, and optimal at exactly 5.
        if you imagine abs(5-10)
        (5-10) <= X' (first constraint) is always true, as it's -5, and will remain true for any positive value of X'
        -(5-10) <= X' (second constraint) will become true when X' >= 5, and optimal at exactly 5
        """
        abs_diff_var = pyo.Var(domain=pyo.NonNegativeReals, name=name)
        self.model.add_component(name, abs_diff_var)
        diff_expr= a - b
        positive_constraint = pyo.Constraint(expr=diff_expr <= abs_diff_var)
        negative_constraint = pyo.Constraint(expr=-diff_expr <= abs_diff_var)
        self.model.add_component(name, positive_constraint)
        self.model.add_component(f"{-name}", negative_constraint)
        return abs_diff_var
    
    def try_maximize_output(self):
        USE_LINEAR_FORMULATION = False

        @self.model.Objective(sense=pyo.maximize)
        def maximize_output(m):
            # Maximize the inputs while minimizing produced resources
            # We do not try to maximize sink nodes since they are unconstrained
            all_input_vars, all_output_vars, producer_output_vars = collect_vars(self.node_inputs, self.node_outputs,
                                                                                 self.model_data.Nodes)

            if USE_LINEAR_FORMULATION:
                sum_of_differences = self.calculate_sum_of_differences()
                return sum(x for x in all_output_vars) - sum_of_differences - sum(x for x in producer_output_vars)
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

    def try_maximize_input_minimize_producers(self):
        USE_LINEAR_FORMULATION = True

        @self.model.Objective(sense=pyo.maximize)
        def maximize_needed_input_with_minimum_resource_use(m):
            # Maximize the inputs while minimizing produced resources
            # We do not try to maximize sink nodes since they are unconstrained
            all_input_vars, all_output_vars, producer_output_vars = collect_vars(self.node_inputs, self.node_outputs,
                                                                                 self.model_data.Nodes)

            if USE_LINEAR_FORMULATION:
                sum_of_differences = self.calculate_sum_of_differences()
                return sum(x for x in all_input_vars) - sum_of_differences - sum(x for x in producer_output_vars)
            else:
                return pyo.prod(x for x in all_input_vars) / pyo.prod(x for x in producer_output_vars)

    def constrain_output_to_node_max(self, node, recipe):
        # Outputs are separately constrained by any max that is present
        node_max = node.Max or (node.Machine.DefaultMax if node.Machine else None)
        if not node_max:
            return
        if node_max is not None:
            # See if machine is in terms of PPM or batches per minute
            in_ppm = recipe.Machine.ShowPpm if recipe.Machine else False
            for part in recipe.Outputs:
                output_var = self.node_outputs[node.Id][part.Part.Name]
                batches_per_minute = 60.0 / recipe.BatchTime
                # if it's in PPM, use it
                # if it's in machines, multiply how much we can get out of a single machine times the max number of machines
                output_max = node_max if in_ppm else node_max * abs(part.Amount) * batches_per_minute
                constraint = pyo.Constraint(expr=defractionize(output_var <= output_max))
                self.model.add_component(f"Node {node.Id}.{node.Name} max constraint for output {part.Part.Name}",
                                         constraint)

    def constrain_io_to_same_batch_count(self, node, recipe):
        # Inputs and outputs are constrained to have the same batches per minute as each other
        num_batch_per_min_exprs = []
        for part in recipe.Inputs:
            input_var = self.node_inputs[node.Id][part.Part.Name]
            amount_needed_per_batch = abs(part.Amount)
            num_batch_per_min_exprs.append((input_var / amount_needed_per_batch, part.Part.Name))
        for part in recipe.Outputs:
            output_var = self.node_outputs[node.Id][part.Part.Name]
            amount_made_per_batch = abs(part.Amount)
            num_batch_per_min_exprs.append((output_var / amount_made_per_batch, part.Part.Name))
        # Make all expressions equivalent to the first one
        if len(num_batch_per_min_exprs) > 1:
            first, rest = num_batch_per_min_exprs[0], num_batch_per_min_exprs[1:]
            for (expr, name) in rest:
                constraint = pyo.Constraint(expr=defractionize(expr == first[0]))
                self.model.add_component(
                    f"Node {node.Id}.{node.Name} all input and output same number of batches constraint for {name}",
                    constraint)

    def constrain_output_amount_to_input_amount(self, node, recipe):
        # Handle the case that the outputs are bound by the inputs
        # this is not always true, some things are producers like miners that have no input.
        for input_part in recipe.Inputs:
            input_var = self.node_inputs[node.Id][input_part.Part.Name]
            batches_per_minute = 60.0 / recipe.BatchTime
            input_rate_per_minute = abs(input_part.Amount) * batches_per_minute
            for output_part in recipe.Outputs:
                output_var = self.node_outputs[node.Id][output_part.Part.Name]
                output_rate_per_minute = abs(output_part.Amount) * batches_per_minute
                max_output = (input_var / input_rate_per_minute) * output_rate_per_minute
                # We make as much as the input allows us to
                constraint = pyo.Constraint(expr=defractionize(output_var <= max_output))
                self.model.add_component(
                    f"Node {node.Id}.{node.Name} output per minute constraint for input {input_part.Part.Name}, output {output_part.Part.Name}",
                    constraint)

    def create_edge_constraints(self):
        for graph_node in self.g.nodes(data="node_model"):
            node = graph_node[1]
            in_edge_vars_by_part = defaultdict(list)
            for (_, _, edge_data) in self.g.in_edges(node.Id, data=True):
                in_edge_vars_by_part[edge_data["part_name"]].append(edge_data["edge_var"])
            for (part_name, var_list) in in_edge_vars_by_part.items():
                # The input variable must be the same as the sum of the incoming edges
                constraint = pyo.Constraint(expr=defractionize(self.node_inputs[node.Id][part_name] == sum(var_list)))
                self.model.add_component(f"Incoming edge sum constraint for {node.Id}.{node.Name}.{part_name}",
                                         constraint)
            out_edge_vars_by_part = defaultdict(list)
            for (_, _, edge_data) in self.g.out_edges(node.Id, data=True):
                out_edge_vars_by_part[edge_data["part_name"]].append(edge_data["edge_var"])
            for (part_name, var_list) in out_edge_vars_by_part.items():
                # The output variable must be the same as the sum of the outgoing edges
                constraint = pyo.Constraint(expr=defractionize(self.node_outputs[node.Id][part_name] == sum(var_list)))
                self.model.add_component(
                    f"Outgoing edge sum constraint for real part for {node.Id}.{node.Name}.{part_name}", constraint)

    # Represent an edge between modeler nodes, annotated with the model variables
    def create_modeler_node_graph(self):
        g = nx.DiGraph()
        for node in self.model_data.Nodes:
            g.add_node(node.Id, node_model=node)
        for node in self.model_data.Nodes:
            for (input_part_name, input_node_list) in node.InputNodes:
                for input_node in input_node_list:
                    dest_var = self.node_inputs[node.Id][input_part_name]
                    src_var = self.node_outputs[input_node.Id][input_part_name]
                    edge_var = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, None),
                                       name=f"edge from {src_var.name} to {dest_var.name}")
                    self.model.add_component(edge_var.name, edge_var)
                    g.add_edge(input_node.Id, node.Id, part_name=input_part_name, src_var=src_var, dest_var=dest_var,
                               edge_var=edge_var)
        return g

    def create_modeler_node_inputs(self):
        # Create an input variable for each regular node's input
        for node in self.model_data.Nodes:
            if node.Name == "AWESOME Sink":
                # Create an input variable for each AWESOME Sink's actual input, since it has no extra inputs
                self.node_inputs[node.Id] = {}
                for (input_kind, input_node_list) in node.InputNodesById.items():
                    var = pyo.Var(domain=pyo.NonNegativeReals, name=f"{node.Id}.{node.Name}.SinkInput.{input_kind}",
                                  bounds=(0, None))
                    self.model.add_component(var.name, var)
                    self.node_inputs[node.Id][input_kind] = var
            else:
                # Create an input variable for each regular node's recipe input (so we can calculate the expected values
                # even when edges do not exist)
                num_input_nodes = len(node.Inputs)
                self.node_inputs[node.Id] = {}
                for i in range(num_input_nodes):
                    var = pyo.Var(domain=pyo.NonNegativeReals,
                                  name=f"{node.Id}.{node.Name}.Input.{node.Inputs[i].Part.Name}", bounds=(0, None))
                    self.model.add_component(var.name, var)
                    self.node_inputs[node.Id][node.Inputs[i].Part.Name] = var

    def create_modeler_node_outputs(self):
        # Create an output variable for each node's output
        for node in self.model_data.Nodes:
            num_output_nodes = len(node.Outputs)
            self.node_outputs[node.Id] = {}
            for i in range(num_output_nodes):
                var = pyo.Var(domain=pyo.NonNegativeReals,
                              name=f"{node.Id}.{node.Name}.Output.{node.Outputs[i].Part.Name}", bounds=(0, None))
                self.model.add_component(var.name, var)
                self.node_outputs[node.Id][node.Outputs[i].Part.Name] = var

    # Create constraints to bound produced resources by max possible amount in the world
    def create_produced_resource_constraints(self):
        for node in self.model_data.Nodes:
            # Producers have no inputs but 1 output
            if len(node.Inputs) == 0 and len(node.Outputs) == 1:
                # get the resource limit and add a bound
                limit = ResourceLimits.get_limit_for_node(node)
                if limit == -1:
                    continue
                constraint = pyo.Constraint(
                    expr=defractionize(self.node_outputs[node.Id][node.Outputs[0].Part.Name] <= limit))
                self.model.add_component(
                    f"Node {node.Id}.{node.Name} output constraint for resource {node.Outputs[0].Part.Name}",
                    constraint)

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
