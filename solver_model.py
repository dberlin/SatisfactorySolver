from abc import ABC, abstractmethod
from collections import defaultdict
from fractions import Fraction

import networkx as nx
from rich import table, console

from solver_helpers import ResourceLimits


class SolverModel(ABC):
    def __init__(self, model_data):
        self.node_inputs = {}
        self.node_outputs = {}
        self.solver_model = None
        self.model_data = model_data
        self.edge_vars = []

    @abstractmethod
    def create_real_var(self, name):
        pass

    @staticmethod
    def _create_and_populate_table(title, data, model_result):
        """Creates and populates a rich table with given data."""
        generated_table = table.Table(title=title)
        generated_table.add_column("Node name", justify="left", no_wrap=True)
        generated_table.add_column("Float Value", justify="left", no_wrap=True)
        generated_table.add_column("True Value", justify="left", no_wrap=True)
        for node, variables in data.items():
            for var in variables.values():
                result = model_result[var]
                fraction = Fraction(str(result))
                generated_table.add_row(str(var), str(float(fraction)), str(result))
        return generated_table

    def print_inputs_outputs(self):
        """Print the resulting input and output values as two tables."""
        model_result = self.solver_model.model()
        rich_console = console.Console()

        # Create and print Node Inputs table
        input_table = self._create_and_populate_table("Node Inputs", self.node_inputs, model_result)
        rich_console.print(input_table)

        # Create and print Node Outputs table
        output_table = self._create_and_populate_table("Node Outputs", self.node_outputs, model_result)
        rich_console.print(output_table)

    def create_modeler_node_graph(self):
        g = nx.DiGraph()
        for node in self.model_data.Nodes:
            g.add_node(node.Id, node_model=node)
        for node in self.model_data.Nodes:
            for (input_part_name, input_node_list) in node.InputNodes:
                for input_node in input_node_list:
                    dest_var = self.node_inputs[node.Id][input_part_name]
                    src_var = self.node_outputs[input_node.Id][input_part_name]
                    edge_var = self.create_real_var(name=f"edge from {src_var} to {dest_var}")
                    self.edge_vars.append(edge_var)
                    g.add_edge(input_node.Id, node.Id, part_name=input_part_name, src_var=src_var, dest_var=dest_var,
                               edge_var=edge_var, )
        return g

    def create_edge_constraints(self, g):
        for graph_node in g.nodes(data="node_model"):
            node = graph_node[1]
            in_edge_vars_by_part = defaultdict(list)
            for (_, _, edge_data) in g.in_edges(node.Id, data=True):
                in_edge_vars_by_part[edge_data["part_name"]].append(edge_data["edge_var"])
            for part_name, var_list in in_edge_vars_by_part.items():
                self.solver_model.add(sum(var_list) == self.node_inputs[node.Id][part_name])
            out_edge_vars_by_part = defaultdict(list)
            for (_, _, edge_data) in g.out_edges(node.Id, data=True):
                out_edge_vars_by_part[edge_data["part_name"]].append(edge_data["edge_var"])
            for part_name, var_list in out_edge_vars_by_part.items():
                self.solver_model.add(sum(var_list) == self.node_outputs[node.Id][part_name])

    def create_produced_resource_constraints(self):
        for node in self.model_data.Nodes:
            if len(node.Inputs) == 0 and len(node.Outputs) == 1:
                limit = ResourceLimits.get_limit_for_node(node)
                if limit == -1:
                    continue
                self.solver_model.add(self.node_outputs[node.Id][node.Outputs[0].Part.Name] <= limit)

    # Eh, this might be too clever
    def _create_node_map_helper(self, nodes, attribute_name, kind_prefix, special_case=None):

        for node in nodes:
            target_dict = self.node_inputs if attribute_name == "Inputs" else self.node_outputs
            target_dict[node.Id] = {}

            if special_case and node.Name == special_case["name"]:
                for input_kind, input_node_list in node.InputNodesById.items():
                    var = self.create_real_var(f"{node.Id}.{node.Name}.{special_case['prefix']}.{input_kind}")
                    target_dict[node.Id][input_kind] = var
            else:
                attributes = getattr(node, attribute_name)
                for attr in attributes:
                    part_name = attr.Part.Name
                    var = self.create_real_var(f"{node.Id}.{node.Name}.{kind_prefix}.{part_name}")
                    target_dict[node.Id][part_name] = var

    def create_modeler_node_inputs(self):
        self._create_node_map_helper(self.model_data.Nodes, attribute_name="Inputs", kind_prefix="Input",
            special_case={"name": "AWESOME Sink", "prefix": "SinkInput"})

    def create_modeler_node_outputs(self):
        self._create_node_map_helper(self.model_data.Nodes, attribute_name="Outputs", kind_prefix="Output")

    def constrain_output_amount_to_input_amount(self, node, recipe):
        for input_part in recipe.Inputs:
            input_var = self.node_inputs[node.Id][input_part.Part.Name]
            batches_per_minute = 60.0 / recipe.BatchTime
            input_rate_per_minute = abs(input_part.Amount) * batches_per_minute
            for output_part in recipe.Outputs:
                output_var = self.node_outputs[node.Id][output_part.Part.Name]
                output_rate_per_minute = abs(output_part.Amount) * batches_per_minute
                self.solver_model.add(output_var <= ((input_var / input_rate_per_minute) * output_rate_per_minute))

    def constrain_io_to_same_batch_count(self, node, recipe):
        num_batch_per_min_exprs = []
        for part in recipe.Inputs:
            input_var = self.node_inputs[node.Id][part.Part.Name]
            amount_needed_per_batch = abs(part.Amount)
            num_batch_per_min_exprs.append(input_var / float(amount_needed_per_batch))
        for part in recipe.Outputs:
            output_var = self.node_outputs[node.Id][part.Part.Name]
            amount_made_per_batch = abs(part.Amount)
            num_batch_per_min_exprs.append(output_var / float(amount_made_per_batch))
        if len(num_batch_per_min_exprs) > 1:
            first, rest = num_batch_per_min_exprs[0], num_batch_per_min_exprs[1:]
            for expr in rest:
                self.solver_model.add(expr == first)

    def constrain_output_to_node_max(self, node, recipe):
        node_max = node.Max or (node.Machine.DefaultMax if node.Machine else None)
        if node_max is None:
            return
        in_ppm = recipe.Machine.ShowPpm if recipe.Machine else False
        for part in recipe.Outputs:
            output_var = self.node_outputs[node.Id][part.Part.Name]
            batches_per_minute = 60.0 / recipe.BatchTime
            output_max = (node_max if in_ppm else node_max * abs(part.Amount) * batches_per_minute)
            self.solver_model.add(output_var <= float(output_max))
