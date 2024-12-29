from abc import ABC, abstractmethod
from collections import defaultdict
from fractions import Fraction

import networkx as nx
from rich import table, console

from satisfactorysolver.solver_helpers import ResourceLimits


class SolverModel(ABC):
    """
    SolverModel is an abstract base class that provides the foundational structure for creating
    and managing a constraint solver model for our problems.
    It depends on a z3 compatible python interface (such as cvc5's pythonic interface)

    SolverModel has functions to handle initialization of input and output variables
    for nodes and enforces constraints like resource limits, input-output proportions, and maximum
    production capacity.

    The implementation also  includes utility methods for creating tables, presenting input-output data,
    and handling special node cases.

    :ivar node_inputs: Mapping of node IDs to their respective input variables.
    :ivar node_outputs: Mapping of node IDs to their respective output variables.
    :ivar solver_model: Instance of the solver backend model, used for defining constraints and performing optimization.
    :ivar model_data: Reference to the modeler file data containing
    :ivar edge_vars: List of solver variables representing values carried by edges between nodes in the graph.
    """
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
        """
        Creates a directed graph representation of the modeler node graph using networkx.
        Edges are annotated with solver source and destination variables.
        Solver edge variables are created here.

        :raises KeyError: If node inputs or outputs cannot be accessed in the process.

        :rtype: networkx.DiGraph
        :return: A directed graph where nodes represent network elements and edges
            represent directional dependencies between them.
        """
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
        """
        Creates and applies edge constraints for a given modeler graph. The method ensures that the sum of edge
        variables for each part matches corresponding input and output constraints associated with a node.

        :param g: Graph object representing the modeler nodes.
        :return: None
        """
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
        """
        Creates constraints in the solver model for maximal possible resources
        producible in the game.  This helps bound the problem even when
        the user has not specified resource limits.

        :raises KeyError: If the solver model or resource limit lookup fails due
            to a missing node or invalid configuration.
        """
        for node in self.model_data.Nodes:
            if len(node.Inputs) == 0 and len(node.Outputs) == 1:
                limit = ResourceLimits.get_limit_for_node(node)
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
        """
        Constrains the output amount of a given recipe in a particular node to its corresponding
        input amount, maintaining the ratio restrictions that exist for the recipe input/output

        :param node: Modeler node, this is usually representing a machine.
        :param recipe: Recipe being used.
        :return: None. This method modifies the solver model by adding constraints and does
            not return any value.
        :rtype: None
        """
        for input_part in recipe.Inputs:
            input_var = self.node_inputs[node.Id][input_part.Part.Name]
            batches_per_minute = 60.0 / recipe.BatchTime
            input_rate_per_minute = abs(input_part.Amount) * batches_per_minute
            for output_part in recipe.Outputs:
                output_var = self.node_outputs[node.Id][output_part.Part.Name]
                output_rate_per_minute = abs(output_part.Amount) * batches_per_minute
                self.solver_model.add(output_var <= ((input_var / input_rate_per_minute) * output_rate_per_minute))

    def constrain_io_to_same_batch_count(self, node, recipe):
        """
        Constrains the input and output flow of a node to ensure that the number of
        batches processed or produced per minute remains consistent across all
        inputs and outputs. This is achieved by generating expressions that relate
        the flow rates of materials.

        :param node: Modeler node, this is usually representing a machine.
        :param recipe: Recipe being used.
        :return: None
        """
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
        """
        Restricts the output of a given node to its maximum production capacity.

        :param node: Modeler node, this is usually representing a machine.
        :param recipe: Recipe being used.
        :return: None
        """
        node_max = node.Max or (node.Machine.DefaultMax if node.Machine else None)
        if node_max is None:
            return
        in_ppm = recipe.Machine.ShowPpm if recipe.Machine else False
        for part in recipe.Outputs:
            output_var = self.node_outputs[node.Id][part.Part.Name]
            batches_per_minute = 60.0 / recipe.BatchTime
            output_max = (node_max if in_ppm else node_max * abs(part.Amount) * batches_per_minute)
            self.solver_model.add(output_var <= float(output_max))
