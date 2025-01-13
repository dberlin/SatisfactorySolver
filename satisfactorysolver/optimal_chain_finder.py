import itertools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from rich import table, console

from satisfactorysolver.solver_helpers import ResourceLimits

logger = logging.getLogger(__name__)


class OptimalChainFinder[VarType](ABC):
    """
    Class to find optimal production chains to produce a given set of items in a given set of quantities.
    """

    def __init__(self, recipe_data):
        self.items_used = None
        self.resources_scaled = None
        self.resource_weights = {}
        self.count = itertools.count()
        self.user_given_inputs = {}
        self.user_given_outputs = {}
        self.intermediates = {}
        self.num_recipes = {}
        self.recipe_data = recipe_data
        self.products = set()
        self.ingredients = set()
        self.recipes = set()
        self.resources = set()
        self.recipes_by_output = defaultdict(set)
        self.recipes_by_input = defaultdict(set)
        self.solver_model = None

    def build_model(self, inputs, outputs):
        self.resources_scaled = self.create_real_var(name="Resources Scaled")
        self.items_used = self.create_real_var(name="Items Used")
        self.resources = set(ResourceLimits.get_resource_names())
        possibly_used_recipes = self.construct_possibly_used_recipes(outputs)
        self.construct_sets(possibly_used_recipes)
        all_items = self.resources.union(self.ingredients, self.products)
        self.construct_input_items(all_items)
        self.construct_output_items(all_items)
        self.construct_intermediate_items(all_items)
        self.construct_num_recipes()
        self.fix_input_amounts(all_items, inputs)
        self.fix_output_amounts(outputs)
        self.add_product_constraints()
        self.add_ingredient_constraints(all_items)
        self.add_resource_constraints()
        self.calculate_resource_weights()
        self.calculate_resources_scaled()
        self.calculate_item_use(all_items)
        self.add_optimization_constraints([output for (output, amount) in outputs.items() if amount == -1])

    @abstractmethod
    def solve(self):
        pass

    @abstractmethod
    def add_constraint_to_model(self, constraint, name=""):
        pass

    @abstractmethod
    def create_real_var(self, name: str):
        pass

    def get_model_result_by_var(self, var):
        return self.solver_model.model()[var]

    def construct_sets(self, all_recipes):
        for recipe in all_recipes:
            # If you want to keep converters but ignore recipes that are just misnamed producers
            # if len(recipe.Inputs) == 0 and len(recipe.Outputs) == 1 and recipe.Outputs[0].Part.Name in self.resources:
            #     continue
            for (part, amount) in recipe.Inputs:
                self.ingredients.add(part.Name)
            for (part, amount) in recipe.Outputs:
                self.products.add(part.Name)
            self.recipes.add(recipe.Name)

    def fix_input_amounts(self, all_items, inputs):
        for item in all_items:
            if item in inputs.keys():
                self.add_constraint_to_model(self.user_given_inputs[item] == inputs[item])
            else:
                self.add_constraint_to_model(self.user_given_inputs[item] == 0)

    def fix_output_amounts(self, outputs):
        for (item, amount) in outputs.items():
            self.add_constraint_to_model(self.user_given_outputs[item] == outputs[item])

    def construct_input_items(self, all_items):
        for item in all_items:
            self.user_given_inputs[item] = self.create_real_var(name=f"Input {item}")

    def construct_output_items(self, all_items):
        for item in all_items:
            self.user_given_outputs[item] = self.create_real_var(name=f"Output {item}")

    def construct_intermediate_items(self, all_items):
        for item in all_items:
            self.intermediates[item] = self.create_real_var(name=f"Intermediate {item}")

    def construct_num_recipes(self):
        for recipe in self.recipes:
            self.num_recipes[recipe] = self.create_real_var(name=f"Recipe {recipe}")

    def add_related_constraints(self, items, user_given_io, recipes_mapping, kind):
        """ All the input/output constraints are of the same form, placing some sum relationship on the intermediate
        variables.

        IE sum(all inputs needing an intermediate) == intermediate value
        or sum (all outputs producing an intermediate) == intermediate value

        This function generates them given the mappings and relationships.

        It is expected that user_given_io will usually be the opposite thing from recipes_mapping and kind.
        That is, it is normal to have add_related_constraints(products,user_given_inputs, recipes_by_output, "Outputs")

        This is because it represents a user-given input or output.
        User given inputs are added to the output value.
        User given outputs are normally added to the input value.
        """
        for item in items:
            exprs = [user_given_io[item]]
            related_recipes = recipes_mapping[item]

            for recipe in related_recipes:
                batch_time_factor = 60.0 / recipe.BatchTime
                amount = self.get_amount_for_item(getattr(recipe, kind), item)
                exprs.append(batch_time_factor * amount * self.num_recipes[recipe.Name])

            self.add_constraint_to_model(sum(exprs) == self.intermediates[item])

    def add_product_constraints(self):
        self.add_related_constraints(self.products, self.user_given_inputs, self.recipes_by_output, "Outputs")

    def add_ingredient_constraints(self, all_items):
        self.add_related_constraints(all_items, self.user_given_outputs, self.recipes_by_input, "Inputs")

    @staticmethod
    def get_amount_for_item(recipe_parts, item):
        for (part, amount) in recipe_parts:
            if part.Name == item:
                return abs(amount)
        raise KeyError(f"Item {item} not found in recipe parts {recipe_parts}")

    def add_resource_constraints(self):
        for resource in self.resources:
            self.add_constraint_to_model(self.intermediates[resource] <= ResourceLimits.get_limit_for_node(resource))

    def calculate_resource_weights(self):
        filtered_limits = {}
        for resource in self.resources:
            if "Water" not in resource:
                filtered_limits[resource] = ResourceLimits.get_limit_for_node(resource)
        avg_limit = sum(filtered_limits.values()) / len(filtered_limits)
        for resource in self.resources:
            self.resource_weights[resource] = avg_limit / ResourceLimits.get_limit_for_node(resource)

    def calculate_resources_scaled(self):
        expr = sum(self.resource_weights[resource] * self.intermediates[resource] for resource in self.resource_weights)
        self.add_constraint_to_model(expr == self.resources_scaled)

    def calculate_item_use(self, all_items):
        expr = sum(self.intermediates[item] for item in all_items)
        self.add_constraint_to_model(expr == self.items_used)

    @abstractmethod
    def add_optimization_constraints(self, outputs_to_maximize):
        pass

    def _create_and_populate_table(self, title, attr_name, model_result):
        """Creates and populates a rich table with given data."""
        generated_table = table.Table(title=title)
        generated_table.add_column("Node name", justify="left", no_wrap=True)
        generated_table.add_column("Float Value", justify="left", no_wrap=True)
        generated_table.add_column("True Value", justify="left", no_wrap=True)
        value_dict = getattr(self, attr_name)
        for key in sorted(value_dict.keys()):
            var = value_dict[key]
            result = model_result(var)
            if result is None:
                logging.critical(f"Var {var} has a None result. This should not happen.")
                continue
            if self.get_fraction_from_val(result) > 0.001:
                fraction = self.get_fraction_from_val(result)
                generated_table.add_row(str(var), str(round(float(fraction), 2)), str(result))
        return generated_table

    def print_inputs_outputs(self):
        """Print the resulting input and output values as two tables."""
        model_result = self.get_model_result_by_var
        rich_console = console.Console()
        # Create and print Inputs table
        input_table = self._create_and_populate_table("Inputs", "user_given_inputs", model_result)
        rich_console.print(input_table)
        # Create and print Outputs table
        output_table = self._create_and_populate_table("Outputs", "user_given_outputs", model_result)
        rich_console.print(output_table)
        # Create and print Intermediates table
        intermediates_table = self._create_and_populate_table("Intermediates", "intermediates", model_result)
        rich_console.print(intermediates_table)
        # Create and print Num Recipes table
        num_recipes_table = self._create_and_populate_table("Num recipes", "num_recipes", model_result)
        rich_console.print(num_recipes_table)

    @abstractmethod
    def get_fraction_from_val(self, val):
        pass

    def construct_possibly_used_recipes(self, outputs):
        """
            Generate the set of recipes that could possibly be used in making our outputs.
            This is generated by walking the graph, queueing recipes that produce the current
            recipe's inputs, iteratively, until every recipe that is part of any chain
            that can lead to our outputs is added.
        """

        # Networkx does not have multi-source BFS or DFS, and it seems silly to convert the recipe data to a graph just
        # to figure out the reachable set.
        recipe_set = set()
        node_queue = deque()
        visited = set()
        # We need the full recipes by output to start out here, and will overwrite it
        # later with the minimized set.
        for recipe in self.recipe_data:
            for (part, amount) in recipe.Outputs:
                self.recipes_by_output[part.Name].add(recipe)

        # Seed the queue with recipes that can output our outputs
        for needed_output in outputs.keys():
            for recipe in self.recipes_by_output.get(needed_output, []):
                node_queue.append(recipe)
                visited.add(recipe)

        # Go until we have totally emptied the queue
        while node_queue:
            # Take the next recipe off the queue
            # For any input part whose recipes we haven't explored or queued
            # See what produces that part
            # Add them to the queue
            recipe = node_queue.popleft()
            recipe_set.add(recipe)
            for (part, amount) in recipe.Inputs:
                for can_produce_input in self.recipes_by_output[part.Name]:
                    if can_produce_input not in visited:
                        node_queue.append(can_produce_input)
                        visited.add(can_produce_input)

        # Reconstruct the recipe input/output mapping using only the minimized recipe set
        self.recipes_by_output = defaultdict(set)
        self.recipes_by_input = defaultdict(set)
        for recipe in self.recipe_data:
            if recipe not in recipe_set:
                continue
            for (part, amount) in recipe.Outputs:
                self.recipes_by_output[part.Name].add(recipe)
            for (part, amount) in recipe.Inputs:
                self.recipes_by_input[part.Name].add(recipe)
        return recipe_set
