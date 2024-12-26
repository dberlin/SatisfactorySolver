import json

import rich
from deepmerge import always_merger
from pyomo.contrib import appsi

from modeler_models import AllDataModel, ModelerFileModel
from solver_models import build_model


def load_game_data():
    game_data = json.load(open("game_data.json", "r"))
    additional_data = json.load(open("additional_data.json", "r"))
    merged = always_merger.merge(game_data, additional_data)
    return AllDataModel.model_validate(merged)


def load_model_file(name):
    model_data = json.load(open(name, "r"))
    return ModelerFileModel.model_validate(model_data)


game_data = load_game_data()

model_data = load_model_file("simple pure ingot.sfmd")
# model_data = load_model_file("simple coal generator.sfmd")
# model_data = load_model_file("rocket fuel factory.sfmd")
# model_data = load_model_file("simple rocket fuel.sfmd")
# model_data = load_model_file("rocket fuel without loops.sfmd")

model, graph = build_model(model_data)
model.pprint()

# results = solver.solve(model, tee=True, keepfiles=True)
# model.node_inputs[1]["Iron Ore"].set_value(390.0)
# model.node_outputs[3]["Iron Ore"].set_value(780.0)
# model.node_inputs[1]["Water"].set_value(222.8571429)
# model.node_outputs[0]["Water"].set_value(445.7142858)
# model.node_inputs[2]["Iron Ore"].set_value(390.0)
# model.node_inputs[2]["Water"].set_value(222.8571429)
# model.node_outputs[1]["Iron Ingot"].set_value(11.14285714 * 5 * 13)
# model.node_outputs[2]["Iron Ingot"].set_value(11.14285714 * 5 * 13)
# model.node_inputs[1]["Iron Ore"].set_value(22.04)
# model.node_outputs[2]["Iron Ore"].set_value(22.04)
# model.node_inputs[3]["Iron Ingot"].set_value
# for c in model.component_objects(ctype=pyo.Constraint):
#     if c.slack() < 0:  # constraint is not met
#         print(f'Constraint {c.name} is not satisfied')
#         c.display()  # show the evaluation of c
#         c.pprint()  # show the construction of c
#         print()

# If appsi is not installed, run "pyomo build-extensions" in our virtual environment
# Ipopt is non-linear and locally optimal, which should suffice for our type of model.
opt = appsi.solvers.Ipopt()
# Highs is linear-only, which means you can't use the product of variables in the objective
# opt = appsi.solvers.Highs()
results = opt.solve(model)
# Other non-appsi solvers can be used too, but they are not persistent
# import pyomo.environ as pyo
# solver = pyo.SolverFactory('bonmin')
# results = solver.solve(model, tee=True, keepfiles=True)
rich.print(results)  # from IPython.display import display
model.pprint()


# Print the resulting values as two tables
from rich import table
from rich import console

input_table = table.Table(title="Node inputs")
input_table.add_column("Node name", justify="left", no_wrap=True)
input_table.add_column("Value", justify="left", no_wrap=True)
for node_input in model.node_inputs:
    for input_var in model.node_inputs[node_input].values():
        input_table.add_row(input_var.name, str(input_var.value))
console = console.Console()
console.print(input_table)

output_table = table.Table(title="Node outputs")
output_table.add_column("Node name", justify="left", no_wrap=True)
output_table.add_column("Value", justify="left", no_wrap=True)
for node_output in model.node_outputs:
    for output_var in model.node_outputs[node_output].values():
        output_table.add_row(output_var.name, str(output_var.value))
console.print(output_table)
