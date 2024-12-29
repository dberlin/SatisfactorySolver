import argparse
import logging

import json
import math
import sys
from fractions import Fraction

import rich.logging
from deepmerge import always_merger

from satisfactorysolver.modeler_models import AllDataModel, ModelerFileModel

logging.basicConfig(level=logging.INFO, handlers=[rich.logging.RichHandler(rich_tracebacks=True)], )
logger = logging.getLogger(__name__)


def load_game_data():
    game_data = json.load(open("game_data.json", "r"))
    additional_data = json.load(open("additional_data.json", "r"))
    merged = always_merger.merge(game_data, additional_data)
    return AllDataModel.model_validate(merged)


def load_model_file(name):
    model_data = json.load(open(name, "r"))
    return ModelerFileModel.model_validate(model_data)


parser = argparse.ArgumentParser(prog='Satisfactory Model Solver')
parser.add_argument('-v', '--verbose', action='count', help='Increase verbosity', default=0)
parser.add_argument('-s', '--solver', choices=['z3', 'pyomo', 'cvc5'], default='z3',
                    help='Which solver to use. Default is z3.')
parser.add_argument('-c', '--condition', choices=['optimal', 'balanced', 'practical'], default='balanced',
                    help="Which condition to try to optimize for."
                         ""
                         "Optimal means any solution that maximizes output, whether practical to achieve in the game "
                         "or stable or not."
                         ""
                         "Balanced means any optimal solution that balances outputs from a single source to multiple "
                         "sinks, to the degree it does not destroy optimality.  This is easier achieve in game,"
                         "as it matches how splitters and mergers function."
                         ""
                         "Practical means any optimal solution that does not involve splitting inputs and outputs "
                         "into smaller than eighths.")
parser.add_argument('filename')
args = parser.parse_args()
game_data = load_game_data()
logger.setLevel(logging.INFO - (args.verbose * 10))
model_data = load_model_file(args.filename)


def find_objective_max(model, objective_var, sat_val):
    low, high = 0, 2 ** 32
    last_sat_val = None
    while not math.isclose(low, high, abs_tol=0.001) and low < high:
        mid = (low + high) / 2

        logger.debug(f"Checking value {mid}")
        result = model.check(objective_var >= mid)
        logger.debug(f"Is sat:{result == sat_val}")
        if result == sat_val:
            last_sat_val = mid
            model.push()
        if result != sat_val:
            logger.debug(f"High is now {mid}")
            high = mid
        else:
            logger.debug(f"Low is now {mid + 1}")
            low = mid + 0.001
    return status, last_sat_val


if args.solver == 'cvc5':
    import cvc5.pythonic
    from satisfactorysolver.cvc5_model import CVC5Model


    # This currently requires using python -O because we fail the context checks in cvc5 but not z3. This is related
    # to the fact that we don't refresh the list of variables in between solves which is fine for z3, but makes the
    # cvc5 pythonic interface unhappy.
    # Doing so is incredibly annoying. Just turn on -O till they figure this out.
    def cvc5_all_smt(s, initial_terms):
        def block_term(s, m, terms, i):
            val = m[terms[i]]
            t = terms[i]
            # Because CVC5 and Z3 are infinite precision, we have to use some tolerance on the variables Otherwise it
            # will enumerate an infinite number of models that have epsilon differences in practice
            s.add(cvc5.pythonic.Or(cvc5.pythonic.And(t > val, t - val > 1), cvc5.pythonic.And(t <= val, val - t > 1)))

        def fix_term(s, m, terms, i):
            res = m[terms[i]]
            s.add(terms[i] == Fraction(res.numerator(), res.denominator()))

        def all_smt_rec(terms):
            if cvc5.pythonic.sat == s.check():
                m = s.model()
                yield m
                for i in range(len(terms)):
                    s.push()
                    s.check()
                    block_term(s, m, terms, i)
                    for j in range(i):
                        fix_term(s, m, terms, j)
                    yield from all_smt_rec(terms[i:])
                    s.pop()

        yield from all_smt_rec(list(initial_terms))


    model = CVC5Model(model_data, args.condition)

    logger.debug('=====')
    logger.debug(model.solver_model)
    logger.debug('=====')
    status = model.solver_model.check()
    obj_var_idx = 0

    # If the model is satisfiable, binary search for the max objective var
    if status == cvc5.pythonic.sat:
        last_status, last_sat_val = find_objective_max(model.solver_model, model.objective_var, cvc5.pythonic.sat)
        # Model can be popped to last sat point if necessary
        if last_status != cvc5.pythonic.sat:
            model.solver_model.pop()

        status = model.solver_model.check()
        model_result = model.solver_model.model()
        # Fix the objective value at the optimal one
        obj_result = model_result[model.objective_var]
        model.solver_model.add(model.objective_var == Fraction(obj_result.numerator(), obj_result.denominator()))
        # Fix the objective value at the optimal one
        model.solver_model.add(model.objective_var == model_result[model.objective_var])
        logging.info(f"Enumerating all optimal solutions")
        if __debug__:
            logger.critical("Debugging is on, so this will probably crash, see the comment about -O above")
        for m in cvc5_all_smt(model.solver_model, model.edge_vars):
            model.print_inputs_outputs()
    else:
        logging.error(f"No solution found, status: {status}")
if args.solver == 'z3':
    import z3
    from satisfactorysolver.z3_model import Z3Model

    model = Z3Model(model_data, args.condition)


    def z3_all_smt(s, initial_terms):
        def block_term(s, m, t):
            val = m.eval(t, model_completion=True)
            # The joy of infinite precision
            s.add(z3.Or(z3.And(t > val, t - val > 1), z3.And(t <= val, val - t > 1)))

        def fix_term(s, m, t):
            s.add(t == m.eval(t, model_completion=True))

        def all_smt_rec(terms):
            if z3.sat == s.check():
                m = s.model()
                yield m
                for i in range(len(terms)):
                    s.push()
                    block_term(s, m, terms[i])
                    for j in range(i):
                        fix_term(s, m, terms[j])
                    yield from all_smt_rec(terms[i:])
                    s.pop()

        yield from all_smt_rec(list(initial_terms))


    status = model.solver_model.check()
    if status == z3.sat:
        if args.condition == 'balanced':
            model.print_inputs_outputs()
        else:
            model_result = model.solver_model.model()
            last_status, last_sat_val = find_objective_max(model.solver_model, model.objective_var, z3.sat)
            if last_status != z3.sat:
                model.solver_model.pop()
            model.solver_model.check()
            model_result = model.solver_model.model()
            # Fix the objective value at the optimal one
            model.solver_model.add(model.objective_var == model_result[model.objective_var])
            logging.info(f"Enumerating all optimal solutions")
            for m in z3_all_smt(model.solver_model, model.edge_vars):
                model.print_inputs_outputs()
    else:
        logging.error(f"No solution found, status: {status}")
if args.solver == 'pyomo':
    from satisfactorysolver.pyomo_model import PyomoModel
    from pyomo.contrib import appsi

    model = PyomoModel(model_data)
    if logger.level <= logging.DEBUG:
        model.pprint()
    # Debugging example
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
    # TODO(integrate solving into the interface)
    results = opt.solve(model.model)
    # Other non-appsi solvers can be used too, but they are not persistent
    # import pyomo.environ as pyo
    # solver = pyo.SolverFactory('bonmin')
    # results = solver.solve(model, tee=True, keepfiles=True)
    rich.print(results)  # from IPython.display import display
    if logger.level <= logging.DEBUG:
        model.pprint()
    model.print_inputs_outputs()
