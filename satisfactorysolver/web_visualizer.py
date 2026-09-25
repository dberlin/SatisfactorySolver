"""
Render solved modeler graphs as a standalone HTML page, in the style of the
SatisfactoryTools production planner visualization (a vis-network graph of
machines connected by item flows, plus building and item tables).
"""

import json
import math
from collections import defaultdict
from fractions import Fraction
from html import escape
from pathlib import Path

AWESOME_SINK_NAME = "AWESOME Sink"


def to_fraction(result) -> Fraction:
    """Convert a solver result value (z3/cvc5 rational, float or None) to a Fraction."""
    if result is None:
        return Fraction(0)
    if isinstance(result, Fraction):
        return result
    if isinstance(result, (int, float)):
        return Fraction(result).limit_denominator(10**6)
    text = str(result)
    try:
        return Fraction(text)
    except ValueError:
        # z3 prints irrational approximations with a trailing '?'
        return Fraction(float(text.rstrip("?"))).limit_denominator(10**6)


def _rate(value: Fraction) -> dict:
    return {"value": float(value), "exact": str(value)}


def _machine_usage(recipe, rates_in, rates_out):
    """
    Work out how many machines running recipe the solved rates represent.

    :return: (machine count as a Fraction, or None when the machine's Max is a
        per-minute rate cap rather than a machine count)
    """
    if recipe is None or recipe.Machine.ShowPpm:
        return None
    batches_per_minute = Fraction(60) / recipe.BatchTime
    for part, amount in recipe.Outputs:
        if amount:
            return rates_out.get(part.Name, Fraction(0)) / (
                abs(amount) * batches_per_minute
            )
    for part, amount in recipe.Inputs:
        if amount:
            return rates_in.get(part.Name, Fraction(0)) / (
                abs(amount) * batches_per_minute
            )
    return None


def _recipe_kind(recipe):
    if not recipe.Inputs:
        return "resource"
    if not recipe.Outputs:
        return "consumer"
    return "machine"


def _node_kind(node):
    if node.Name == AWESOME_SINK_NAME:
        return "sink"
    if node.Recipe is None:
        return None
    return _recipe_kind(node.Recipe)


def _node_entry(
    node_id, name, kind, recipe, machine_name, node_max, rates_in, rates_out
):
    count = _machine_usage(recipe, rates_in, rates_out)
    return {
        "id": node_id,
        "name": name,
        "kind": kind,
        "recipe": recipe.Name if recipe else None,
        "alternate": bool(recipe and recipe.Alternate),
        "machine": machine_name,
        "max": float(node_max) if node_max is not None else None,
        "maxIsRate": bool(recipe and recipe.Machine.ShowPpm),
        "count": float(count) if count is not None else None,
        "buildings": math.ceil(count - Fraction(1, 10**6))
        if count is not None and count > 0
        else None,
        "inputs": {part: _rate(v) for part, v in rates_in.items()},
        "outputs": {part: _rate(v) for part, v in rates_out.items()},
    }


def capture_solution(model, value_of, label: str) -> dict:
    """
    Snapshot the current solution of a SolverModel as plain JSON-able data.

    :param model: Solved SolverModel (must have its modeler graph in model.g).
    :param value_of: Callable returning the solver's value for a variable.
    :param label: Name for this solution in the viewer.
    """
    graph = model.g
    nodes = []
    edges = []
    connected_in = {(dst, edge["part_name"]) for _, dst, edge in graph.edges(data=True)}
    connected_out = {
        (src, edge["part_name"]) for src, _, edge in graph.edges(data=True)
    }

    for node in model.model_data.Nodes:
        kind = _node_kind(node)
        if kind is None:
            continue
        rates_in = {
            part: to_fraction(value_of(var))
            for part, var in model.node_inputs.get(node.Id, {}).items()
        }
        rates_out = {
            part: to_fraction(value_of(var))
            for part, var in model.node_outputs.get(node.Id, {}).items()
        }
        recipe = node.Recipe
        machine_name = None
        if node.Machine is not None:
            machine_name = node.Machine.Name
        elif recipe is not None:
            machine_name = recipe.Machine.Name
        nodes.append(
            _node_entry(
                node.Id,
                node.Name,
                kind,
                recipe,
                machine_name,
                node.Max,
                rates_in,
                rates_out,
            )
        )

        # Flows that leave or enter the modeled graph get virtual endpoints,
        # like the product/input nodes in SatisfactoryTools.
        for part, rate in rates_out.items():
            if rate > 0 and (node.Id, part) not in connected_out:
                virtual_id = f"out:{node.Id}:{part}"
                nodes.append(
                    {"id": virtual_id, "name": part, "kind": "product", "part": part}
                )
                edges.append(
                    {"from": node.Id, "to": virtual_id, "part": part, **_rate(rate)}
                )
        for part, rate in rates_in.items():
            if rate > 0 and (node.Id, part) not in connected_in:
                virtual_id = f"in:{node.Id}:{part}"
                nodes.append(
                    {"id": virtual_id, "name": part, "kind": "input", "part": part}
                )
                edges.append(
                    {"from": virtual_id, "to": node.Id, "part": part, **_rate(rate)}
                )

    for src, dst, edge in graph.edges(data=True):
        rate = to_fraction(value_of(edge["edge_var"]))
        edges.append({"from": src, "to": dst, "part": edge["part_name"], **_rate(rate)})

    return {"label": label, "nodes": nodes, "edges": edges}


def capture_chain_solution(finder, label: str) -> dict:
    """
    Snapshot a solved OptimalChainFinder as the same JSON-able data as capture_solution.

    The chain model only knows how many of each recipe run and the total rate of
    each item, not which machine feeds which. As in SatisfactoryTools, every
    producer of an item feeds every consumer of it in proportion to the
    consumer's share of the total.
    """

    def value_of(var) -> Fraction:
        return to_fraction(finder.get_model_result_by_var(var))

    recipes_by_name = {recipe.Name: recipe for recipe in finder.recipe_data}
    nodes = []
    producers = defaultdict(list)
    consumers = defaultdict(list)

    for name in sorted(finder.num_recipes):
        count = value_of(finder.num_recipes[name])
        if count <= 0:
            continue
        recipe = recipes_by_name[name]
        batches_per_minute = count * Fraction(60) / recipe.BatchTime
        rates_in = {
            part.Name: abs(amount) * batches_per_minute
            for part, amount in recipe.Inputs
        }
        rates_out = {
            part.Name: abs(amount) * batches_per_minute
            for part, amount in recipe.Outputs
        }
        node_id = f"recipe:{name}"
        nodes.append(
            _node_entry(
                node_id,
                name,
                _recipe_kind(recipe),
                recipe,
                recipe.Machine.Name,
                None,
                rates_in,
                rates_out,
            )
        )
        for part, rate in rates_out.items():
            producers[part].append((node_id, rate))
        for part, rate in rates_in.items():
            consumers[part].append((node_id, rate))

    for items, kind, prefix, flows in (
        (finder.user_given_inputs, "input", "in", producers),
        (finder.user_given_outputs, "product", "out", consumers),
    ):
        for part in sorted(items):
            rate = value_of(items[part])
            if rate <= 0:
                continue
            node_id = f"{prefix}:{part}"
            nodes.append({"id": node_id, "name": part, "kind": kind, "part": part})
            flows[part].append((node_id, rate))

    edges = []
    for part, sources in producers.items():
        total = sum(rate for _, rate in sources)
        for source, produced in sources:
            for sink, consumed in consumers.get(part, []):
                rate = produced * consumed / total
                if rate > 0:
                    edges.append(
                        {"from": source, "to": sink, "part": part, **_rate(rate)}
                    )

    return {"label": label, "nodes": nodes, "edges": edges}


def render_html(solutions: list[dict], title: str, live: dict | None = None) -> str:
    """
    Render the viewer page.

    :param live: When given, the page runs in live mode: it shows the recipe
        sidebar and re-solves through the /api/solve endpoint, starting from
        these settings.
    """
    payload = json.dumps({"title": title, "solutions": solutions, "live": live})
    # Keep '</script>' sequences in item names from closing the data block.
    payload = payload.replace("</", "<\\/")
    return _TEMPLATE.replace("__TITLE__", escape(title)).replace("__DATA__", payload)


def write_html(solutions: list[dict], path: str | Path, title: str) -> Path:
    """Write the solutions to a self-contained HTML viewer at path."""
    path = Path(path)
    path.write_text(render_html(solutions, title), encoding="utf-8")
    return path


_TEMPLATE = (Path(__file__).parent / "web_visualizer_template.html").read_text(
    encoding="utf-8"
)
