import itertools
import json
import re
from pathlib import Path

import pytest
import z3

from satisfactorysolver.game_data import load_game_data
from satisfactorysolver.modeler_models import (
    MachineByName,
    ModelerFileModel,
    ModelerNodeById,
    MultiMachineByName,
    PartByName,
    RecipeByName,
)
from satisfactorysolver.web_visualizer import capture_solution, write_html
from satisfactorysolver.z3_model import Z3Model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rocket_fuel_solution(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(ModelerNodeById, "_by_id", {})
    monkeypatch.setattr(ModelerNodeById, "_next_id", itertools.count())
    for registry in (MachineByName, MultiMachineByName, PartByName, RecipeByName):
        monkeypatch.setattr(registry, "_by_name", {})
    load_game_data()
    data = ModelerFileModel.model_validate_json(
        (ROOT / "rocket fuel factory.sfmd").read_text()
    )
    model = Z3Model(data, "balanced")
    assert model.solver_model.check() == z3.sat
    return capture_solution(model, model.solution_value_getter(), "Solution 1")


def test_edge_flows_match_node_rates(rocket_fuel_solution: dict) -> None:
    nodes = {n["id"]: n for n in rocket_fuel_solution["nodes"]}
    inflow: dict[tuple, float] = {}
    for edge in rocket_fuel_solution["edges"]:
        key = (edge["to"], edge["part"])
        inflow[key] = inflow.get(key, 0) + edge["value"]
    for (node_id, part), total in inflow.items():
        if isinstance(node_id, int):
            assert nodes[node_id]["inputs"][part]["value"] == pytest.approx(total)


def test_machine_counts_and_unconnected_outputs(rocket_fuel_solution: dict) -> None:
    nodes = {n["id"]: n for n in rocket_fuel_solution["nodes"]}
    # Outpost has no recipe and is not drawn
    assert 0 not in nodes
    assert nodes[4]["kind"] == "sink"
    assert nodes[5]["kind"] == "consumer"
    assert nodes[2]["kind"] == "resource"
    # Miners cap by rate, so no machine count is derived for them
    assert nodes[2]["count"] is None and nodes[2]["maxIsRate"]
    # Blender 9 makes 720 rocket fuel/min at 100/min per blender
    blender = nodes[9]
    rate = blender["outputs"]["Rocket Fuel"]["value"]
    assert blender["count"] == pytest.approx(rate / 100)
    assert blender["buildings"] == -(-rate // 100)
    # Node 22's polymer resin has no consumer, so it becomes an output node
    assert any(
        n["kind"] == "product" and n["id"] == "out:22:Polymer Resin"
        for n in rocket_fuel_solution["nodes"]
    )


def test_write_html_embeds_data(rocket_fuel_solution: dict, tmp_path: Path) -> None:
    path = write_html([rocket_fuel_solution], tmp_path / "out.html", "rocket <fuel>")
    html = path.read_text()
    assert "<title>rocket &lt;fuel&gt;</title>" in html
    payload = re.search(
        r'<script id="solver-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    ).group(1)
    assert json.loads(payload)["solutions"][0]["label"] == "Solution 1"
