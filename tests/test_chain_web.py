import json
import threading
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

import pytest

from satisfactorysolver.chain_web_server import ChainSession, make_handler
from satisfactorysolver.game_data import load_game_data
from satisfactorysolver.modeler_models import (
    MachineByName,
    MultiMachineByName,
    PartByName,
    RecipeByName,
)
from satisfactorysolver.optimal_chain_displayer import parse_targets
from satisfactorysolver.web_visualizer import capture_chain_solution
from satisfactorysolver.z3_optimal_chain_finder import Z3OptimalChainFinder


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> ChainSession:
    for registry in (MachineByName, MultiMachineByName, PartByName, RecipeByName):
        monkeypatch.setattr(registry, "_by_name", {})
    data = load_game_data()
    return ChainSession(
        data.Recipes,
        {part.Name for part in data.Parts},
        Z3OptimalChainFinder,
        parse_targets,
    )


def used_recipes(response: dict) -> set[str]:
    return {recipe["name"] for recipe in response["recipes"] if recipe["used"]}


def test_chain_edges_conserve_every_node_rate(session: ChainSession) -> None:
    finder = Z3OptimalChainFinder(session.recipes)
    finder.build_model({}, {"Rocket Fuel": 1000})
    assert finder.solve()
    solution = capture_chain_solution(finder, "Chain")

    flow_out = defaultdict(float)
    flow_in = defaultdict(float)
    for edge in solution["edges"]:
        flow_out[edge["from"], edge["part"]] += edge["value"]
        flow_in[edge["to"], edge["part"]] += edge["value"]
    for node in solution["nodes"]:
        for part, rate in node.get("outputs", {}).items():
            assert flow_out[node["id"], part] == pytest.approx(rate["value"])
        for part, rate in node.get("inputs", {}).items():
            assert flow_in[node["id"], part] == pytest.approx(rate["value"])
    assert flow_in["out:Rocket Fuel", "Rocket Fuel"] == pytest.approx(1000)


def test_disabling_a_recipe_resolves_without_it(session: ChainSession) -> None:
    default = session.solve({"outputs": "Rocket Fuel=1000"})
    assert default["error"] is None
    assert "Rocket Fuel" in used_recipes(default)

    without = session.solve(
        {"outputs": "Rocket Fuel=1000", "disabled": ["Rocket Fuel"]}
    )
    assert without["error"] is None
    assert "Rocket Fuel" not in used_recipes(without)
    assert "Nitro Rocket Fuel" in used_recipes(without)
    # Disabled recipes stay listed so they can be switched back on.
    listed = {recipe["name"]: recipe for recipe in without["recipes"]}
    assert listed["Rocket Fuel"]["enabled"] is False


def test_solve_reports_bad_targets_and_infeasibility(session: ChainSession) -> None:
    assert "unknown item" in session.solve({"outputs": "Rocket Fool=1"})["error"]
    assert "at least one output" in session.solve({"outputs": ""})["error"]
    infeasible = session.solve(
        {"outputs": "Rocket Fuel=1", "disabled": ["Rocket Fuel", "Nitro Rocket Fuel"]}
    )
    assert infeasible["solution"] is None
    assert "No optimal production chain" in infeasible["error"]


def test_server_serves_page_and_solves(session: ChainSession) -> None:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(session, "<html>page</html>")
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/") as page:
            assert page.read() == b"<html>page</html>"
        request = urllib.request.Request(
            base + "/api/solve",
            data=json.dumps({"outputs": "Iron Plate=30"}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            body = json.load(response)
        assert body["error"] is None
        assert any(n["id"] == "out:Iron Plate" for n in body["solution"]["nodes"])
    finally:
        server.shutdown()
        server.server_close()


def test_item_names_ignore_case(session: ChainSession) -> None:
    response = session.solve(
        {"outputs": "alclad aluminum sheet=10", "inputs": "WATER=5"}
    )
    assert response["error"] is None
    ids = {node["id"] for node in response["solution"]["nodes"]}
    assert "out:Alclad Aluminum Sheet" in ids
    inputs, outputs = parse_targets(
        ["water=5"], ["ALCLAD ALUMINUM SHEET=10"], {"Water", "Alclad Aluminum Sheet"}
    )
    assert inputs == {"Water": 5} and outputs == {"Alclad Aluminum Sheet": 10}
    with pytest.raises(ValueError, match="duplicate output"):
        parse_targets([], ["water=1", "Water=2"], {"Water"})


def test_concurrent_solves_share_one_solver_thread(session: ChainSession) -> None:
    # z3 is not thread-safe; overlapping requests used to crash the server.
    requests = [
        {"outputs": "Rocket Fuel=1000", "disabled": ["Rocket Fuel"]},
        {"outputs": "Iron Plate=30"},
        {"outputs": "Rocket Fuel=100"},
        {"outputs": "Iron Plate=10", "disabled": ["Iron Plate"]},
    ] * 3
    with ThreadPoolExecutor(6) as pool:
        responses = list(pool.map(session.solve, requests))
    assert all(response["error"] is None for response in responses)
    assert {
        name
        for response in responses[0::4]
        for name in used_recipes(response)
        if "Rocket Fuel" in name
    } == {"Nitro Rocket Fuel"}
