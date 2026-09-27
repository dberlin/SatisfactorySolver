import math
from collections import Counter
from fractions import Fraction

import pytest

from satisfactorysolver.chain_web_server import ChainSession
from satisfactorysolver.game_data import load_game_data
from satisfactorysolver.modeler_models import (
    MachineByName,
    MultiMachineByName,
    PartByName,
    RecipeByName,
)
from satisfactorysolver.optimal_chain_displayer import main, parse_targets
from satisfactorysolver.pyomo_optimal_chain_finder import PyomoOptimalChainFinder
from satisfactorysolver.resource_nodes import (
    ResourceNode,
    limits_near,
    load_resource_nodes,
)
from satisfactorysolver.solver_helpers import ResourceLimits
from satisfactorysolver.z3_optimal_chain_finder import Z3OptimalChainFinder

BACKENDS = (Z3OptimalChainFinder, PyomoOptimalChainFinder)


@pytest.fixture
def recipes(monkeypatch: pytest.MonkeyPatch):
    for registry in (MachineByName, MultiMachineByName, PartByName, RecipeByName):
        monkeypatch.setattr(registry, "_by_name", {})
    return load_game_data().Recipes


def test_nodes_add_up_to_map_wide_limits() -> None:
    nodes = load_resource_nodes()
    totals = limits_near(nodes, 0, 0, math.inf)
    for resource, total in totals.items():
        assert total == ResourceLimits.get_limit_for_part(resource), resource
    counts = Counter((n.resource, n.purity) for n in nodes if not n.well)
    assert (counts["Iron Ore", "impure"], counts["Iron Ore", "normal"]) == (39, 42)
    assert counts["Iron Ore", "pure"] == 46


def test_limits_count_only_nodes_in_range() -> None:
    nodes = [
        ResourceNode("Iron Ore", "pure", False, 0, 0, 500),
        ResourceNode("Iron Ore", "impure", False, 300, 400, 0),  # 500 m away
        ResourceNode("Crude Oil", "normal", False, 0, 600, 0),
        ResourceNode("Nitrogen Gas", "pure", True, 100, 0, 0),
        ResourceNode("Water", "pure", True, 0, 0, 0),
    ]
    near = limits_near(nodes, 0, 0, 500)
    assert near["Iron Ore"] == 1200 + 300
    assert near["Crude Oil"] == 0
    assert near["Nitrogen Gas"] == 300
    assert near["Coal"] == 0
    assert "Water" not in near
    assert limits_near(nodes, 0, 0, 700)["Crude Oil"] == 300


@pytest.mark.parametrize("finder_class", BACKENDS)
def test_local_limits_cap_extraction_but_not_supplies(recipes, finder_class) -> None:
    limits = {"Iron Ore": Fraction(60)}

    def iron_ore_used(inputs):
        finder = finder_class(recipes)
        finder.build_model(inputs, {"Iron Ingot": -1}, limits)
        assert finder.solve()
        return finder.get_fraction_from_val(
            finder.get_model_result_by_var(finder.intermediates["Iron Ore"])
        )

    assert iron_ore_used({}) == pytest.approx(60, rel=1e-6)
    # Ore supplied from elsewhere does not count against the nodes in range.
    assert iron_ore_used({"Iron Ore": 30}) == pytest.approx(90, rel=1e-6)


def test_local_scarcity_weights(recipes) -> None:
    nodes = [
        ResourceNode("Iron Ore", "pure", False, 0, 0, 0),
        ResourceNode("Copper Ore", "impure", False, 0, 0, 0),
    ]
    finder = Z3OptimalChainFinder(recipes)
    finder.build_model({}, {"Iron Plate": 10}, limits_near(nodes, 0, 0, 1))
    weights = finder.resource_weights
    # Only the resources in range set the scale: here the average is 750.
    assert weights["Iron Ore"] == Fraction(750, 1200)
    assert weights["Copper Ore"] == Fraction(750, 300)
    # None in range, so none can be extracted.
    assert weights["Coal"] == 0
    assert weights["Water"] < Fraction(1, 10**5)


def test_session_applies_point_and_radius(recipes) -> None:
    session = ChainSession(
        recipes,
        {part.Name for part in load_game_data().Parts},
        Z3OptimalChainFinder,
        parse_targets,
    )
    # Pure iron nodes in the south west, with no sulfur anywhere near.
    point = (-2520.33, -1250.01)
    expected = limits_near(load_resource_nodes(), *point, 300)
    nearby = {
        "outputs": "Iron Plate=-1",
        "near": f"{point[0]},{point[1]}",
        "radius": 300,
    }
    response = session.solve(nearby)
    assert response["error"] is None
    assert response["limits"] == {k: float(v) for k, v in expected.items()}
    assert response["limits"]["Iron Ore"] >= 1200
    assert response["limits"]["Sulfur"] == 0
    extracted = sum(
        rate["value"]
        for n in response["solution"]["nodes"]
        if n["kind"] == "resource"
        for item, rate in n["outputs"].items()
        if item == "Iron Ore"
    )
    assert extracted == pytest.approx(response["limits"]["Iron Ore"])
    assert session.solve({"outputs": "Iron Plate=1"})["limits"] is None
    assert "both" in session.solve({"outputs": "Iron Plate=1", "near": "1,2"})["error"]
    assert "X,Y" in session.solve({**nearby, "near": "nowhere"})["error"]


def test_cli_requires_point_and_radius_together(capsys) -> None:
    with pytest.raises(SystemExit):
        main(["--output", "Iron Plate=10", "--near", "0,0"])
    assert "together" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["--output", "Iron Plate=10", "--near", "0,0", "--radius", "-5"])
    assert "positive" in capsys.readouterr().err


@pytest.mark.parametrize("finder_class", BACKENDS)
def test_supplied_inputs_are_free(recipes, finder_class) -> None:
    finder = finder_class(recipes)
    finder.build_model({"Iron Ore": 60}, {"Iron Ingot": 60})
    assert finder.solve()
    cost = finder.get_fraction_from_val(
        finder.get_model_result_by_var(finder.resources_scaled)
    )
    assert cost == pytest.approx(0, abs=1e-9)
