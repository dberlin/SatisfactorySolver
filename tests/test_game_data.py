import json
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from satisfactorysolver.game_data import load_game_data
from satisfactorysolver.modeler_models import (
    MachineByName,
    MultiMachineByName,
    PartByName,
    RecipeByName,
)


def _write_data(directory: Path, game_data: object, additional_data: object) -> None:
    directory.mkdir()
    _ = (directory / "game_data.json").write_text(json.dumps(game_data))
    _ = (directory / "additional_data.json").write_text(json.dumps(additional_data))


@pytest.fixture(autouse=True)
def isolate_model_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    for registry in (MachineByName, MultiMachineByName, PartByName, RecipeByName):
        monkeypatch.setattr(registry, "_by_name", {})


def test_overlay_replaces_named_recipe_fields_and_preserves_references(
    tmp_path: Path,
) -> None:
    data_directory = tmp_path / "data"
    _write_data(
        data_directory,
        {
            "Machines": [{"Name": "Test Constructor", "AveragePower": "-4"}],
            "MultiMachines": [],
            "Parts": [
                {"Name": "Test Ore", "SinkPoints": 1},
                {"Name": "Test Ingot", "SinkPoints": 2},
            ],
            "Recipes": [
                {
                    "Name": "Test Smelting",
                    "Machine": "Test Constructor",
                    "Tier": "test-tier",
                    "BatchTime": "6",
                    "Parts": [
                        {"Part": "Test Ore", "Amount": "-1"},
                        {"Part": "Test Ingot", "Amount": "1"},
                    ],
                }
            ],
        },
        {
            "Parts": [{"Name": "Test Byproduct", "SinkPoints": 3}],
            "Recipes": [
                {
                    "Name": "Test Smelting",
                    "BatchTime": "3",
                    "Parts": [
                        {"Part": "Test Ore", "Amount": "-2"},
                        {"Part": "Test Ingot", "Amount": "5"},
                        {"Part": "Test Byproduct", "Amount": "1"},
                    ],
                }
            ],
        },
    )

    data = load_game_data(data_directory)

    recipe = next(recipe for recipe in data.Recipes if recipe.Name == "Test Smelting")
    assert recipe.Tier == "test-tier"
    assert recipe.BatchTime == 3
    assert len(recipe.Parts) == 3
    assert {part.Part.Name: part.Amount for part in recipe.Parts} == {
        "Test Ore": Fraction(-2),
        "Test Ingot": Fraction(5),
        "Test Byproduct": Fraction(1),
    }
    ingot_output = next(
        part for part in recipe.Outputs if part.Part.Name == "Test Ingot"
    )
    assert Fraction(60, recipe.BatchTime) * ingot_output.Amount == 100
    assert RecipeByName.get(recipe.Name) is recipe

    retained_parts = {part.Name: part for part in data.Parts}
    assert set(retained_parts) == {"Test Ore", "Test Ingot", "Test Byproduct"}
    for recipe_part in recipe.Parts:
        assert recipe_part.Part is retained_parts[recipe_part.Part.Name]
        assert PartByName.get(recipe_part.Part.Name) is recipe_part.Part


def test_loads_packaged_steam_data_outside_current_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    data = load_game_data()

    multi_machines = {machine.Name: machine for machine in data.MultiMachines}
    assert multi_machines["Dimensional Depot Uploader"].DefaultMax is None
    assert multi_machines["AWESOME Sink"].DefaultMax is None
    assert multi_machines["Space Elevator"].Capacities == []
    assert MultiMachineByName.get("Space Elevator") is multi_machines["Space Elevator"]

    retained_parts = {part.Name: part for part in data.Parts}
    iron_ore_recipe = next(
        recipe for recipe in data.Recipes if recipe.Name == "Iron Ore"
    )
    assert RecipeByName.get("Iron Ore") is iron_ore_recipe
    assert iron_ore_recipe.Parts[0].Part is retained_parts["Iron Ore"]
    assert PartByName.get("Iron Ore") is retained_parts["Iron Ore"]
    assert iron_ore_recipe.Machine is MultiMachineByName.get("Miner")


def test_rejects_non_numeric_default_max(tmp_path: Path) -> None:
    data_directory = tmp_path / "data"
    _write_data(
        data_directory,
        {
            "Machines": list[object](),
            "MultiMachines": [
                {
                    "Name": "Invalid Maximum",
                    "DefaultMax": "not-a-number",
                    "Capacities": list[object](),
                }
            ],
            "Parts": list[object](),
            "Recipes": list[object](),
        },
        {},
    )

    with pytest.raises(ValidationError, match="DefaultMax"):
        _ = load_game_data(data_directory)
