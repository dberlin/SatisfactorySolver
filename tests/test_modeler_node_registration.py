import itertools
from pathlib import Path

import pytest
from pyomo.contrib import appsi

from satisfactorysolver.game_data import load_game_data
from satisfactorysolver.modeler_models import (
    MachineByName,
    ModelerFileModel,
    ModelerNodeById,
    MultiMachineByName,
    PartByName,
    RecipeByName,
)
from satisfactorysolver.pyomo_model import PyomoModel


@pytest.fixture
def game_data(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ModelerNodeById, "_by_id", {})
    monkeypatch.setattr(ModelerNodeById, "_next_id", itertools.count())
    for registry in (MachineByName, MultiMachineByName, PartByName, RecipeByName):
        monkeypatch.setattr(registry, "_by_name", {})
    root = Path(__file__).resolve().parents[1]
    load_game_data()
    return root


def test_instant_scrap_recycles_water_after_node_revalidation(game_data: Path) -> None:
    data = ModelerFileModel.model_validate_json(
        (game_data / "instant scrap 10.sfmd").read_text()
    )
    # Nesting existing nodes must not change the file's connection targets.
    data = ModelerFileModel(Version=data.Version, Data=data.Nodes)
    model = PyomoModel(data, "balanced")
    result = appsi.solvers.Highs().solve(model.model)

    assert result.termination_condition == appsi.base.TerminationCondition.optimal
    # Six water units in, five recycled: 10 fresh water supports 300 scrap/min.
    water, scrap = data.Nodes
    assert model.node_outputs[water.Id]["Water"].value == pytest.approx(10)
    assert model.node_inputs[scrap.Id]["Water"].value == pytest.approx(60)
    assert model.node_outputs[scrap.Id]["Water"].value == pytest.approx(50)
    assert model.node_outputs[scrap.Id]["Aluminum Scrap"].value == pytest.approx(300)
