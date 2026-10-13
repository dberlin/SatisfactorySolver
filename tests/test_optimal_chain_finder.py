from fractions import Fraction

import pytest

from satisfactorysolver.modeler_models import (
    MachineByName,
    MachineModel,
    MultiMachineByName,
    PartByName,
    PartModel,
    RecipeByName,
    RecipeModel,
)
from satisfactorysolver.optimal_chain_finder import InputLimit, SurplusPolicy
from satisfactorysolver.pyomo_optimal_chain_finder import PyomoOptimalChainFinder
from satisfactorysolver.z3_optimal_chain_finder import Z3OptimalChainFinder

BACKENDS = (Z3OptimalChainFinder, PyomoOptimalChainFinder)


@pytest.fixture(autouse=True)
def isolate_model_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    for registry in (MachineByName, MultiMachineByName, PartByName, RecipeByName):
        monkeypatch.setattr(registry, "_by_name", {})


@pytest.fixture
def recipes() -> set[RecipeModel]:
    MachineModel.model_validate({"Name": "Test Machine"})
    for name in (
        "Iron Ore",
        "Copper Ore",
        "Plate",
        "Catalyst",
        "Widget",
        "Unobtainium",
    ):
        PartModel.model_validate({"Name": name})

    recipe_values = (
        {
            "Name": "Iron Source",
            "Parts": [{"Part": "Iron Ore", "Amount": "1"}],
            "Machine": "Test Machine",
            "BatchTime": "60",
        },
        {
            "Name": "Copper Source",
            "Parts": [{"Part": "Copper Ore", "Amount": "1"}],
            "Machine": "Test Machine",
            "BatchTime": "60",
        },
        {
            "Name": "Efficient Plate",
            "Parts": [
                {"Part": "Iron Ore", "Amount": "-1"},
                {"Part": "Plate", "Amount": "1"},
            ],
            "Machine": "Test Machine",
            "BatchTime": "60",
        },
        {
            "Name": "Inefficient Plate",
            "Parts": [
                {"Part": "Copper Ore", "Amount": "-2"},
                {"Part": "Plate", "Amount": "1"},
            ],
            "Machine": "Test Machine",
            "BatchTime": "60",
            "Alternate": True,
        },
        {
            "Name": "Catalyst Widget",
            "Parts": [
                {"Part": "Catalyst", "Amount": "-2"},
                {"Part": "Widget", "Amount": "1"},
            ],
            "Machine": "Test Machine",
            "BatchTime": "60",
        },
    )
    return {RecipeModel.model_validate(value) for value in recipe_values}


def solved_value(finder, variable) -> float:
    value = finder.get_model_result_by_var(variable)
    return float(finder.get_fraction_from_val(value))


@pytest.mark.parametrize("backend", BACKENDS)
def test_fixed_output_minimizes_weighted_resources(backend, recipes) -> None:
    finder = backend(recipes)
    finder.build_model({}, {"Plate": Fraction(10)})

    assert finder.solve() is True
    assert solved_value(finder, finder.num_recipes["Efficient Plate"]) == pytest.approx(
        10
    )
    assert solved_value(
        finder, finder.num_recipes["Inefficient Plate"]
    ) == pytest.approx(0)
    assert solved_value(finder, finder.intermediates["Iron Ore"]) == pytest.approx(10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_equal_resource_chains_prefer_fewer_machines(backend, recipes) -> None:
    # Like packaging and unpackaging fuel: the same resources, more machines.
    slow = RecipeModel.model_validate(
        {
            "Name": "Slow Plate",
            "Parts": [
                {"Part": "Iron Ore", "Amount": "-1"},
                {"Part": "Plate", "Amount": "1"},
            ],
            "Machine": "Test Machine",
            "BatchTime": "600",
            "Alternate": True,
        }
    )
    for inputs in ({}, {"Catalyst": Fraction(1)}):
        finder = backend(recipes | {slow})
        finder.build_model(inputs, {"Plate": Fraction(10)})

        assert finder.solve() is True
        assert solved_value(finder, finder.num_recipes["Slow Plate"]) == pytest.approx(
            0
        )
        assert solved_value(finder, finder.machines) == pytest.approx(20)


@pytest.mark.parametrize("backend", BACKENDS)
def test_marked_output_is_maximized_with_all_recipes_available(
    backend, recipes
) -> None:
    finder = backend(recipes)
    finder.build_model({}, {"Plate": Fraction(-1)})

    assert finder.solve() is True
    expected = 92100 + 36900 / 2
    assert solved_value(finder, finder.user_given_outputs["Plate"]) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_external_input_is_conserved_and_cannot_be_manufactured(
    backend, recipes
) -> None:
    supplied = backend(recipes)
    supplied.build_model({"Catalyst": Fraction(2)}, {"Widget": Fraction(1)})

    assert supplied.solve() is True
    assert solved_value(
        supplied, supplied.user_given_inputs["Catalyst"]
    ) == pytest.approx(2)
    assert solved_value(
        supplied, supplied.user_given_outputs["Widget"]
    ) == pytest.approx(1)

    missing = backend(recipes)
    missing.build_model({}, {"Widget": Fraction(1)})

    assert missing.solve() is False


@pytest.mark.parametrize("backend", BACKENDS)
def test_excess_input_is_left_unused_instead_of_output(backend, recipes) -> None:
    finder = backend(recipes)
    finder.build_model({"Catalyst": Fraction(10)}, {"Widget": Fraction(1)})

    assert finder.solve() is True
    assert solved_value(finder, finder.user_given_inputs["Catalyst"]) == pytest.approx(
        2
    )
    assert solved_value(finder, finder.user_given_outputs["Catalyst"]) == pytest.approx(
        0
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_excess_resource_input_is_left_unused(backend, recipes) -> None:
    finder = backend(recipes)
    finder.build_model({"Iron Ore": Fraction(100)}, {"Plate": Fraction(10)})

    assert finder.solve() is True
    assert solved_value(finder, finder.user_given_inputs["Iron Ore"]) == pytest.approx(
        10
    )
    assert solved_value(finder, finder.user_given_outputs["Iron Ore"]) == pytest.approx(
        0
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_input_limit_caps_total_use_without_supplying_anything(
    backend, recipes
) -> None:
    finder = backend(recipes)
    finder.build_model({"Iron Ore": InputLimit(10)}, {"Plate": Fraction(20)})

    assert finder.solve() is True
    assert solved_value(finder, finder.intermediates["Iron Ore"]) == pytest.approx(10)
    assert solved_value(finder, finder.user_given_inputs["Iron Ore"]) == pytest.approx(
        0
    )
    assert solved_value(
        finder, finder.num_recipes["Inefficient Plate"]
    ) == pytest.approx(10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_input_limit_that_does_not_bind_changes_nothing(backend, recipes) -> None:
    # A cap is not a supply: capped copper must not become free and win.
    finder = backend(recipes)
    finder.build_model({"Copper Ore": InputLimit(1000)}, {"Plate": Fraction(10)})

    assert finder.solve() is True
    assert solved_value(finder, finder.num_recipes["Efficient Plate"]) == pytest.approx(
        10
    )
    assert solved_value(
        finder, finder.user_given_inputs["Copper Ore"]
    ) == pytest.approx(0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_water_costs_nothing(backend, recipes) -> None:
    finder = backend(recipes)
    finder.build_model({}, {"Plate": Fraction(1)})

    assert finder.resource_weights["Water"] == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_input_limit_bounds_a_maximized_output(backend, recipes) -> None:
    recipes = {recipe for recipe in recipes if recipe.Name != "Inefficient Plate"}
    finder = backend(recipes)
    finder.build_model({"Iron Ore": InputLimit(10)}, {"Plate": Fraction(-1)})

    assert finder.solve() is True
    assert solved_value(finder, finder.user_given_outputs["Plate"]) == pytest.approx(10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_output_without_a_recipe_is_infeasible(backend, recipes) -> None:
    finder = backend(recipes)
    finder.build_model({}, {"Unobtainium": Fraction(1)})

    assert finder.solve() is False


@pytest.mark.parametrize("backend", BACKENDS)
def test_unbounded_maximum_fails_without_accessing_a_solution(backend) -> None:
    MachineModel.model_validate({"Name": "Test Machine"})
    PartModel.model_validate({"Name": "Free Widget"})
    free_recipe = RecipeModel.model_validate(
        {
            "Name": "Free Widget Recipe",
            "Parts": [{"Part": "Free Widget", "Amount": "1"}],
            "Machine": "Test Machine",
            "BatchTime": "60",
        }
    )
    finder = backend({free_recipe})
    finder.build_model({}, {"Free Widget": Fraction(-1)})

    assert finder.solve() is False


@pytest.mark.parametrize("backend", BACKENDS)
def test_build_model_can_be_reused_for_a_new_problem(backend, recipes) -> None:
    finder = backend(recipes)
    finder.build_model({}, {"Plate": Fraction(1)})
    assert finder.solve() is True

    finder.build_model({}, {"Plate": Fraction(2)})

    assert finder.solve() is True
    assert solved_value(finder, finder.user_given_outputs["Plate"]) == pytest.approx(2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_small_positive_values_are_kept_in_tables(backend, recipes, capsys) -> None:
    finder = backend(recipes)
    finder.build_model({}, {"Plate": Fraction(1, 10000)})
    assert finder.solve() is True

    finder.print_inputs_outputs()
    output = capsys.readouterr().out
    assert "Output Plate" in output
    assert "1/10000" in output or "0.0001" in output


@pytest.fixture
def sludge_recipes() -> set[RecipeModel]:
    """A cheap widget leaving unsinkable sludge, and a dear one leaving none."""
    MachineModel.model_validate({"Name": "Test Machine"})
    MachineModel.model_validate({"Name": "Packager"})
    for part in (
        {"Name": "Iron Ore", "SinkPoints": 1},
        {"Name": "Widget", "SinkPoints": 10},
        {"Name": "Gadget", "SinkPoints": 10},
        {"Name": "Canister", "SinkPoints": 5},
        {"Name": "Packaged Sludge", "SinkPoints": 5},
        {"Name": "Sludge", "SinkPoints": 0, "Fluid": True},
        {"Name": "Uranium Waste", "SinkPoints": 0},
    ):
        PartModel.model_validate(part)

    def recipe(name, parts, machine="Test Machine"):
        return RecipeModel.model_validate(
            {
                "Name": name,
                "Parts": [{"Part": part, "Amount": amount} for part, amount in parts],
                "Machine": machine,
                "BatchTime": "60",
            }
        )

    return {
        recipe("Iron Source", [("Iron Ore", "1")]),
        recipe("Cheap Widget", [("Iron Ore", "-1"), ("Widget", "1"), ("Sludge", "1")]),
        recipe("Dear Widget", [("Iron Ore", "-3"), ("Widget", "1")]),
        recipe("Canister", [("Iron Ore", "-1"), ("Canister", "1")]),
        recipe(
            "Packaged Sludge",
            [("Sludge", "-1"), ("Canister", "-1"), ("Packaged Sludge", "1")],
            machine="Packager",
        ),
        recipe(
            "Hot Gadget", [("Iron Ore", "-1"), ("Gadget", "1"), ("Uranium Waste", "1")]
        ),
    }


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("surplus", "cheap", "packaged", "ore"),
    [
        # Sludge leaves freely.
        (SurplusPolicy.IGNORED, 10, 0, 10),
        # Ore for each widget and canister still beats the dear widget.
        (SurplusPolicy.PACKAGED, 10, 10, 20),
        (SurplusPolicy.FORBIDDEN, 0, 0, 30),
    ],
)
def test_unsinkable_surplus_policy(
    backend, sludge_recipes, surplus, cheap, packaged, ore
) -> None:
    finder = backend(sludge_recipes)
    finder.build_model({}, {"Widget": Fraction(10)}, surplus=surplus)

    assert finder.solve() is True
    assert solved_value(finder, finder.num_recipes["Cheap Widget"]) == pytest.approx(
        cheap
    )
    # Packaging is only reachable, so only modeled, when surplus is packaged.
    packages = finder.user_given_outputs.get("Packaged Sludge")
    assert (solved_value(finder, packages) if packages is not None else 0) == (
        pytest.approx(packaged)
    )
    assert solved_value(finder, finder.intermediates["Iron Ore"]) == pytest.approx(ore)
    if surplus is not SurplusPolicy.IGNORED:
        assert solved_value(
            finder, finder.user_given_outputs["Sludge"]
        ) == pytest.approx(0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_requested_unsinkable_output_is_allowed(backend, sludge_recipes) -> None:
    finder = backend(sludge_recipes)
    finder.build_model({}, {"Sludge": Fraction(5)}, surplus=SurplusPolicy.FORBIDDEN)

    assert finder.solve() is True
    assert solved_value(finder, finder.user_given_outputs["Sludge"]) == pytest.approx(5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_nuclear_waste_surplus_may_always_leave(backend, sludge_recipes) -> None:
    finder = backend(sludge_recipes)
    finder.build_model({}, {"Gadget": Fraction(4)}, surplus=SurplusPolicy.FORBIDDEN)

    assert finder.solve() is True
    assert solved_value(
        finder, finder.user_given_outputs["Uranium Waste"]
    ) == pytest.approx(4)


def test_packaging_is_possible_only_when_surplus_is_packaged(sludge_recipes) -> None:
    finder = Z3OptimalChainFinder(sludge_recipes)
    names = {
        surplus: {
            recipe.Name
            for recipe in finder.construct_possibly_used_recipes({"Widget"}, surplus)
        }
        for surplus in SurplusPolicy
    }

    assert {"Packaged Sludge", "Canister"} <= names[SurplusPolicy.PACKAGED]
    assert "Packaged Sludge" not in names[SurplusPolicy.FORBIDDEN]
    assert "Packaged Sludge" not in names[SurplusPolicy.IGNORED]
