"""Solve production targets and display the resulting chain as Rich tables."""

import argparse
import logging
import webbrowser
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from satisfactorysolver.game_data import load_game_data


@dataclass
class ChainArguments(argparse.Namespace):
    inputs: list[tuple[str, Fraction]] = field(default_factory=list)
    outputs: list[tuple[str, Fraction]] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    html: Path | None = None
    serve: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    open: bool = False
    solver: str = "z3"
    data_dir: Path | None = None
    verbose: int = 0


def parse_rate(value: str) -> tuple[str, Fraction]:
    """Parse an item and an exact per-minute rate, including fractional rates."""
    item, separator, rate = value.rpartition("=")
    if not separator or not item.strip():
        raise argparse.ArgumentTypeError('expected "Item name=rate"')
    try:
        amount = Fraction(rate.strip())
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(
            "rate must be a finite number or fraction"
        ) from error
    return item.strip(), amount


def canonical_names(names) -> dict[str, str]:
    """Map case-folded names to their canonical spelling."""
    return {name.casefold(): name for name in names}


def validate_targets(
    input_entries: list[tuple[str, Fraction]],
    output_entries: list[tuple[str, Fraction]],
    item_names: set[str],
) -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    """Check parsed targets and return them as input and output rate maps.

    Item names are matched ignoring case and returned in their canonical spelling.

    :raises ValueError: On duplicate or unknown items or invalid rates.
    """
    canonical = canonical_names(item_names)
    inputs: dict[str, Fraction] = {}
    outputs: dict[str, Fraction] = {}
    for entries, target, kind in (
        (input_entries, inputs, "input"),
        (output_entries, outputs, "output"),
    ):
        for item, amount in entries:
            item = canonical.get(item.casefold(), item)
            if item in target:
                raise ValueError(f"duplicate {kind} item: {item}")
            if amount < 0 and not (kind == "output" and amount == -1):
                raise ValueError(
                    f"{kind} rates must be nonnegative"
                    + (" or -1 to maximize" if kind == "output" else "")
                )
            target[item] = amount
    unknown = (inputs.keys() | outputs.keys()) - item_names
    if unknown:
        raise ValueError("unknown item(s): " + ", ".join(sorted(unknown)))
    return inputs, outputs


def parse_targets(
    input_lines: list[str], output_lines: list[str], item_names: set[str]
) -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    """Parse and validate ITEM=RATE lines, raising ValueError on any problem."""
    try:
        input_entries = [parse_rate(line) for line in input_lines]
        output_entries = [parse_rate(line) for line in output_lines]
    except argparse.ArgumentTypeError as error:
        raise ValueError(str(error)) from error
    return validate_targets(input_entries, output_entries, item_names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="satisfactory-chain",
        description=(
            "Find a production chain using the fewest weighted raw resources. "
            "All rates are per minute; alternate recipes are included."
        ),
        epilog=(
            'Example: satisfactory-chain --output "Iron Plate=20". '
            "Outputs with rate -1 are maximized together before resource use is minimized."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="inputs",
        action="append",
        type=parse_rate,
        default=[],
        metavar="ITEM=RATE",
        help="exact externally supplied input; repeat for multiple items",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="outputs",
        action="append",
        type=parse_rate,
        default=[],
        metavar="ITEM=RATE",
        help="required output, or -1 to maximize; repeat for multiple items "
        "(required unless --serve)",
    )
    parser.add_argument(
        "-x",
        "--exclude-recipe",
        dest="excluded",
        action="append",
        default=[],
        metavar="RECIPE",
        help="do not use this recipe; repeat for multiple recipes",
    )
    parser.add_argument(
        "--html",
        type=Path,
        metavar="PATH",
        help="also write the chain to an interactive HTML visualization at PATH",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="serve a live visualization where recipes can be toggled and targets edited",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="address for --serve (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="port for --serve (default: 8000)"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the visualization in a browser (with --html or --serve)",
    )
    parser.add_argument(
        "-s",
        "--solver",
        choices=("z3", "pyomo"),
        default="z3",
        help="solver backend: Z3 or Pyomo/HiGHS (default: z3)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="directory containing game_data.json and additional_data.json (default: bundled data)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv, namespace=ChainArguments())
    errors = Console(stderr=True)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        handlers=[RichHandler(console=errors, rich_tracebacks=True)],
    )

    if not args.outputs and not args.serve:
        parser.error("at least one --output is required (unless --serve)")

    try:
        data = load_game_data(args.data_dir)
    except (OSError, ValueError) as error:
        errors.print(f"Cannot load game data: {error}", markup=False)
        return 1
    item_names = {part.Name for part in data.Parts}
    try:
        inputs, outputs = validate_targets(args.inputs, args.outputs, item_names)
    except ValueError as error:
        parser.error(str(error))
    recipe_names = canonical_names(recipe.Name for recipe in data.Recipes)
    excluded = {recipe_names.get(name.casefold(), name) for name in args.excluded}
    unknown_recipes = excluded - set(recipe_names.values())
    if unknown_recipes:
        parser.error("unknown recipe(s): " + ", ".join(sorted(unknown_recipes)))

    if args.solver == "pyomo":
        from satisfactorysolver.pyomo_optimal_chain_finder import (
            PyomoOptimalChainFinder as finder_class,
        )
    else:
        from satisfactorysolver.z3_optimal_chain_finder import (
            Z3OptimalChainFinder as finder_class,
        )

    if args.serve:
        from satisfactorysolver.chain_web_server import ChainSession, serve

        session = ChainSession(data.Recipes, item_names, finder_class, parse_targets)
        serve(session, inputs, outputs, excluded, args.host, args.port, args.open)
        return 0

    finder = finder_class(
        {recipe for recipe in data.Recipes if recipe.Name not in excluded}
    )
    finder.build_model(inputs, outputs)
    if not finder.solve():
        errors.print(
            "No optimal production chain found (infeasible, unbounded, or solver unknown)."
        )
        return 1
    finder.print_inputs_outputs()
    if args.html:
        from satisfactorysolver.web_visualizer import (
            capture_chain_solution,
            write_html,
        )

        title = ", ".join(
            item if rate == -1 else f"{rate} {item}" for item, rate in outputs.items()
        )
        write_html([capture_chain_solution(finder, "Chain")], args.html, title)
        errors.print(f"Wrote {args.html}", markup=False)
        if args.open:
            webbrowser.open(args.html.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
