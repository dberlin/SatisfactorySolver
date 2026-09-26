<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![BSD][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">Satisfactory Model Solvers</h3>

  <div align="left">
This project is intended to give working, correct, usable Python based solvers for satisfactory factory models.
There are lots of random solvers and calculators out there, but they vary in quality, optimality, and usability.

This is meant to be used as a library in other projects (modelers, calculators, etc.) and provide fast, correct,
solving using a variety of models, solvers, and conditions.

An example that can solve files in satisfactory modeler format can be
seen in [main.py](main.py), and shows usage of the various models and solvers under various conditions.

It supports building models that get solved using SMT solvers (z3/CVC5), as well as models that get solved using
MIP/MINLP solvers (CBC, IPOPT, HIGHS, GLPK).

It makes incremental solving easy (so can be used in live modelers), and has other features that make it useful.

For the MILP/MINLP based models, we support both linear and non-linear formulations,
as well as incremental solving under addition/removal of nodes and edges.

The advantage of this model is that it can easily handle incremental solving under almost all conditions
without having to regenerate the entire model.

The downsides are that it is a bit inflexible - everything must be formulated as
either a linear or non-linear constraint (depending on solver choice),
debugging is trickier (but possible), and things like enumeration of
alternative solutions is harder (but possible)

For SMT based models, we support both z3 and CVC5 based models, and can work with and without
an optimizer (z3 provides one, cvc5 does not) while still finding optimal answers.

The advantages are that the models are very flexible and easy to debug.
It is trivial to enumerate all possible optimal solutions, test what happens if you add
various conditions, and conditions can be formulated in complex ways.
It also provides exact answers as fractions rather than floats.

It can support incremental solving under node addition, edge addition, and max
addition without regenerating the model.  Edge removal/Node removal require model regeneration
due to how SMT solvers work.  Except for truly huge models, model generation is not
likely to be a meaningful amount of time (IE >1 second)

Theoretically this approach is slower to solve, in practice, for the size models
here there is no meaningful difference (both SMT and MILP/MINLP can solve in milliseconds).
If you discover a case the SMT solver is not fast enough, happy to help.

</div>
</div>

## Build and run

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv build
uv run pytest
```

The build uses `uv_build` and produces a wheel and source distribution. The wheel
includes the game data, so installed commands do not depend on the working directory.

The existing factory-model example is still available:

```sh
uv run main.py "instant scrap 10.sfmd"
```

## Optimal production chains

```sh
# Produce 20 Iron Plates per minute, using Z3.
uv run satisfactory-chain --output "Iron Plate=20"

# Use Pyomo with its bundled HiGHS solver.
uv run satisfactory-chain --solver pyomo --output "Iron Plate=20"

# Supply an existing intermediate and request multiple outputs.
uv run satisfactory-chain --input "Iron Ingot=30" \
  --output "Iron Plate=20" --output "Iron Rod=10"

# Find the maximum possible output under the resource limits.
uv run satisfactory-chain --solver pyomo --output "Iron Plate=-1"
```

The command prints **Inputs**, **Outputs**, **Intermediates**, and **Num recipes**
as Rich tables, with decimal and solver-value columns. Z3 preserves exact fractions;
HiGHS uses floating-point values. Recipe multipliers refer to each recipe's catalog
rate, not rounded machine counts.

All quantities are rates per minute. Decimal and fractional rates such as `1/3` are
accepted. `ITEM=RATE` inputs are external supplies of up to that rate; any excess
is left unused, and the item may still be mined or produced beyond it. `ITEM<=RATE`
also caps the item's total use, supplied plus mined or produced, which is useful with
maximized outputs (for example `--input "Crude Oil<=600" --output "Plastic=-1"`).
Requested outputs are exact requirements; unrequested byproducts may also appear
in Outputs.
All available recipes, including alternates, participate.

For fixed targets, the objective minimizes resource throughput weighted by scarcity,
using the map-wide bounds in `ResourceLimits`. An output rate of `-1` instead
maximizes that output first, then minimizes weighted resource use without sacrificing
the maximum. With multiple `-1` targets, their total rate is maximized.
Infeasible or unbounded problems exit with an error rather than displaying a solution.

In the HTML and `--serve` visualizations, the **Belts/lifts** and **Pipes** tiers
split each recipe into groups whose every item flow fits on one belt or pipe of that
tier (for example 1000 Sulfur/min at Mk.5 becomes groups of 780 and 220), and route
items between groups so no connection carries more than one belt or pipe. Groups hold
whole machines at 100% where possible. **No split** keeps each recipe as one node.

The command is also available as
`uv run python -m satisfactorysolver.optimal_chain_displayer`.

## Game data

The bundled Steam snapshot supplied on 2026-09-16 is stored in
`satisfactorysolver/data/game_data.json`, alongside `additional_data.json`.
Both files are included in distributions. To refresh the snapshot, replace those
files with the corresponding Satisfactory Modeler files.

`satisfactorysolver.game_data.load_game_data()` loads the bundled data. Additional
data overrides matching records by `Name`, preserving fields omitted by the overlay
and replacing supplied lists rather than duplicating recipe parts or capacities.

To use a separate data directory without changing the bundled snapshot:

```sh
uv run satisfactory-chain --data-dir /path/to/modeler/game_data \
  --output "Iron Plate=20"
```

That directory must contain both `game_data.json` and `additional_data.json`.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/dberlin/SatisfactorySolver.svg?style=for-the-badge
[contributors-url]: https://github.com/dberlin/SatisfactorySolver/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dberlin/SatisfactorySolver.svg?style=for-the-badge
[forks-url]: https://github.com/dberlin/SatisfactorySolver/network/members
[stars-shield]: https://img.shields.io/github/stars/dberlin/SatisfactorySolver.svg?style=for-the-badge
[stars-url]: https://github.com/dberlin/SatisfactorySolver/stargazers
[issues-shield]: https://img.shields.io/github/issues/dberlin/SatisfactorySolver.svg?style=for-the-badge
[issues-url]: https://github.com/dberlin/SatisfactorySolver/issues
[license-shield]: https://img.shields.io/github/license/dberlin/SatisfactorySolver.svg?style=for-the-badge
[license-url]: https://github.com/dberlin/SatisfactorySolver/blob/master/LICENSE.txt
