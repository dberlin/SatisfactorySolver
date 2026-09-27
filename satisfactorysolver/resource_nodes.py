"""
Resource node locations, extracted from the game by tools/extract_resource_nodes,
and the extraction limits they give within a distance of a point.
"""

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from importlib import resources

from satisfactorysolver.solver_helpers import ResourceLimits

# Most a node yields per minute at 250% clock speed: a Miner Mk.3, an Oil
# Extractor, or one Resource Well Extractor on a well satellite. These are the
# rates the map-wide ResourceLimits add up.
MINER_RATES = {"impure": 300, "normal": 600, "pure": 1200}
OIL_EXTRACTOR_RATES = {"impure": 150, "normal": 300, "pure": 600}
WELL_EXTRACTOR_RATES = {"impure": 75, "normal": 150, "pure": 300}


@dataclass(frozen=True)
class ResourceNode:
    resource: str
    purity: str
    well: bool
    # Meters, in the game's world coordinates.
    x: float
    y: float
    z: float

    @property
    def rate(self) -> int:
        if self.well:
            return WELL_EXTRACTOR_RATES[self.purity]
        if self.resource == "Crude Oil":
            return OIL_EXTRACTOR_RATES[self.purity]
        return MINER_RATES[self.purity]


def parse_point(text: str) -> tuple[float, float]:
    """Parse "X,Y" map coordinates in meters, raising ValueError."""
    try:
        x, y = (float(part) for part in str(text).split(","))
    except ValueError as error:
        raise ValueError('expected "X,Y" in meters') from error
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("coordinates must be finite")
    return x, y


def parse_radius(text) -> float:
    """Parse a positive distance in meters, raising ValueError."""
    try:
        radius = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError("radius must be a number of meters") from error
    if not (math.isfinite(radius) and radius > 0):
        raise ValueError("radius must be positive")
    return radius


def load_resource_nodes() -> list[ResourceNode]:
    """Load the bundled resource nodes."""
    text = (
        resources.files("satisfactorysolver")
        .joinpath("data", "resource_nodes.json")
        .read_text(encoding="utf-8")
    )
    return [ResourceNode(**node) for node in json.loads(text)]


def limits_near(
    nodes: list[ResourceNode], x: float, y: float, radius: float
) -> dict[str, Fraction]:
    """
    Extraction limits from the nodes within radius meters of (x, y), measured
    horizontally.

    Every limited resource gets an entry, zero when no node is in range. Water
    is left out: Water Extractors work on any body of water, so it stays
    unlimited.
    """
    limits = {
        resource: Fraction(0)
        for resource in ResourceLimits.get_resource_names()
        if resource != "Water"
    }
    for node in nodes:
        if node.resource in limits and math.hypot(node.x - x, node.y - y) <= radius:
            limits[node.resource] += node.rate
    return limits
