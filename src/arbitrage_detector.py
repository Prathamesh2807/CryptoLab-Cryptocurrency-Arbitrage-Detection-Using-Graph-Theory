from __future__ import annotations

"""
Bellman-Ford algorithm for CryptoLab.
Finds profitable cryptocurrency arbitrage cycles.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ArbitrageResult:
    has_arbitrage: bool
    cycle: list[str] | None = None
    log_weight_sum: float | None = None
    rate_product: float | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_arbitrage": self.has_arbitrage,
            "cycle": self.cycle,
            "log_weight_sum": self.log_weight_sum,
            "rate_product": self.rate_product,
            "description": self.description,
        }


def detect_arbitrage(graph: Any, tolerance: float = 1e-9) -> ArbitrageResult:
    """
    Detect whether a negative-weight cycle (arbitrage opportunity) exists
    anywhere in the directed graph using Bellman-Ford.
    """
    # Check whether the graph has enough edges and vertices
    if graph is None or graph.number_of_nodes() < 2 or graph.number_of_edges() < 2:
        return ArbitrageResult(
            False,
            description="Graph has insufficient vertices or edges.",
        )

    vertices = list(graph.nodes())
    num_vertices = len(vertices)

    # Store all edges as (source, destination, weight)
    edges = []

    for source, target, data in graph.edges(data=True):
        if "weight" not in data:
            raise ValueError(f"Edge ({source}, {target}) has no weight.")

        weight = float(data["weight"])

        if not math.isfinite(weight):
            raise ValueError(f"Edge ({source}, {target}) has an invalid weight.")

        edges.append((source, target, weight))

    # Set every distance to zero so every component can be checked
    distance = {currency: 0.0 for currency in vertices}

    # Remember the previous currency for each destination
    predecessor: dict[str, str | None] = {currency: None for currency in vertices}

    # Run Bellman-Ford for at most V passes
    last_updated = None

    for _ in range(num_vertices):
        last_updated = None

        # Check every edge
        for source, target, weight in edges:
            new_distance = distance[source] + weight

            # Update if the new route is cheaper
            if new_distance < distance[target] - tolerance:
                distance[target] = new_distance
                predecessor[target] = source
                last_updated = target

        # No changes means no negative cycle was found
        if last_updated is None:
            return ArbitrageResult(
                False,
                description="No arbitrage opportunity detected.",
            )

    # If a vertex was updated on pass V, investigate its cycle
    cycle = extract_cycle(predecessor, last_updated, num_vertices)

    if cycle is None:
        return ArbitrageResult(
            False,
            description="A negative cycle could not be reconstructed.",
        )

    # Check the cycle using the original exchange rates
    result = validate_cycle(graph, cycle, tolerance)

    if result is None:
        return ArbitrageResult(
            False,
            description="The detected cycle failed profitability validation.",
        )

    log_weight_sum, rate_product = result
    cycle_text = " -> ".join(cycle)

    return ArbitrageResult(
        has_arbitrage=True,
        cycle=cycle,
        log_weight_sum=log_weight_sum,
        rate_product=rate_product,
        description=(
            f"Profitable arbitrage cycle: {cycle_text}. "
            f"Rate product: {rate_product:.6f}. "
            f"Log weight sum: {log_weight_sum:.6f}."
        ),
    )


def extract_cycle(
    predecessor: dict[str, str | None], vertex: str, num_vertices: int
) -> list[str] | None:
    """
    Reconstruct a directed cycle from the predecessor map.
    """
    # Follow predecessors V times to enter the cycle
    current: str | None = vertex

    for _ in range(num_vertices):
        if current is None:
            return None
        current = predecessor.get(current)

    if current is None:
        return None

    # Collect the cycle by following predecessor links
    start = current
    cycle = [start]
    current = predecessor.get(start)

    while current is not None and current != start:
        # Avoid an endless loop if the links are unexpected
        if current in cycle:
            return None

        cycle.append(current)
        current = predecessor.get(current)

        if len(cycle) > num_vertices:
            return None

    if current is None:
        return None

    cycle.append(start)

    # Predecessors are followed backwards, so reverse the cycle
    cycle.reverse()

    return cycle


def validate_cycle(
    graph: Any, cycle: list[str], tolerance: float = 1e-9
) -> tuple[float, float] | None:
    """
    Validate that every edge in the cycle exists and calculate rate product and log weight sum.
    """
    if len(cycle) < 3 or cycle[0] != cycle[-1]:
        return None

    log_weight_sum = 0.0
    rate_product = 1.0

    # Check each directed edge in the cycle
    for i in range(len(cycle) - 1):
        source = cycle[i]
        target = cycle[i + 1]

        if not graph.has_edge(source, target):
            return None

        edge_data = graph[source][target]

        if "weight" not in edge_data or "rate" not in edge_data:
            return None

        weight = float(edge_data["weight"])
        rate = float(edge_data["rate"])

        if not math.isfinite(weight):
            return None

        if not math.isfinite(rate) or rate <= 0:
            return None

        log_weight_sum += weight
        rate_product *= rate

    # A profitable cycle has negative total log weight
    # and a product of exchange rates greater than one
    if log_weight_sum < -tolerance and rate_product > 1.0 + tolerance:
        return log_weight_sum, rate_product

    return None
