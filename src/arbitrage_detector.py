"""
Arbitrage Detector Module for CryptoLab

Implements the Bellman-Ford algorithm from scratch to detect negative-weight
cycles in directed weighted graphs and reconstruct profitable cryptocurrency
arbitrage cycles.

No third-party black-box cycle detection libraries are used; the relaxation
and cycle-extraction logic is implemented directly using Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class ArbitrageResult:
    """
    Structured outcome of the Bellman-Ford arbitrage detection algorithm.

    Attributes:
        has_arbitrage (bool): True if a valid negative-weight cycle was detected.
        cycle (Optional[List[str]]): Ordered list of currency symbols representing
            the cycle, starting and ending with the same currency (e.g. ['BTC', 'ETH', 'USDT', 'BTC']).
        log_weight_sum (Optional[float]): The sum of logarithmic edge weights along the cycle (< 0).
        rate_product (Optional[float]): The product of original exchange rates along the cycle (> 1).
        description (str): Human-readable summary of the detection result.
    """

    has_arbitrage: bool
    cycle: Optional[List[str]] = None
    log_weight_sum: Optional[float] = None
    rate_product: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a plain dictionary representation."""
        return {
            "has_arbitrage": self.has_arbitrage,
            "cycle": self.cycle,
            "log_weight_sum": self.log_weight_sum,
            "rate_product": self.rate_product,
            "description": self.description,
        }



def detect_arbitrage(graph, tolerance=1e-9):

    # Step 1: Check whether the graph has enough edges and currencies
    if graph is None or graph.number_of_nodes() < 2 or graph.number_of_edges() < 2:
        return ArbitrageResult(
            has_arbitrage=False,
            description="Graph has insufficient vertices or edges to form an arbitrage cycle."
        )

    vertices = list(graph.nodes())
    num_vertices = len(vertices)

    # Step 2: Initially, give every currency a distance of zero
    distance = {}
    predecessor = {}

    for currency in vertices:
        distance[currency] = 0.0
        predecessor[currency] = None

    # Step 3: Store all exchange edges and their weights
    edges = []

    for source, target, data in graph.edges(data=True):
        weight = data.get("weight")

        if weight is None:
            raise ValueError(f"Missing weight on edge {source} -> {target}")

        edges.append((source, target, float(weight)))

    # Step 4: Apply Bellman-Ford relaxation |V| - 1 times
    for _ in range(num_vertices - 1):

        for source, target, weight in edges:

            new_distance = distance[source] + weight

            if new_distance < distance[target] - tolerance:
                distance[target] = new_distance
                predecessor[target] = source

    # Step 5: Check whether any edge can still be improved
    # If it can, a negative-weight cycle may exist
    for source, target, weight in edges:

        new_distance = distance[source] + weight

        if new_distance < distance[target] - tolerance:

            # Update the predecessor so we can trace the cycle
            predecessor[target] = source

            # Step 6: Reconstruct the cycle using the existing helper
            cycle = _extract_cycle(
                predecessor,
                target,
                num_vertices
            )

            if cycle is None:
                continue

            # Step 7: Verify that the cycle is valid and profitable
            result = _validate_and_evaluate_cycle(
                graph,
                cycle,
                tolerance
            )

            if result is None:
                continue

            log_weight_sum, rate_product = result
            cycle_text = " -> ".join(cycle)

            # Step 8: Return the profitable arbitrage opportunity
            return ArbitrageResult(
                has_arbitrage=True,
                cycle=cycle,
                log_weight_sum=log_weight_sum,
                rate_product=rate_product,
                description=(
                    f"Profitable arbitrage detected: {cycle_text}. "
                    f"Rate product: {rate_product:.6f}. "
                    f"Log weight sum: {log_weight_sum:.6f}."
                )
            )

    # No valid profitable cycle was found
    return ArbitrageResult(
        has_arbitrage=False,
        description="No arbitrage opportunity detected."
    )



def _extract_cycle(predecessor: Dict[str, Optional[str]], start_vertex: str, num_vertices: int) -> Optional[List[str]]:
    """
    Reconstruct a directed cycle from the predecessor map.

    To ensure entry into the cycle, we first step backwards |V| times along
    predecessor pointers. Once inside the cycle, we record vertices until
    the cycle repeats.

    Args:
        predecessor: Mapping from vertex to its predecessor.
        start_vertex: The vertex where relaxation succeeded on the |V|-th pass.
        num_vertices: Total number of vertices in the graph.

    Returns:
        Optional[List[str]]: Ordered cycle starting and ending with the same vertex,
            e.g. ['BTC', 'ETH', 'USDT', 'BTC'], or None if extraction fails.
    """
    curr: Optional[str] = start_vertex

    # Step backwards |V| times to guarantee entering the negative cycle
    for _ in range(num_vertices):
        if curr is None or curr not in predecessor:
            return None
        curr = predecessor[curr]

    if curr is None:
        return None

    # Step backwards from curr to collect the cycle vertices
    cycle_start: str = curr
    cycle_backwards: List[str] = [cycle_start]
    step = predecessor.get(cycle_start)

    visited: Set[str] = {cycle_start}
    while step != cycle_start:
        if step is None or step not in predecessor:
            return None
        if step in visited:
            # A sub-cycle was encountered; trim to that sub-cycle
            cycle_backwards.append(step)
            break
        visited.add(step)
        cycle_backwards.append(step)
        step = predecessor.get(step)

    cycle_backwards.append(cycle_start)

    # Predecessor tracing goes backwards (target -> source); reverse to get forward directed order
    forward_cycle = list(reversed(cycle_backwards))

    # Standardize: Ensure first and last vertices are identical
    if forward_cycle[0] != forward_cycle[-1]:
        # If trimming occurred, align start and end
        first_node = forward_cycle[0]
        if first_node in forward_cycle[1:]:
            idx = forward_cycle[1:].index(first_node) + 1
            forward_cycle = forward_cycle[: idx + 1]
        else:
            return None

    return forward_cycle


def _validate_and_evaluate_cycle(
    graph: Any, cycle: List[str], tolerance: float = 1e-9
) -> Optional[Tuple[float, float]]:
    """
    Validate that every edge in the reconstructed cycle exists in the graph,
    and calculate the exact logarithmic weight sum and exchange rate product.

    Args:
        graph: Directed graph containing edge attributes 'rate' and 'weight'.
        cycle: List of vertex symbols forming a closed walk (e.g. ['BTC', 'ETH', 'BTC']).
        tolerance: Numerical threshold for profitability verification.

    Returns:
        Optional[Tuple[float, float]]: (log_weight_sum, rate_product) if valid
            and profitable, otherwise None.
    """
    if len(cycle) < 3 or cycle[0] != cycle[-1]:
        return None

    log_weight_sum = 0.0
    rate_product = 1.0

    for i in range(len(cycle) - 1):
        u = cycle[i]
        v = cycle[i + 1]

        # Verify that the directed edge actually exists in the graph
        if not graph.has_edge(u, v):
            return None

        edge_data = graph[u][v]
        rate = edge_data.get("rate")
        weight = edge_data.get("weight")

        if rate is None or weight is None:
            return None

        log_weight_sum += float(weight)
        rate_product *= float(rate)

    # A profitable cycle MUST satisfy both:
    # 1. Total log weight < 0
    # 2. Product of exchange rates > 1
    if log_weight_sum < -tolerance and rate_product > 1.0 + tolerance:
        return log_weight_sum, rate_product

    return None
