"""Dijkstra shortest-path solver on an India road network.

This file supports two data modes:
1) Built-in India road graph (major state/UT capitals + large cities).
2) A user-provided CSV with open-source road distances.

CSV format (header can vary, but must map to these logical fields):
- city_a / source / from
- city_b / target / to
- distance_km / distance / weight
"""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

Graph = Dict[str, Dict[str, float]]

# Distances are in km and represent a compact India-wide road network suitable
# for shortest-path demonstrations. Replace via --roads-csv for a larger corpus.
BUILTIN_ROAD_EDGES_KM: List[Tuple[str, str, float]] = [
    ("Delhi", "Chandigarh", 244),
    ("Delhi", "Jaipur", 281),
    ("Delhi", "Agra", 233),
    ("Delhi", "Lucknow", 555),
    ("Delhi", "Dehradun", 248),
    ("Delhi", "Jammu", 588),
    ("Delhi", "Bhopal", 780),
    ("Chandigarh", "Dehradun", 168),
    ("Chandigarh", "Jammu", 340),
    ("Jammu", "Srinagar", 270),
    ("Jaipur", "Agra", 240),
    ("Jaipur", "Indore", 625),
    ("Jaipur", "Ahmedabad", 657),
    ("Agra", "Lucknow", 335),
    ("Lucknow", "Varanasi", 320),
    ("Lucknow", "Patna", 530),
    ("Lucknow", "Bhopal", 585),
    ("Lucknow", "Dehradun", 545),
    ("Varanasi", "Patna", 255),
    ("Varanasi", "Raipur", 525),
    ("Patna", "Ranchi", 330),
    ("Patna", "Kolkata", 583),
    ("Patna", "Guwahati", 1000),
    ("Ranchi", "Kolkata", 403),
    ("Ranchi", "Raipur", 610),
    ("Kolkata", "Bhubaneswar", 442),
    ("Kolkata", "Guwahati", 1030),
    ("Kolkata", "Gangtok", 670),
    ("Bhubaneswar", "Visakhapatnam", 447),
    ("Bhubaneswar", "Raipur", 530),
    ("Visakhapatnam", "Vijayawada", 350),
    ("Visakhapatnam", "Hyderabad", 620),
    ("Vijayawada", "Hyderabad", 275),
    ("Vijayawada", "Chennai", 451),
    ("Hyderabad", "Bengaluru", 569),
    ("Hyderabad", "Nagpur", 500),
    ("Hyderabad", "Mumbai", 710),
    ("Hyderabad", "Chennai", 627),
    ("Nagpur", "Bhopal", 350),
    ("Nagpur", "Raipur", 285),
    ("Nagpur", "Mumbai", 837),
    ("Bhopal", "Indore", 190),
    ("Indore", "Ahmedabad", 390),
    ("Ahmedabad", "Mumbai", 531),
    ("Mumbai", "Pune", 150),
    ("Mumbai", "Bengaluru", 984),
    ("Pune", "Panaji", 450),
    ("Pune", "Bengaluru", 840),
    ("Panaji", "Kochi", 730),
    ("Bengaluru", "Mysuru", 150),
    ("Bengaluru", "Chennai", 347),
    ("Bengaluru", "Coimbatore", 365),
    ("Bengaluru", "Kochi", 550),
    ("Chennai", "Madurai", 463),
    ("Chennai", "Kochi", 680),
    ("Coimbatore", "Kochi", 190),
    ("Coimbatore", "Madurai", 215),
    ("Kochi", "Thiruvananthapuram", 200),
    ("Madurai", "Thiruvananthapuram", 297),
    ("Guwahati", "Shillong", 100),
    ("Guwahati", "Itanagar", 330),
    ("Guwahati", "Imphal", 490),
    ("Guwahati", "Aizawl", 460),
    ("Guwahati", "Agartala", 550),
    ("Guwahati", "Kohima", 350),
    ("Guwahati", "Gangtok", 520),
]


def build_graph(edges: Iterable[Tuple[str, str, float]]) -> Graph:
    graph: Graph = {}
    for city_a, city_b, distance_km in edges:
        graph.setdefault(city_a, {})[city_b] = float(distance_km)
        graph.setdefault(city_b, {})[city_a] = float(distance_km)
    return graph


def load_graph_from_csv(csv_path: Path) -> Graph:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file must include a header row.")

        normalised = {name.strip().lower(): name for name in reader.fieldnames}

        city_a_key = (
            normalised.get("city_a")
            or normalised.get("source")
            or normalised.get("from")
        )
        city_b_key = (
            normalised.get("city_b")
            or normalised.get("target")
            or normalised.get("to")
        )
        distance_key = (
            normalised.get("distance_km")
            or normalised.get("distance")
            or normalised.get("weight")
        )

        if not (city_a_key and city_b_key and distance_key):
            raise ValueError(
                "CSV header must provide city_a/city_b/distance_km "
                "(or source/target/distance)."
            )

        edges: List[Tuple[str, str, float]] = []
        for row in reader:
            city_a = row[city_a_key].strip()
            city_b = row[city_b_key].strip()
            if not city_a or not city_b:
                continue
            distance_km = float(row[distance_key])
            edges.append((city_a, city_b, distance_km))

    if not edges:
        raise ValueError("CSV file did not contain any valid edges.")
    return build_graph(edges)


def dijkstra(graph: Graph, start: str) -> Tuple[Dict[str, float], Dict[str, str | None]]:
    if start not in graph:
        raise ValueError(f"Start city '{start}' not present in graph.")

    distances: Dict[str, float] = {node: float("inf") for node in graph}
    previous: Dict[str, str | None] = {node: None for node in graph}
    distances[start] = 0.0

    queue: List[Tuple[float, str]] = [(0.0, start)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue

        for neighbor, edge_cost in graph[current_node].items():
            candidate_distance = current_distance + edge_cost
            if candidate_distance < distances[neighbor]:
                distances[neighbor] = candidate_distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (candidate_distance, neighbor))

    return distances, previous


def shortest_path(graph: Graph, start: str, goal: str) -> Tuple[List[str], float]:
    if goal not in graph:
        raise ValueError(f"Goal city '{goal}' not present in graph.")

    distances, previous = dijkstra(graph, start)
    if distances[goal] == float("inf"):
        return [], float("inf")

    path: List[str] = []
    cursor: str | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = previous[cursor]
    path.reverse()
    return path, distances[goal]


def print_route_breakdown(graph: Graph, path: List[str]) -> None:
    for city_a, city_b in zip(path, path[1:]):
        segment_km = graph[city_a][city_b]
        print(f"  {city_a:>17} -> {city_b:<17} : {segment_km:7.1f} km")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Dijkstra shortest path on an India road network."
    )
    parser.add_argument(
        "--start",
        default="Delhi",
        help="Start city name (default: Delhi)",
    )
    parser.add_argument(
        "--goal",
        default="Thiruvananthapuram",
        help="Goal city name (default: Thiruvananthapuram)",
    )
    parser.add_argument(
        "--roads-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing open-source road edges "
            "(columns: city_a, city_b, distance_km)."
        ),
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show shortest-path cost from start to every city.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.roads_csv:
        graph = load_graph_from_csv(args.roads_csv)
        source_name = f"CSV: {args.roads_csv}"
    else:
        graph = build_graph(BUILTIN_ROAD_EDGES_KM)
        source_name = "Built-in India road network"

    print(f"Data source: {source_name}")
    print(f"Cities in graph: {len(graph)}")

    if args.show_all:
        distances, _ = dijkstra(graph, args.start)
        print(f"\nShortest distances from {args.start}:")
        for city in sorted(distances):
            value = distances[city]
            shown = "unreachable" if value == float("inf") else f"{value:.1f} km"
            print(f"  {city:<20} {shown}")
        return

    path, total_distance = shortest_path(graph, args.start, args.goal)
    if not path:
        print(f"No route found from {args.start} to {args.goal}.")
        return

    print(f"\nShortest path from {args.start} to {args.goal}:")
    print(" -> ".join(path))
    print(f"Total distance: {total_distance:.1f} km")
    print("\nSegment breakdown:")
    print_route_breakdown(graph, path)


if __name__ == "__main__":
    main()
