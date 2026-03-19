"""UGV path planning with static obstacles using Dijkstra's algorithm.

Assignment-oriented defaults:
- Battlefield map: 70 x 70 cells (1 cell ~= 1 km)
- Obstacle density levels: 0.10, 0.20, 0.30
- Obstacles are static and known a-priori
- Reports multiple Measures of Effectiveness (MOE)
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

FREE = 0
BLOCKED = 1
ORTHOGONAL_MOVES: Sequence[Tuple[int, int, float]] = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
]
DIAGONAL_MOVES: Sequence[Tuple[int, int, float]] = [
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
]
Cell = Tuple[int, int]


@dataclass
class SearchResult:
    found: bool
    path: List[Cell]
    cost_km: float
    nodes_expanded: int
    runtime_ms: float


@dataclass
class StaticRunSummary:
    density: float
    solved: bool
    generation_attempts: int
    waypoints: int
    cost_km: float
    nodes_expanded: int
    runtime_ms: float
    turns: int
    avg_clearance_cells: float
    detour_ratio: float
    path_trace: str


def parse_point(text: str) -> Cell:
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid point '{text}'. Use row,col format.")
    return int(parts[0].strip()), int(parts[1].strip())


def neighbors(cell: Cell, size: int, allow_diagonal: bool) -> Iterable[Tuple[Cell, float]]:
    moves = list(ORTHOGONAL_MOVES)
    if allow_diagonal:
        moves.extend(DIAGONAL_MOVES)

    row, col = cell
    for d_row, d_col, cost in moves:
        n_row, n_col = row + d_row, col + d_col
        if 0 <= n_row < size and 0 <= n_col < size:
            yield (n_row, n_col), cost


def reconstruct_path(previous: Dict[Cell, Cell], start: Cell, goal: Cell) -> List[Cell]:
    if goal not in previous and goal != start:
        return []
    path = [goal]
    cursor = goal
    while cursor != start:
        cursor = previous[cursor]
        path.append(cursor)
    path.reverse()
    return path


def dijkstra_grid(grid: np.ndarray, start: Cell, goal: Cell, allow_diagonal: bool) -> SearchResult:
    start_time = time.perf_counter()
    size = grid.shape[0]
    queue: List[Tuple[float, Cell]] = [(0.0, start)]
    distances: Dict[Cell, float] = {start: 0.0}
    previous: Dict[Cell, Cell] = {}
    visited: set[Cell] = set()
    nodes_expanded = 0

    while queue:
        current_cost, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        for nxt, edge_cost in neighbors(current, size, allow_diagonal):
            if grid[nxt] == BLOCKED or nxt in visited:
                continue
            new_cost = current_cost + edge_cost
            if new_cost < distances.get(nxt, float("inf")):
                distances[nxt] = new_cost
                previous[nxt] = current
                heapq.heappush(queue, (new_cost, nxt))

    runtime_ms = (time.perf_counter() - start_time) * 1000.0
    if goal not in distances:
        return SearchResult(
            found=False,
            path=[],
            cost_km=float("inf"),
            nodes_expanded=nodes_expanded,
            runtime_ms=runtime_ms,
        )

    path = reconstruct_path(previous, start, goal)
    return SearchResult(
        found=True,
        path=path,
        cost_km=distances[goal],
        nodes_expanded=nodes_expanded,
        runtime_ms=runtime_ms,
    )


def generate_grid(size: int, density: float, rng: random.Random, start: Cell, goal: Cell) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.int8)
    blocked_count = int(size * size * density)
    cells = [(row, col) for row in range(size) for col in range(size)]
    cells.remove(start)
    if goal in cells:
        cells.remove(goal)
    blocked_cells = rng.sample(cells, k=min(blocked_count, len(cells)))
    for row, col in blocked_cells:
        grid[row, col] = BLOCKED
    return grid


def generate_solvable_grid(
    size: int,
    density: float,
    start: Cell,
    goal: Cell,
    allow_diagonal: bool,
    seed: int,
    max_attempts: int = 250,
) -> Tuple[np.ndarray, SearchResult, int]:
    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        grid = generate_grid(size, density, rng, start, goal)
        result = dijkstra_grid(grid, start, goal, allow_diagonal)
        if result.found:
            return grid, result, attempt
    raise RuntimeError(
        f"Could not generate a solvable grid for density={density} after {max_attempts} attempts."
    )


def count_turns(path: Sequence[Cell]) -> int:
    if len(path) < 3:
        return 0
    turns = 0
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for idx in range(2, len(path)):
        new_dir = (path[idx][0] - path[idx - 1][0], path[idx][1] - path[idx - 1][1])
        if new_dir != prev_dir:
            turns += 1
        prev_dir = new_dir
    return turns


def nearest_obstacle_distance_map(grid: np.ndarray) -> np.ndarray:
    size = grid.shape[0]
    distance_map = np.full((size, size), fill_value=np.inf, dtype=float)
    q: deque[Cell] = deque()

    blocked_cells = np.argwhere(grid == BLOCKED)
    if blocked_cells.size == 0:
        return distance_map

    for row, col in blocked_cells:
        cell = (int(row), int(col))
        distance_map[cell] = 0.0
        q.append(cell)

    while q:
        row, col = q.popleft()
        current = distance_map[row, col]
        for nxt, _ in neighbors((row, col), size, allow_diagonal=False):
            n_row, n_col = nxt
            if distance_map[n_row, n_col] > current + 1.0:
                distance_map[n_row, n_col] = current + 1.0
                q.append((n_row, n_col))
    return distance_map


def average_clearance(path: Sequence[Cell], obstacle_distance_map: np.ndarray) -> float:
    if not path:
        return 0.0
    values = [float(obstacle_distance_map[cell]) for cell in path]
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return float("inf")
    return float(sum(finite_values) / len(finite_values))


def compact_path_trace(path: Sequence[Cell], max_points: int = 18) -> str:
    if not path:
        return "[]"
    if len(path) <= max_points:
        return str(list(path))
    head = list(path[: max_points // 2])
    tail = list(path[-(max_points // 2) :])
    return f"{head} ... {tail}"


def run_static_scenario(
    size: int,
    density: float,
    start: Cell,
    goal: Cell,
    allow_diagonal: bool,
    seed: int,
) -> StaticRunSummary:
    grid, search_result, attempts = generate_solvable_grid(
        size=size,
        density=density,
        start=start,
        goal=goal,
        allow_diagonal=allow_diagonal,
        seed=seed,
    )

    obstacle_map = nearest_obstacle_distance_map(grid)
    direct_distance = math.dist(start, goal)
    detour_ratio = search_result.cost_km / direct_distance if direct_distance else 1.0

    return StaticRunSummary(
        density=density,
        solved=search_result.found,
        generation_attempts=attempts,
        waypoints=len(search_result.path),
        cost_km=search_result.cost_km,
        nodes_expanded=search_result.nodes_expanded,
        runtime_ms=search_result.runtime_ms,
        turns=count_turns(search_result.path),
        avg_clearance_cells=average_clearance(search_result.path, obstacle_map),
        detour_ratio=detour_ratio,
        path_trace=compact_path_trace(search_result.path),
    )


def write_csv(path: Path, rows: Sequence[StaticRunSummary]) -> None:
    fieldnames = [
        "density",
        "solved",
        "generation_attempts",
        "waypoints",
        "cost_km",
        "nodes_expanded",
        "runtime_ms",
        "turns",
        "avg_clearance_cells",
        "detour_ratio",
        "path_trace",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "density": f"{row.density:.2f}",
                    "solved": row.solved,
                    "generation_attempts": row.generation_attempts,
                    "waypoints": row.waypoints,
                    "cost_km": f"{row.cost_km:.3f}",
                    "nodes_expanded": row.nodes_expanded,
                    "runtime_ms": f"{row.runtime_ms:.3f}",
                    "turns": row.turns,
                    "avg_clearance_cells": f"{row.avg_clearance_cells:.3f}",
                    "detour_ratio": f"{row.detour_ratio:.3f}",
                    "path_trace": row.path_trace,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static-obstacle UGV navigation with Dijkstra and MOE reporting."
    )
    parser.add_argument("--size", type=int, default=70, help="Grid size (default: 70)")
    parser.add_argument(
        "--densities",
        type=str,
        default="0.10,0.20,0.30",
        help="Comma-separated obstacle densities (default: 0.10,0.20,0.30)",
    )
    parser.add_argument("--start", type=str, default="0,0", help="Start cell row,col")
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Goal cell row,col (default: bottom-right)",
    )
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    parser.add_argument(
        "--no-diagonal",
        action="store_true",
        help="Disable diagonal motion and use 4-neighbor movement only.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("UGV_Static_Obstacles/static_moe_results.csv"),
        help="Where to write MOE results CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    size = args.size
    start = parse_point(args.start)
    goal = parse_point(args.goal) if args.goal else (size - 1, size - 1)
    densities = [float(chunk.strip()) for chunk in args.densities.split(",") if chunk.strip()]
    allow_diagonal = not args.no_diagonal

    for point, name in [(start, "start"), (goal, "goal")]:
        if not (0 <= point[0] < size and 0 <= point[1] < size):
            raise ValueError(f"{name} cell {point} is outside a {size}x{size} grid.")

    print("UGV Static-Obstacle Path Planning")
    print(f"Grid: {size} x {size} cells")
    print(f"Start: {start} | Goal: {goal}")
    print(f"Movement model: {'8-neighbor' if allow_diagonal else '4-neighbor'}")
    print(f"Densities: {', '.join(f'{d:.2f}' for d in densities)}")

    summaries: List[StaticRunSummary] = []
    for idx, density in enumerate(densities):
        run_seed = args.seed + idx
        summary = run_static_scenario(
            size=size,
            density=density,
            start=start,
            goal=goal,
            allow_diagonal=allow_diagonal,
            seed=run_seed,
        )
        summaries.append(summary)

        print(f"\nObstacle density: {density:.2f}")
        print(f"  Solved: {summary.solved} (map attempts: {summary.generation_attempts})")
        print(f"  Path cost: {summary.cost_km:.3f} km")
        print(f"  Waypoints: {summary.waypoints}")
        print(f"  Nodes expanded: {summary.nodes_expanded}")
        print(f"  Compute time: {summary.runtime_ms:.3f} ms")
        print(f"  Turns in path: {summary.turns}")
        print(f"  Avg clearance from obstacles: {summary.avg_clearance_cells:.3f} cells")
        print(f"  Detour ratio vs straight-line: {summary.detour_ratio:.3f}")
        print(f"  Trace: {summary.path_trace}")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv_out, summaries)
    print(f"\nMOE results saved to: {args.csv_out}")


if __name__ == "__main__":
    main()
