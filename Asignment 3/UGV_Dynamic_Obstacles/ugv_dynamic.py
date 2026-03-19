"""UGV path planning in dynamic, partially-known obstacle environments.

Key assignment behaviors:
- Obstacles are dynamic (they move over time).
- Obstacles are not known a-priori (UGV senses locally and replans online).
- UGV executes one step per cycle and continuously replans with Dijkstra.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

UNKNOWN = -1
FREE = 0
BLOCKED = 1
Cell = Tuple[int, int]

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


@dataclass
class SearchResult:
    found: bool
    path: List[Cell]
    cost_km: float
    nodes_expanded: int
    runtime_ms: float


@dataclass
class DynamicRunSummary:
    density: float
    success: bool
    reason: str
    steps_taken: int
    travel_km: float
    replans: int
    blocked_next_cell_events: int
    nodes_expanded_total: int
    planning_time_ms: float
    map_coverage_pct: float
    baseline_static_cost_km: float
    competitive_ratio: float
    trajectory_trace: str


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


def dijkstra_known_map(
    known_grid: np.ndarray,
    start: Cell,
    goal: Cell,
    allow_diagonal: bool,
    unknown_penalty: float,
) -> SearchResult:
    """Plan on known map; unknown cells are allowed with penalty."""
    start_time = time.perf_counter()
    size = known_grid.shape[0]
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
            cell_state = known_grid[nxt]
            if cell_state == BLOCKED or nxt in visited:
                continue
            penalty = unknown_penalty if cell_state == UNKNOWN else 1.0
            new_cost = current_cost + edge_cost * penalty
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


def dijkstra_true_map(grid: np.ndarray, start: Cell, goal: Cell, allow_diagonal: bool) -> SearchResult:
    known = np.array(grid, dtype=np.int8)
    return dijkstra_known_map(known, start, goal, allow_diagonal, unknown_penalty=1.0)


def generate_true_grid(size: int, density: float, rng: random.Random, start: Cell, goal: Cell) -> np.ndarray:
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


def sense_environment(
    true_grid: np.ndarray,
    known_grid: np.ndarray,
    current: Cell,
    goal: Cell,
    sensor_range: int,
) -> None:
    size = true_grid.shape[0]
    row, col = current
    for r in range(max(0, row - sensor_range), min(size, row + sensor_range + 1)):
        for c in range(max(0, col - sensor_range), min(size, col + sensor_range + 1)):
            if abs(r - row) + abs(c - col) <= sensor_range:
                known_grid[r, c] = int(true_grid[r, c])
    known_grid[current] = FREE
    known_grid[goal] = FREE


def move_dynamic_obstacles(
    true_grid: np.ndarray,
    rng: random.Random,
    move_prob: float,
    protected_cells: set[Cell],
) -> np.ndarray:
    """Move each obstacle with probability move_prob to a nearby free cell."""
    size = true_grid.shape[0]
    new_grid = np.array(true_grid, copy=True)
    obstacle_positions = [tuple(pos) for pos in np.argwhere(true_grid == BLOCKED)]
    rng.shuffle(obstacle_positions)
    occupied = set(obstacle_positions)

    for obs in obstacle_positions:
        if rng.random() >= move_prob:
            continue
        row, col = obs
        candidates = []
        for (n_row, n_col), _ in neighbors((row, col), size, allow_diagonal=False):
            nxt = (n_row, n_col)
            if nxt in protected_cells:
                continue
            if nxt in occupied:
                continue
            if true_grid[nxt] == FREE and new_grid[nxt] == FREE:
                candidates.append(nxt)
        if not candidates:
            continue

        target = rng.choice(candidates)
        new_grid[obs] = FREE
        new_grid[target] = BLOCKED
        occupied.remove(obs)
        occupied.add(target)

    return new_grid


def compact_trajectory(path: Sequence[Cell], max_points: int = 20) -> str:
    if not path:
        return "[]"
    if len(path) <= max_points:
        return str(list(path))
    head = list(path[: max_points // 2])
    tail = list(path[-(max_points // 2) :])
    return f"{head} ... {tail}"


def simulate_dynamic_navigation(
    size: int,
    density: float,
    start: Cell,
    goal: Cell,
    allow_diagonal: bool,
    sensor_range: int,
    move_prob: float,
    unknown_penalty: float,
    seed: int,
    max_steps: int,
) -> DynamicRunSummary:
    rng = random.Random(seed)
    true_grid = generate_true_grid(size, density, rng, start, goal)
    initial_grid = np.array(true_grid, copy=True)

    known_grid = np.full((size, size), fill_value=UNKNOWN, dtype=np.int8)
    known_grid[start] = FREE
    known_grid[goal] = FREE

    baseline = dijkstra_true_map(initial_grid, start, goal, allow_diagonal=allow_diagonal)
    baseline_cost = baseline.cost_km if baseline.found else float("inf")

    current = start
    trajectory: List[Cell] = [start]
    travel_km = 0.0
    replans = 0
    blocked_next_cell_events = 0
    planning_time_ms = 0.0
    nodes_expanded_total = 0
    reason = "max_steps_exceeded"
    success = False

    for _ in range(max_steps):
        sense_environment(true_grid, known_grid, current, goal, sensor_range=sensor_range)
        plan = dijkstra_known_map(
            known_grid=known_grid,
            start=current,
            goal=goal,
            allow_diagonal=allow_diagonal,
            unknown_penalty=unknown_penalty,
        )
        replans += 1
        planning_time_ms += plan.runtime_ms
        nodes_expanded_total += plan.nodes_expanded

        if not plan.found or len(plan.path) < 2:
            reason = "no_feasible_plan_with_current_knowledge"
            break

        next_cell = plan.path[1]
        if true_grid[next_cell] == BLOCKED:
            known_grid[next_cell] = BLOCKED
            blocked_next_cell_events += 1
            true_grid = move_dynamic_obstacles(
                true_grid,
                rng,
                move_prob=move_prob,
                protected_cells={current, goal},
            )
            continue

        prev = current
        current = next_cell
        trajectory.append(current)
        travel_km += math.dist(prev, current)

        if current == goal:
            success = True
            reason = "goal_reached"
            break

        true_grid = move_dynamic_obstacles(
            true_grid,
            rng,
            move_prob=move_prob,
            protected_cells={current, goal},
        )
    else:
        reason = "max_steps_exceeded"

    observed_cells = int(np.count_nonzero(known_grid != UNKNOWN))
    coverage_pct = (observed_cells / (size * size)) * 100.0

    if success and math.isfinite(baseline_cost) and baseline_cost > 0:
        competitive_ratio = travel_km / baseline_cost
    else:
        competitive_ratio = float("inf")

    return DynamicRunSummary(
        density=density,
        success=success,
        reason=reason,
        steps_taken=max(0, len(trajectory) - 1),
        travel_km=travel_km,
        replans=replans,
        blocked_next_cell_events=blocked_next_cell_events,
        nodes_expanded_total=nodes_expanded_total,
        planning_time_ms=planning_time_ms,
        map_coverage_pct=coverage_pct,
        baseline_static_cost_km=baseline_cost,
        competitive_ratio=competitive_ratio,
        trajectory_trace=compact_trajectory(trajectory),
    )


def write_csv(path: Path, rows: Sequence[DynamicRunSummary]) -> None:
    fieldnames = [
        "density",
        "success",
        "reason",
        "steps_taken",
        "travel_km",
        "replans",
        "blocked_next_cell_events",
        "nodes_expanded_total",
        "planning_time_ms",
        "map_coverage_pct",
        "baseline_static_cost_km",
        "competitive_ratio",
        "trajectory_trace",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "density": f"{row.density:.2f}",
                    "success": row.success,
                    "reason": row.reason,
                    "steps_taken": row.steps_taken,
                    "travel_km": f"{row.travel_km:.3f}",
                    "replans": row.replans,
                    "blocked_next_cell_events": row.blocked_next_cell_events,
                    "nodes_expanded_total": row.nodes_expanded_total,
                    "planning_time_ms": f"{row.planning_time_ms:.3f}",
                    "map_coverage_pct": f"{row.map_coverage_pct:.3f}",
                    "baseline_static_cost_km": (
                        f"{row.baseline_static_cost_km:.3f}"
                        if math.isfinite(row.baseline_static_cost_km)
                        else "inf"
                    ),
                    "competitive_ratio": (
                        f"{row.competitive_ratio:.3f}"
                        if math.isfinite(row.competitive_ratio)
                        else "inf"
                    ),
                    "trajectory_trace": row.trajectory_trace,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dynamic-obstacle UGV navigation with local sensing and online replanning."
    )
    parser.add_argument("--size", type=int, default=70, help="Grid size (default: 70)")
    parser.add_argument(
        "--densities",
        type=str,
        default="0.10,0.20,0.30",
        help="Comma-separated obstacle densities (default: 0.10,0.20,0.30)",
    )
    parser.add_argument("--start", type=str, default="0,0", help="Start cell row,col")
    parser.add_argument("--goal", type=str, default=None, help="Goal cell row,col")
    parser.add_argument("--sensor-range", type=int, default=4, help="Local sensing range")
    parser.add_argument("--move-prob", type=float, default=0.20, help="Obstacle move probability")
    parser.add_argument(
        "--unknown-penalty",
        type=float,
        default=1.15,
        help="Cost multiplier when planning through unknown cells.",
    )
    parser.add_argument("--seed", type=int, default=23, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=2800, help="Step budget for simulation")
    parser.add_argument(
        "--no-diagonal",
        action="store_true",
        help="Disable diagonal movement for the UGV.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("UGV_Dynamic_Obstacles/dynamic_moe_results.csv"),
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

    print("UGV Dynamic-Obstacle Navigation")
    print(f"Grid: {size} x {size} cells")
    print(f"Start: {start} | Goal: {goal}")
    print(f"Sensor range: {args.sensor_range} cells")
    print(f"Obstacle move probability: {args.move_prob:.2f}")
    print(f"Movement model: {'8-neighbor' if allow_diagonal else '4-neighbor'}")
    print(f"Densities: {', '.join(f'{d:.2f}' for d in densities)}")

    summaries: List[DynamicRunSummary] = []
    for idx, density in enumerate(densities):
        run_seed = args.seed + idx
        summary = simulate_dynamic_navigation(
            size=size,
            density=density,
            start=start,
            goal=goal,
            allow_diagonal=allow_diagonal,
            sensor_range=args.sensor_range,
            move_prob=args.move_prob,
            unknown_penalty=args.unknown_penalty,
            seed=run_seed,
            max_steps=args.max_steps,
        )
        summaries.append(summary)

        print(f"\nObstacle density: {density:.2f}")
        print(f"  Success: {summary.success} ({summary.reason})")
        print(f"  Steps taken: {summary.steps_taken}")
        print(f"  Travel distance: {summary.travel_km:.3f} km")
        print(f"  Replans: {summary.replans}")
        print(f"  Blocked-next-cell events: {summary.blocked_next_cell_events}")
        print(f"  Nodes expanded (total): {summary.nodes_expanded_total}")
        print(f"  Planning time (total): {summary.planning_time_ms:.3f} ms")
        print(f"  Known-map coverage: {summary.map_coverage_pct:.3f}%")
        baseline_text = (
            f"{summary.baseline_static_cost_km:.3f} km"
            if math.isfinite(summary.baseline_static_cost_km)
            else "unreachable"
        )
        ratio_text = (
            f"{summary.competitive_ratio:.3f}"
            if math.isfinite(summary.competitive_ratio)
            else "inf"
        )
        print(f"  Baseline static shortest-path: {baseline_text}")
        print(f"  Competitive ratio (travel / baseline): {ratio_text}")
        print(f"  Trajectory: {summary.trajectory_trace}")

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.csv_out, summaries)
    print(f"\nMOE results saved to: {args.csv_out}")


if __name__ == "__main__":
    main()
