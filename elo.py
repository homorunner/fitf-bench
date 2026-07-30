"""Compute Elo ratings from tournament results.

Reads all result JSON files from the results directory and computes Elo ratings
for each model using the standard Elo algorithm.

Usage:
    python elo.py [--results-dir results] [--k 32] [--initial 1500] [--iterations 100]
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple


def load_results(results_dir: str,
                 game_id: str = "fox-in-the-forest") -> List[Dict]:
    results = []
    if not os.path.isdir(results_dir):
        return results
    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Validate required fields
            if (data.get("game_id") == game_id and "player_names" in data
                    and "winner" in data):
                results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def compute_elo(results: List[Dict], k: float = 32.0, initial: float = 1500.0,
                iterations: int = 100, seed: int = 42) -> Dict[str, float]:
    """Compute Elo ratings by replaying all games multiple times in shuffled order.

    To reduce sensitivity to game ordering, we shuffle and replay the full set of
    games `iterations` times and average the final ratings.

    Args:
        results: List of game result dicts with 'player_names' and 'winner'.
        k: K-factor (controls rating sensitivity per game).
        initial: Initial Elo rating for all models.
        iterations: Number of shuffle-and-replay passes for stability.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping model name to Elo rating.
    """
    # Collect all unique model names
    all_models = set()
    for r in results:
        for name in r.get("player_names", []):
            all_models.add(name)

    if not all_models:
        return {}

    rng = random.Random(seed)

    # Accumulate ratings across iterations
    rating_sums: Dict[str, float] = defaultdict(float)

    for _ in range(iterations):
        ratings: Dict[str, float] = {name: initial for name in all_models}
        shuffled = results.copy()
        rng.shuffle(shuffled)

        for r in shuffled:
            names = r.get("player_names", [])
            winner = r.get("winner")

            if len(names) != 2:
                continue

            name_a, name_b = names[0], names[1]
            ra, rb = ratings[name_a], ratings[name_b]

            # Expected scores
            ea = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
            eb = 1.0 - ea

            # Actual scores
            if winner == 0:
                sa, sb = 1.0, 0.0
            else:
                sa, sb = 0.0, 1.0

            # Update ratings
            ratings[name_a] = ra + k * (sa - ea)
            ratings[name_b] = rb + k * (sb - eb)

        for name in all_models:
            rating_sums[name] += ratings[name]

    final_ratings = {name: rating_sums[name] / iterations for name in all_models}
    return final_ratings


def compute_confidence_intervals(results: List[Dict], k: float = 32.0,
                                  initial: float = 1500.0,
                                  bootstrap_rounds: int = 1000,
                                  seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
    """Compute Elo ratings with 95% confidence intervals via bootstrapping.

    Returns:
        Dict mapping model name to (lower_95, median, upper_95).
    """
    rng = random.Random(seed)
    all_ratings: Dict[str, List[float]] = defaultdict(list)

    for _ in range(bootstrap_rounds):
        sample = [rng.choice(results) for _ in range(len(results))]
        ratings = compute_elo(sample, k=k, initial=initial, iterations=1,
                              seed=rng.randint(0, 2**31))
        for name, rating in ratings.items():
            all_ratings[name].append(rating)

    ci = {}
    for name, ratings_list in all_ratings.items():
        ratings_list.sort()
        n = len(ratings_list)
        lower = ratings_list[int(n * 0.025)]
        median = ratings_list[n // 2]
        upper = ratings_list[int(n * 0.975)]
        ci[name] = (lower, median, upper)

    return ci


def print_elo_table(ratings: Dict[str, float],
                    ci: Dict[str, Tuple[float, float, float]] = None,
                    game_counts: Dict[str, int] = None,
                    win_rates: Dict[str, float] = None):
    print("\n  ELO RATINGS:")

    if ci:
        print(f"  {'#':<4} {'Model':<25} {'Elo':>7} {'95% CI':>16} {'Games':>7} {'WinRate':>8}")
        print("-" * 72)
        for rank, (name, rating) in enumerate(
            sorted(ratings.items(), key=lambda x: x[1], reverse=True), 1
        ):
            lower, _, upper = ci.get(name, (0, 0, 0))
            games = game_counts.get(name, 0) if game_counts else 0
            wr = win_rates.get(name, 0) if win_rates else 0
            print(f"  {rank:<4} {name:<25} {rating:>7.1f} "
                  f"[{lower:>6.1f}, {upper:>6.1f}] {games:>7} {wr:>7.1f}%")
    else:
        print(f"  {'#':<4} {'Model':<25} {'Elo':>7} {'Games':>7} {'WinRate':>8}")
        print("-" * 72)
        for rank, (name, rating) in enumerate(
            sorted(ratings.items(), key=lambda x: x[1], reverse=True), 1
        ):
            games = game_counts.get(name, 0) if game_counts else 0
            wr = win_rates.get(name, 0) if win_rates else 0
            print(f"  {rank:<4} {name:<25} {rating:>7.1f} {games:>7} {wr:>7.1f}%")


def print_head_to_head(results: List[Dict], model_names: List[str]):
    """Print head-to-head win matrix."""
    h2h: Dict[str, Dict[str, List[int]]] = {
        a: {b: [0, 0] for b in model_names} for a in model_names
    }  # [wins, losses]

    for r in results:
        names = r.get("player_names", [])
        winner = r.get("winner")
        if len(names) != 2:
            continue
        a, b = names[0], names[1]
        if a not in h2h or b not in h2h:
            continue
        if winner == 0:
            h2h[a][b][0] += 1
            h2h[b][a][1] += 1
        elif winner == 1:
            h2h[a][b][1] += 1
            h2h[b][a][0] += 1

    # Determine column width based on model names
    col_width = max(10, max((len(n[:12]) for n in model_names), default=10) + 2)

    print("\n  Head-to-Head (W-L):")
    print(f"  {'':25}", end="")
    for name in model_names:
        print(f"{name[:col_width]:<{col_width}}", end="")
    print()

    for a in model_names:
        print(f"  {a:25}", end="")
        for b in model_names:
            if a == b:
                print(f"{'--':<{col_width}}", end="")
            else:
                w, l = h2h[a][b]
                cell = f"{w}-{l}"
                print(f"{cell:<{col_width}}", end="")
        print()
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute Elo ratings from tournament results")
    parser.add_argument("--game", default="fox-in-the-forest",
                        help="Game results to include (default: fox-in-the-forest)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Results directory (default: results)")
    parser.add_argument("--k", type=float, default=32.0,
                        help="Elo K-factor (default: 32)")
    parser.add_argument("--initial", type=float, default=1500.0,
                        help="Initial Elo rating (default: 1500)")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of shuffle-replay iterations (default: 200)")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Number of bootstrap rounds for confidence intervals (default: 1000)")
    parser.add_argument("--no-ci", action="store_true",
                        help="Skip confidence interval computation (faster)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    results = load_results(results_dir, args.game)

    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded {len(results)} game result(s) from {results_dir}")

    # Count games per model
    game_counts: Dict[str, int] = defaultdict(int)
    for r in results:
        for name in r.get("player_names", []):
            game_counts[name] += 1

    # Compute Elo ratings
    ratings = compute_elo(results, k=args.k, initial=args.initial,
                          iterations=args.iterations, seed=args.seed)

    # Compute confidence intervals
    ci = None
    if not args.no_ci and len(results) >= 10:
        print("Computing confidence intervals (bootstrapping)...")
        ci = compute_confidence_intervals(results, k=args.k, initial=args.initial,
                                           bootstrap_rounds=args.bootstrap, seed=args.seed)

    # Compute win rates per model
    model_wins: Dict[str, int] = defaultdict(int)
    for r in results:
        winner = r.get("winner")
        names = r.get("player_names", [])
        if winner is not None and winner < len(names):
            model_wins[names[winner]] += 1
    win_rates = {name: (model_wins[name] / game_counts[name] * 100) if game_counts[name] else 0
                 for name in game_counts}

    # Print results
    print_elo_table(ratings, ci=ci, game_counts=game_counts, win_rates=win_rates)
    model_names = sorted(ratings.keys(), key=lambda n: ratings[n], reverse=True)
    print_head_to_head(results, model_names)

    # Print token usage summary if available
    token_data: Dict[str, List[int]] = defaultdict(list)
    for r in results:
        names = r.get("player_names", [])
        tokens = r.get("output_tokens", [])
        for i, name in enumerate(names):
            if i < len(tokens):
                token_data[name].append(tokens[i])

    if token_data:
        print("  Token Usage (output tokens per game):")
        print(f"  {'Model':<25} {'Avg':>10} {'Min':>10} {'Max':>10} {'Total':>12}")
        print("-" * 72)
        for name in model_names:
            if name in token_data and token_data[name]:
                vals = token_data[name]
                avg = sum(vals) / len(vals)
                print(f"  {name:<25} {avg:>10.0f} {min(vals):>10} {max(vals):>10} {sum(vals):>12}")
        print()


if __name__ == "__main__":
    main()
