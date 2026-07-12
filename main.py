"""Round-robin tournament for Fox in the Forest LLM benchmark.

Defines a set of LLM models and runs a full round-robin tournament where each
pair of models plays k games. Supports:
- Fixed random seeds for reproducibility (seed derived from matchup + game index)
- Resume from previous runs (skips already-completed matches found in results/)
- Parallel execution via concurrent.futures
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict, Optional

from tqdm import tqdm

from fitf_bench import GameRunner, LLMPlayer

GAMES_PER_PAIR = 5
TARGET_SCORE = 21
MAX_WORKERS = 8
RESULTS_DIR = "results"
LOGS_DIR = "logs"

MODELS: List[Dict[str, str]] = [
    {
        "name": "DS-V4-Pro-Thinking",
        "model": "deepseek-v4-pro",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
        "extra_api_params": {"extra_body": {"thinking": {"type": "enabled"}}},
    },
    {
        "name": "DS-V4-Pro-NonThinking",
        "model": "deepseek-v4-pro",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
        "extra_api_params": {"extra_body": {"thinking": {"type": "disabled"}}},
    },
    {
        "name": "DS-V4-Flash-Thinking",
        "model": "deepseek-v4-flash",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
        "extra_api_params": {"extra_body": {"thinking": {"type": "enabled"}}},
    },
    {
        "name": "DS-V4-Flash-NonThinking",
        "model": "deepseek-v4-flash",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
        "extra_api_params": {"extra_body": {"thinking": {"type": "disabled"}}},
    },
    {
        "name": "GLM-5.2-Max",
        "model": "zai/glm-5.2",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
    },
    {
        "name": "GPT-5.6-Terra",
        "model": "openai/gpt-5.6-terra",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
        "extra_api_params": {"extra_body": {"reasoning_effort": "high"}},
    },
    {
        "name": "GPT-5.6-Sol",
        "model": "openai/gpt-5.6-sol",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
        "extra_api_params": {"extra_body": {"reasoning_effort": "high"}},
    },
    # {
    #     "name": "Claude-Fable-5",
    #     "model": "anthropic/claude-fable-5",
    #     "api_base": os.environ.get("API_BASE"),
    #     "api_key": os.environ.get("API_KEY"),
    # },
    # {
    #     "name": "Claude-Sonnet-5",
    #     "model": "anthropic/claude-sonnet-5",
    #     "api_base": os.environ.get("API_BASE"),
    #     "api_key": os.environ.get("API_KEY"),
    # },
]

# ===========================================================================
# MATCH ID AND RESUME LOGIC
# ===========================================================================

def make_match_id(model_a_name: str, model_b_name: str, game_index: int) -> str:
    """Create a deterministic match ID from the two model names and game index.
    
    The match ID is independent of player order (alphabetically sorted names),
    so A-vs-B game 1 and B-vs-A game 1 produce the same ID.
    """
    names = sorted([model_a_name, model_b_name])
    return f"{names[0]}_vs_{names[1]}_game{game_index}"


def get_completed_matches(results_dir: str) -> set:
    """Scan results directory and return set of completed match IDs."""
    completed = set()
    if not os.path.isdir(results_dir):
        return completed
    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            match_id = data.get("match_id")
            if match_id:
                completed.add(match_id)
        except (json.JSONDecodeError, OSError):
            continue
    return completed


# ===========================================================================
# SINGLE MATCH EXECUTION
# ===========================================================================

@dataclass
class MatchTask:
    """Describes a single match to be played."""
    model_a: Dict[str, str]
    model_b: Dict[str, str]
    game_index: int
    match_id: str
    seed: int


def run_single_match(task: MatchTask, results_dir: str, logs_dir: str,
                     target_score: int, verbose: bool) -> Optional[Dict]:
    """Run a single match between two models. Returns result dict or None on error."""
    result_path = os.path.join(results_dir, f"{task.match_id}.json")
    log_path = os.path.join(logs_dir, f"{task.match_id}.jsonl")

    player1 = LLMPlayer(
        player_id=0,
        api_base=task.model_a["api_base"],
        api_key=task.model_a["api_key"],
        model=task.model_a["model"],
        player_name=task.model_a["name"],
        log_path=log_path,
        extra_api_params=task.model_a.get("extra_api_params"),
    )
    player2 = LLMPlayer(
        player_id=1,
        api_base=task.model_b["api_base"],
        api_key=task.model_b["api_key"],
        model=task.model_b["model"],
        player_name=task.model_b["name"],
        log_path=log_path,
        extra_api_params=task.model_b.get("extra_api_params"),
    )

    runner = GameRunner(
        player1, player2,
        target_score=target_score,
        verbose=verbose,
        seed=task.seed,
    )

    try:
        result = runner.run_game()
    except Exception as e:
        print(f"[ERROR] Match {task.match_id} failed with exception: {e}")
        return None

    if result.get("reason") == "forfeit":
        winner_name = result["player_names"][result["winner"]]
        print(f"[FORFEIT] {task.match_id}: {winner_name} by forfeit (not recorded)")
        return None

    # Enrich result with tournament metadata
    result["match_id"] = task.match_id
    result["seed"] = task.seed
    result["game_index"] = task.game_index
    result["model_a"] = task.model_a["name"]
    result["model_b"] = task.model_b["name"]

    # Write result
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    winner_name = result["player_names"][result["winner"]]
    print(f"[DONE] {task.match_id}: {winner_name} wins "
          f"({result['scores'][0]}-{result['scores'][1]}, {result['reason']})")

    return result


# ===========================================================================
# TOURNAMENT ORCHESTRATION
# ===========================================================================

def build_match_schedule(models: List[Dict[str, str]], games_per_pair: int,
                         completed: set) -> List[MatchTask]:
    """Build list of matches to run, skipping already-completed ones.
    
    For each pair (A, B), we play `games_per_pair` games. In even-indexed games
    (0, 2, 4, ...) model_a is player 1; in odd-indexed games (1, 3, 5, ...)
    model_b is player 1. This ensures each model gets roughly equal first-player
    opportunities (though actual first move is determined by the seed/RNG).
    """
    tasks = []
    for model_a, model_b in combinations(models, 2):
        for game_idx in range(games_per_pair):
            match_id = make_match_id(model_a["name"], model_b["name"], game_idx)
            if match_id in completed:
                continue
            seed = str(game_idx)
            # Alternate who is player 1 vs player 2
            if game_idx % 2 == 0:
                p1, p2 = model_a, model_b
            else:
                p1, p2 = model_b, model_a
            tasks.append(MatchTask(
                model_a=p1,
                model_b=p2,
                game_index=game_idx,
                match_id=match_id,
                seed=seed,
            ))
    return tasks


def print_summary(results_dir: str, models: List[Dict[str, str]]):
    """Print a summary table of all completed results."""
    # Collect all results
    all_results = []
    if os.path.isdir(results_dir):
        for filename in os.listdir(results_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    all_results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

    if not all_results:
        print("\nNo results found.")
        return

    # Build stats per model
    model_names = [m["name"] for m in models]
    stats = {name: {"wins": 0, "losses": 0, "forfeits_won": 0,
                    "total_score": 0, "games": 0}
             for name in model_names}

    for r in all_results:
        names = r.get("player_names", [])
        winner = r.get("winner")
        reason = r.get("reason", "")
        scores = r.get("scores", [0, 0])

        for i, name in enumerate(names):
            if name not in stats:
                continue
            stats[name]["games"] += 1
            stats[name]["total_score"] += scores[i]
            if winner == i:
                stats[name]["wins"] += 1
                if reason == "forfeit":
                    stats[name]["forfeits_won"] += 1
            else:
                stats[name]["losses"] += 1

    # Print table
    print("\n" + "=" * 70)
    print("  TOURNAMENT SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<25} {'Games':>6} {'Wins':>6} {'Losses':>6} {'WinRate':>8}")
    print("-" * 70)
    for name in sorted(model_names, key=lambda n: stats[n]["wins"], reverse=True):
        s = stats[name]
        win_rate = s["wins"] / s["games"] * 100 if s["games"] > 0 else 0
        print(f"  {name:<25} {s['games']:>6} {s['wins']:>6} {s['losses']:>6} {win_rate:>7.1f}%")
    print("=" * 70)

    # Head-to-head matrix
    print("\n  Head-to-Head (row wins vs column):")
    print(f"  {'':20}", end="")
    for name in model_names:
        print(f"{name[:8]:>10}", end="")
    print()

    h2h = {a: {b: 0 for b in model_names} for a in model_names}
    for r in all_results:
        names = r.get("player_names", [])
        winner = r.get("winner")
        if winner is not None and len(names) == 2:
            winner_name = names[winner]
            loser_name = names[1 - winner]
            if winner_name in h2h and loser_name in h2h[winner_name]:
                h2h[winner_name][loser_name] += 1

    for a in model_names:
        print(f"  {a:20}", end="")
        for b in model_names:
            if a == b:
                print(f"{'--':>10}", end="")
            else:
                print(f"{h2h[a][b]:>10}", end="")
        print()
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Fox in the Forest - Round-Robin LLM Tournament"
    )
    parser.add_argument("--games", type=int, default=GAMES_PER_PAIR,
                        help=f"Number of games per model pair (default: {GAMES_PER_PAIR})")
    parser.add_argument("--target-score", type=int, default=TARGET_SCORE,
                        help=f"Target score to win a game (default: {TARGET_SCORE})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Max parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help=f"Results directory (default: {RESULTS_DIR})")
    parser.add_argument("--logs-dir", type=str, default=LOGS_DIR,
                        help=f"Logs directory (default: {LOGS_DIR})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed game output")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print summary of existing results, don't run new games")

    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    logs_dir = os.path.abspath(args.logs_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    if args.summary_only:
        print_summary(results_dir, MODELS)
        return

    # Resume: find completed matches
    completed = get_completed_matches(results_dir)
    if completed:
        print(f"[RESUME] Found {len(completed)} completed match(es), skipping them.")

    # Build schedule
    tasks = build_match_schedule(MODELS, args.games, completed)
    total_possible = len(list(combinations(MODELS, 2))) * args.games
    print(f"[SCHEDULE] {len(tasks)} match(es) to run "
          f"({total_possible - len(tasks)} already completed, {total_possible} total)")

    if not tasks:
        print("[DONE] All matches already completed.")
        print_summary(results_dir, MODELS)
        return

    # Run matches
    results = []
    if args.workers <= 1:
        # Sequential execution
        for task in tqdm(tasks, desc="Matches", unit="game"):
            result = run_single_match(task, results_dir, logs_dir,
                                      args.target_score, args.verbose)
            if result:
                results.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {}
            for task in tasks:
                future = executor.submit(
                    run_single_match, task, results_dir, logs_dir,
                    args.target_score, args.verbose
                )
                future_to_task[future] = task

            with tqdm(total=len(tasks), desc="Matches", unit="game") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                            result = future.result()
                            if result:
                                results.append(result)
                                winner_name = result["player_names"][result["winner"]]
                                pbar.set_postfix_str(
                                    f"{task.match_id}: {winner_name}")
                    except Exception as e:
                        pbar.write(f"[ERROR] {task.match_id} raised: {e}")
                    pbar.update(1)

    print(f"\n[COMPLETE] {len(results)} match(es) finished this run.")
    print_summary(results_dir, MODELS)


if __name__ == "__main__":
    main()
