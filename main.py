"""Round-robin tournament for two-player LLM game benchmarks.

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

from fitf_bench import GAMES, LLMPlayer, get_game

GAMES_PER_PAIR = 5
MAX_WORKERS = 8
RESULTS_DIR = "results"
LOGS_DIR = "logs"

MODELS: List[Dict[str, str]] = [
    {
        "name": "DS-V4-Pro",
        "model": "deepseek-v4-pro",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
        "extra_api_params": {"extra_body": {"thinking": {"type": "enabled"}}},
    },
    {
        "name": "DS-V4-Flash-GA",
        "model": "deepseek-v4-flash",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
        "extra_api_params": {"extra_body": {"thinking": {"type": "enabled"}}, "max_tokens": 65536},
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
    {
        "name": "Claude-Fable-5",
        "model": "anthropic/claude-fable-5",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
    },
    {
       "name": "Gemini-3.5-Flash",
       "model": "google/gemini-3.5-flash",
       "api_base": os.environ.get("API_BASE"),
       "api_key": os.environ.get("API_KEY"),
    },
    {
        "name": "Kimi-K3",
        "model": "moonshotai/kimi-k3",
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY")
    },
]

# ===========================================================================
# MATCH ID AND RESUME LOGIC
# ===========================================================================

def make_match_id(model_a_name: str, model_b_name: str, game_index: int,
                  game_id: str) -> str:
    """Create a deterministic match ID from the two model names and game index.
    
    The match ID is independent of player order (alphabetically sorted names),
    so A-vs-B game 1 and B-vs-A game 1 produce the same ID.
    """
    names = sorted([model_a_name, model_b_name])
    return f"{game_id}_{names[0]}_vs_{names[1]}_game{game_index}"


def get_completed_matches(results_dir: str, game_ids) -> set:
    """Scan results directory and return set of completed match IDs."""
    completed = set()
    selected_games = set(game_ids)
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
            if data.get("game_id") in selected_games and match_id:
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
    game_id: str
    game_index: int
    match_id: str
    seed: str


def run_single_match(task: MatchTask, results_dir: str, logs_dir: str,
                     verbose: bool) -> Optional[Dict]:
    """Run a single match between two models. Returns result dict or None on error."""
    result_path = os.path.join(results_dir, f"{task.match_id}.json")
    log_path = os.path.join(logs_dir, f"{task.match_id}.jsonl")

    player1 = LLMPlayer(
        player_id=0,
        api_base=task.model_a["api_base"],
        api_key=task.model_a["api_key"],
        model=task.model_a["model"],
        model_name=task.model_a["name"],
        game_id=task.game_id,
        log_path=log_path,
        extra_api_params=task.model_a.get("extra_api_params"),
    )
    player2 = LLMPlayer(
        player_id=1,
        api_base=task.model_b["api_base"],
        api_key=task.model_b["api_key"],
        model=task.model_b["model"],
        model_name=task.model_b["name"],
        game_id=task.game_id,
        log_path=log_path,
        extra_api_params=task.model_b.get("extra_api_params"),
    )

    game = get_game(task.game_id)
    runner = game.create_runner(
        player1, player2, verbose=verbose, seed=task.seed
    )

    try:
        result = runner.run_game()
    except Exception as e:
        print(f"[ERROR] Match {task.match_id} failed with exception: {e}")
        return None

    if result.get("reason") == "api_error":
        # Infrastructure failure: nobody's fault, don't record, replay later.
        failed = result["player_names"][result["api_error_player"]]
        print(f"[ABORT] {task.match_id}: repeated API errors for {failed} "
              f"(not recorded, will be replayed)")
        return None

    # Enrich result with tournament metadata
    result["match_id"] = task.match_id
    result["seed"] = task.seed
    result["game_index"] = task.game_index
    result["model_a"] = task.model_a["name"]
    result["model_b"] = task.model_b["name"]
    result["game_id"] = task.game_id

    # Write result
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    winner_name = result["player_names"][result["winner"]]
    score_text = ""
    if len(result.get("scores", [])) == 2:
        score_text = f" ({result['scores'][0]}-{result['scores'][1]})"
    print(f"[DONE] {task.match_id}: {winner_name} wins{score_text} "
          f"({result['reason']})")

    return result


# ===========================================================================
# TOURNAMENT ORCHESTRATION
# ===========================================================================

def build_match_schedule(models: List[Dict[str, str]], games_per_pair: int,
                         completed: set, game_id: str) -> List[MatchTask]:
    """Build list of matches to run, skipping already-completed ones.
    
    For each pair (A, B), we play `games_per_pair` games. In even-indexed games
    (0, 2, 4, ...) model_a is player 1; in odd-indexed games (1, 3, 5, ...)
    model_b is player 1. This ensures each model gets roughly equal first-player
    opportunities (though actual first move is determined by the seed/RNG).
    """
    tasks = []
    for model_a, model_b in combinations(models, 2):
        for game_idx in range(games_per_pair):
            match_id = make_match_id(
                model_a["name"], model_b["name"], game_idx, game_id
            )
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
                game_id=game_id,
                game_index=game_idx,
                match_id=match_id,
                seed=seed,
            ))
    return tasks


def _load_all_results(results_dir: str, game_id: str) -> List[Dict]:
    all_results = []
    if os.path.isdir(results_dir):
        for filename in os.listdir(results_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    result = json.load(f)
                if result.get("game_id") == game_id:
                    all_results.append(result)
            except (json.JSONDecodeError, OSError):
                continue
    return all_results


def print_summary(results_dir: str, models: List[Dict[str, str]], game_id: str):
    all_results = _load_all_results(results_dir, game_id)
    if not all_results:
        print("\nNo results found.")
        return

    # Build stats per model
    model_names = [m["name"] for m in models]
    stats = {name: {"wins": 0, "losses": 0, "forfeits_won": 0,
                    "forfeits_lost": 0, "total_score": 0, "games": 0}
             for name in model_names}

    for r in all_results:
        names = r.get("player_names", [])
        winner = r.get("winner")
        reason = r.get("reason", "")
        scores = r.get("scores")

        for i, name in enumerate(names):
            if name not in stats:
                continue
            stats[name]["games"] += 1
            if scores and len(scores) == 2:
                stats[name]["total_score"] += scores[i]
            if winner == i:
                stats[name]["wins"] += 1
                if reason == "forfeit":
                    stats[name]["forfeits_won"] += 1
            else:
                stats[name]["losses"] += 1
                if reason == "forfeit":
                    stats[name]["forfeits_lost"] += 1

    # Print table
    print("\n" + "=" * 70)
    print(f"  TOURNAMENT SUMMARY: {GAMES[game_id].name}")
    print("=" * 70)
    print(f"  {'Model':<25} {'Games':>6} {'Wins':>6} {'Losses':>6} {'WinRate':>8} {'FfWins':>7} {'FfLosses':>9}")
    print("-" * 70)
    for name in sorted(model_names, key=lambda n: stats[n]["wins"], reverse=True):
        s = stats[name]
        win_rate = s["wins"] / s["games"] * 100 if s["games"] > 0 else 0
        print(f"  {name:<25} {s['games']:>6} {s['wins']:>6} {s['losses']:>6} {win_rate:>7.1f}% "
              f"{s['forfeits_won']:>7} {s['forfeits_lost']:>9}")
    print("=" * 70)




def main():
    parser = argparse.ArgumentParser(
        description="Two-Player LLM Game Tournament"
    )
    parser.add_argument("--game", choices=GAMES,
                        help="Game to run (default: all registered games)")
    parser.add_argument("--games", type=int, default=GAMES_PER_PAIR,
                        help=f"Number of games per model pair (default: {GAMES_PER_PAIR})")
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

    selected_games = [args.game] if args.game else list(GAMES)

    if args.summary_only:
        for game_id in selected_games:
            print_summary(results_dir, MODELS, game_id)
        return

    # Resume: find completed matches
    completed = get_completed_matches(results_dir, selected_games)
    if completed:
        print(f"[RESUME] Found {len(completed)} completed match(es), skipping them.")

    # Build schedule
    tasks = []
    for game_id in selected_games:
        tasks.extend(build_match_schedule(
            MODELS, args.games, completed, game_id
        ))
    total_possible = (len(list(combinations(MODELS, 2))) * args.games
                      * len(selected_games))
    print(f"[SCHEDULE] {len(tasks)} match(es) to run "
          f"({total_possible - len(tasks)} already completed, {total_possible} total)")

    if not tasks:
        print("[DONE] All matches already completed.")
        for game_id in selected_games:
            print_summary(results_dir, MODELS, game_id)
        return

    # Run matches
    results = []
    if args.workers <= 1:
        # Sequential execution
        for task in tqdm(tasks, desc="Matches", unit="game"):
            result = run_single_match(task, results_dir, logs_dir,
                                      args.verbose)
            if result:
                results.append(result)
    else:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {}
            for task in tasks:
                future = executor.submit(
                    run_single_match, task, results_dir, logs_dir,
                    args.verbose
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
    for game_id in selected_games:
        print_summary(results_dir, MODELS, game_id)


if __name__ == "__main__":
    main()
