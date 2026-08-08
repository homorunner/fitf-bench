"""Continuous Elo-driven matchmaking for two-player LLM game benchmarks (v2).

Matches are sampled continuously:
- Models with higher rating uncertainty (fewer games played) are picked more
  often, so new models converge quickly.
- Opponents are chosen with probability decaying in Elo distance, so matches
  are mostly played between models of similar strength (more informative).
- A fixed parallelism (--workers) is kept saturated; the run continues until
  the process is interrupted (Ctrl+C).

Ratings are seeded from the archived v1 standings (initial_elo.V1_ELO); models
without a v1 baseline start at the default rating with high uncertainty, which
automatically prioritizes them. Removing a model from MODELS simply stops
scheduling it; its recorded games still influence other models' ratings.

Every finished match is written to results/ immediately, so interrupting and
restarting is always safe: prior results are replayed on startup to restore the
live ratings.

In-flight matches are checkpointed to checkpoints/ after every successful
action. On restart, unfinished matches are resumed.
Checkpoints of finished matches are deleted; checkpoints referencing models no
longer in MODELS are discarded.
"""

import argparse
import json
import math
import os
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Dict, List, Optional

from fitf_bench import GAMES, LLMPlayer, get_game
from fitf_bench.checkpoint import MatchCheckpoint
from initial_elo import DEFAULT_ELO, V1_ELO

MAX_WORKERS = 8
RESULTS_DIR = "results"
LOGS_DIR = "logs"
CHECKPOINTS_DIR = "checkpoints"

# Online Elo K-factor used for the scheduler's live ratings.
ELO_K = 32.0
# Rating uncertainty (sigma) of a brand-new model.
SIGMA_INIT = 350.0
SIGMA_MIN = 60.0
# Pseudo-games credited to models that carry a v1 baseline rating.
PRIOR_GAMES_V1 = 10
# Elo-distance scale for opponent selection.
PAIRING_SCALE = 200.0
# Print live standings every N completed matches.
STANDINGS_EVERY = 10

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
# SINGLE MATCH EXECUTION
# ===========================================================================

@dataclass
class MatchTask:
    """Describes a single match to be played (or resumed)."""
    model_a: Dict[str, str]
    model_b: Dict[str, str]
    game_id: str
    match_id: str
    seed: str
    # Checkpoint with recorded actions to replay.
    checkpoint: Optional[MatchCheckpoint] = None


def make_match_id(model_a_name: str, model_b_name: str, game_id: str,
                  sequence: int) -> str:
    names = sorted([model_a_name, model_b_name])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{game_id}_{names[0]}_vs_{names[1]}_{stamp}_{sequence:04d}"


def run_single_match(task: MatchTask, results_dir: str, logs_dir: str,
                     checkpoints_dir: str, verbose: bool) -> Optional[Dict]:
    """Run (or resume) a single match. Returns result dict or None on error."""
    result_path = os.path.join(results_dir, f"{task.match_id}.json")
    log_path = os.path.join(logs_dir, f"{task.match_id}.jsonl")
    checkpoint_path = os.path.join(checkpoints_dir, f"{task.match_id}.jsonl")

    checkpoint = task.checkpoint
    if checkpoint is None:
        checkpoint = MatchCheckpoint.create(checkpoint_path, {
            "match_id": task.match_id,
            "game_id": task.game_id,
            "model_a": task.model_a["name"],
            "model_b": task.model_b["name"],
            "seed": task.seed,
        })
    else:
        print(f"[RESUME] {task.match_id}: replaying recorded actions")

    player1 = LLMPlayer(
        player_id=0,
        api_base=task.model_a["api_base"],
        api_key=task.model_a["api_key"],
        model=task.model_a["model"],
        model_name=task.model_a["name"],
        game_id=task.game_id,
        log_path=log_path,
        extra_api_params=task.model_a.get("extra_api_params"),
        checkpoint=checkpoint,
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
        checkpoint=checkpoint,
    )

    game = get_game(task.game_id)
    runner = game.create_runner(
        player1, player2, verbose=verbose, seed=task.seed,
        results_dir=results_dir,
    )

    try:
        result = runner.run_game()
    except Exception as e:
        print(f"[ERROR] Match {task.match_id} failed with exception: {e}")
        return None

    if result.get("reason") == "api_error":
        # Infrastructure failure: nobody's fault, don't record the result.
        # Keep the checkpoint so the match resumes on the next run.
        failed = result["player_names"][result["api_error_player"]]
        print(f"[ABORT] {task.match_id}: repeated API errors for {failed} "
              f"(not recorded, checkpoint kept for resume)")
        return None

    # Enrich result with tournament metadata
    result["match_id"] = task.match_id
    result["seed"] = task.seed
    result["model_a"] = task.model_a["name"]
    result["model_b"] = task.model_b["name"]
    result["game_id"] = task.game_id
    result["timestamp"] = time.time()

    # Write result, then drop the checkpoint (match is durably recorded).
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    checkpoint.delete()

    winner_name = result["player_names"][result["winner"]]
    score_text = ""
    if len(result.get("scores", [])) == 2:
        score_text = f" ({result['scores'][0]}-{result['scores'][1]})"
    print(f"[DONE] {task.match_id}: {winner_name} wins{score_text} "
          f"({result['reason']})")

    return result


def load_resumable_tasks(checkpoints_dir: str, results_dir: str,
                         models: List[Dict[str, str]],
                         game_ids: List[str]) -> List[MatchTask]:
    """Scan checkpoints/ for healthy half-finished matches to resume."""
    if not os.path.isdir(checkpoints_dir):
        return []
    models_by_name = {m["name"]: m for m in models}
    tasks = []
    for filename in sorted(os.listdir(checkpoints_dir)):
        if not filename.endswith(".jsonl"):
            continue
        path = os.path.join(checkpoints_dir, filename)
        meta, checkpoint = MatchCheckpoint.load(path)
        match_id = (meta or {}).get("match_id")
        model_a = models_by_name.get((meta or {}).get("model_a"))
        model_b = models_by_name.get((meta or {}).get("model_b"))
        game_id = (meta or {}).get("game_id")
        usable = (match_id and model_a and model_b and game_id in game_ids
                  and (meta or {}).get("seed") is not None)
        already_done = match_id and os.path.exists(
            os.path.join(results_dir, f"{match_id}.json"))
        if not usable or already_done:
            checkpoint.delete()
            continue
        tasks.append(MatchTask(
            model_a=model_a,
            model_b=model_b,
            game_id=game_id,
            match_id=match_id,
            seed=meta["seed"],
            checkpoint=checkpoint,
        ))
    return tasks


# ===========================================================================
# ELO-DRIVEN MATCHMAKING
# ===========================================================================

def sigma_for(games: float) -> float:
    """Rating uncertainty as a function of (effective) games played."""
    return max(SIGMA_MIN, SIGMA_INIT / math.sqrt(1.0 + games))


def _weighted_choice(rng: random.Random, items: List, weights: List[float]):
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    point = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if cumulative >= point:
            return item
    return items[-1]


def load_all_results(results_dir: str) -> List[Dict]:
    entries = []
    if not os.path.isdir(results_dir):
        return []
    for filename in os.listdir(results_dir):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(results_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            order = data.get("timestamp") or os.path.getmtime(filepath)
            entries.append((order, data))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Skipping unreadable result {filepath}: {exc}")
            continue
    entries.sort(key=lambda pair: pair[0])
    return [data for _, data in entries]


class MatchScheduler:
    def __init__(self, models: List[Dict[str, str]], game_ids: List[str],
                 prior_results: List[Dict], rng: Optional[random.Random] = None):
        if len(models) < 2:
            raise ValueError("Need at least two models to schedule matches.")
        self.models = {m["name"]: m for m in models}
        self.game_ids = list(game_ids)
        self.rng = rng or random.Random()
        self._lock = threading.Lock()
        self._sequence = 0

        self.ratings: Dict[str, float] = defaultdict(lambda: DEFAULT_ELO)
        self.prior_games: Dict[str, int] = {}
        self.games_played: Counter = Counter()
        self.first_played: Counter = Counter()
        self.game_match_counts: Counter = Counter()
        self.in_flight_pairs: Counter = Counter()
        self.in_flight_games: Counter = Counter()

        for name in self.models:
            self.ratings[name] = V1_ELO.get(name, DEFAULT_ELO)
            self.prior_games[name] = PRIOR_GAMES_V1 if name in V1_ELO else 0

        for result in prior_results:
            self._apply_result(result)

    def effective_games(self, name: str) -> float:
        return self.prior_games.get(name, 0) + self.games_played[name]

    def sigma(self, name: str) -> float:
        return sigma_for(self.effective_games(name))

    def _apply_result(self, result: Dict):
        names = result.get("player_names", [])
        winner = result.get("winner")
        if len(names) != 2 or winner not in (0, 1):
            return
        name_a, name_b = names
        if name_a not in self.ratings:
            self.ratings[name_a] = V1_ELO.get(name_a, DEFAULT_ELO)
        if name_b not in self.ratings:
            self.ratings[name_b] = V1_ELO.get(name_b, DEFAULT_ELO)
        rating_a, rating_b = self.ratings[name_a], self.ratings[name_b]
        expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        score_a = 1.0 if winner == 0 else 0.0
        self.ratings[name_a] = rating_a + ELO_K * (score_a - expected_a)
        self.ratings[name_b] = rating_b + ELO_K * ((1.0 - score_a) - (1.0 - expected_a))
        self.games_played[name_a] += 1
        self.games_played[name_b] += 1
        self.first_played[name_a] += 1
        game_id = result.get("game_id")
        if game_id in self.game_ids:
            self.game_match_counts[game_id] += 1

    @staticmethod
    def _pair_key(name_a: str, name_b: str) -> tuple:
        return tuple(sorted((name_a, name_b)))

    def sample_match(self) -> MatchTask:
        with self._lock:
            names = list(self.models)

            # 1. Pick the model most in need of games: weight = variance.
            first_weights = [self.sigma(name) ** 2 for name in names]
            picked = _weighted_choice(self.rng, names, first_weights)

            # 2. Pick an opponent: closer Elo => higher weight; uncertain
            #    opponents preferred; pairs already in flight are penalized.
            opponents = [name for name in names if name != picked]
            opponent_weights = []
            for name in opponents:
                distance = self.ratings[picked] - self.ratings[name]
                proximity = math.exp(-0.5 * (distance / PAIRING_SCALE) ** 2)
                uncertainty = self.sigma(name)
                repeat_penalty = 1.0 / (1.0 + self.in_flight_pairs[self._pair_key(picked, name)])
                opponent_weights.append(proximity * uncertainty * repeat_penalty)
            opponent = _weighted_choice(self.rng, opponents, opponent_weights)

            # 3. Pick the least-played game to keep games balanced.
            game_id = min(
                self.game_ids,
                key=lambda g: (self.game_match_counts[g] + self.in_flight_games[g],
                               self.rng.random()),
            )

            # 4. Balance who plays first (player 1).
            if self.first_played[picked] <= self.first_played[opponent]:
                model_a, model_b = self.models[picked], self.models[opponent]
            else:
                model_a, model_b = self.models[opponent], self.models[picked]

            self._sequence += 1
            match_id = make_match_id(picked, opponent, game_id, self._sequence)
            seed = str(self.rng.randrange(2 ** 31))

            self.in_flight_pairs[self._pair_key(picked, opponent)] += 1
            self.in_flight_games[game_id] += 1

            return MatchTask(
                model_a=model_a,
                model_b=model_b,
                game_id=game_id,
                match_id=match_id,
                seed=seed,
            )

    def register_in_flight(self, task: MatchTask):
        """Count an externally created task (e.g. resumed from a checkpoint)
        toward the in-flight pair/game counters so matchmaking stays balanced
        and finish_match stays symmetric."""
        with self._lock:
            pair = self._pair_key(task.model_a["name"], task.model_b["name"])
            self.in_flight_pairs[pair] += 1
            self.in_flight_games[task.game_id] += 1

    def finish_match(self, task: MatchTask, result: Optional[Dict]):
        with self._lock:
            pair = self._pair_key(task.model_a["name"], task.model_b["name"])
            if self.in_flight_pairs[pair] > 0:
                self.in_flight_pairs[pair] -= 1
            if self.in_flight_games[task.game_id] > 0:
                self.in_flight_games[task.game_id] -= 1
            if result is not None:
                self._apply_result(result)

    def format_standings(self) -> str:
        with self._lock:
            lines = ["  Live Elo (scheduler estimate):"]
            ranked = sorted(self.models, key=lambda n: self.ratings[n], reverse=True)
            for rank, name in enumerate(ranked, 1):
                lines.append(
                    f"  {rank:<3} {name:<25} {self.ratings[name]:>7.1f} "
                    f"(sigma {self.sigma(name):>5.1f}, v2 games {self.games_played[name]})"
                )
        return "\n".join(lines)


def print_summary(results_dir: str, models: List[Dict[str, str]], game_id: str):
    all_results = [r for r in load_all_results(results_dir)
                   if r.get("game_id") == game_id]
    if not all_results:
        print(f"\nNo results found for {game_id}.")
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


def run_tournament(scheduler: MatchScheduler, results_dir: str, logs_dir: str,
                   checkpoints_dir: str, workers: int, verbose: bool,
                   resume_tasks: Optional[List[MatchTask]] = None) -> int:
    completed = 0
    executor = ThreadPoolExecutor(max_workers=workers)
    futures: Dict = {}
    pending_resume = list(resume_tasks or [])
    try:
        while True:
            while len(futures) < workers:
                if pending_resume:
                    task = pending_resume.pop(0)
                    scheduler.register_in_flight(task)
                else:
                    task = scheduler.sample_match()
                future = executor.submit(
                    run_single_match, task, results_dir, logs_dir,
                    checkpoints_dir, verbose
                )
                futures[future] = task
                print(f"[MATCH] {task.match_id} "
                      f"({task.model_a['name']} vs {task.model_b['name']})")

            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[ERROR] {task.match_id} raised: {e}")
                    result = None
                scheduler.finish_match(task, result)
                if result is not None:
                    completed += 1
                    if completed % STANDINGS_EVERY == 0:
                        print(f"\n[PROGRESS] {completed} match(es) recorded this run.")
                        print(scheduler.format_standings() + "\n")
    except KeyboardInterrupt:
        print(f"\n[INTERRUPT] Stopping: no new matches will be scheduled. "
              f"{len(futures)} match(es) still in flight are checkpointed "
              f"and will be resumed on the next run.")
        executor.shutdown(wait=False, cancel_futures=True)
    return completed


def main():
    parser = argparse.ArgumentParser(
        description="Two-Player LLM Game Tournament"
    )
    parser.add_argument("--game", choices=GAMES,
                        help="Game to run (default: all registered games)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel matches to keep in flight (default: {MAX_WORKERS})")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help=f"Results directory (default: {RESULTS_DIR})")
    parser.add_argument("--logs-dir", type=str, default=LOGS_DIR,
                        help=f"Logs directory (default: {LOGS_DIR})")
    parser.add_argument("--checkpoints-dir", type=str, default=CHECKPOINTS_DIR,
                        help=f"Checkpoints directory for resuming half-finished "
                             f"matches (default: {CHECKPOINTS_DIR})")
    parser.add_argument("--seed", type=int, default=1437,
                        help="Random seed for the matchmaking sampler")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed game output")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print summary of existing results, don't run new games")

    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    logs_dir = os.path.abspath(args.logs_dir)
    checkpoints_dir = os.path.abspath(args.checkpoints_dir)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    selected_games = [args.game] if args.game else list(GAMES)

    if args.summary_only:
        for game_id in selected_games:
            print_summary(results_dir, MODELS, game_id)
        return

    prior_results = load_all_results(results_dir)
    if prior_results:
        print(f"[RESUME] Replayed {len(prior_results)} prior result(s) "
              f"to restore live ratings.")

    resume_tasks = load_resumable_tasks(
        checkpoints_dir, results_dir, MODELS, selected_games
    )
    if resume_tasks:
        print(f"[RESUME] Found {len(resume_tasks)} half-finished match(es) "
              f"to resume from checkpoints.")

    scheduler = MatchScheduler(
        MODELS, selected_games, prior_results,
        rng=random.Random(args.seed),
    )
    print(scheduler.format_standings())
    print(f"\n[START] Continuous matchmaking with {args.workers} worker(s). "
          f"Press Ctrl+C to stop.\n")

    completed = run_tournament(
        scheduler, results_dir, logs_dir, checkpoints_dir,
        workers=args.workers, verbose=args.verbose,
        resume_tasks=resume_tasks,
    )

    print(f"\n[COMPLETE] {completed} match(es) finished this run.")
    print(scheduler.format_standings())
    for game_id in selected_games:
        print_summary(results_dir, MODELS, game_id)


if __name__ == "__main__":
    main()
