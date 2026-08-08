import json
import os
import random
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch

import main
from main import (
    MatchScheduler,
    MatchTask,
    load_all_results,
    make_match_id,
    run_single_match,
)
from initial_elo import DEFAULT_ELO, V1_ELO


def _models(*names):
    return [{"name": name, "api_base": "x", "api_key": "x", "model": name}
            for name in names]


def _result(name_a, name_b, winner, game_id="fox-in-the-forest", **extra):
    return {
        "player_names": [name_a, name_b],
        "winner": winner,
        "reason": "score",
        "game_id": game_id,
        **extra,
    }


def _make_task():
    return MatchTask(
        model_a={"name": "model-a", "api_base": "x", "api_key": "x",
                 "model": "a"},
        model_b={"name": "model-b", "api_base": "x", "api_key": "x",
                 "model": "b"},
        game_id="fox-in-the-forest",
        match_id="fox-in-the-forest_model-a_vs_model-b_test_0001",
        seed="0",
    )


class FakeGame:
    def __init__(self, result):
        self._result = result

    def create_runner(self, player1, player2, verbose, seed, results_dir=None):
        result = self._result

        class Runner:
            def run_game(self):
                return dict(result)

        return Runner()


class MatchIdTests(unittest.TestCase):
    def test_match_id_is_order_independent_and_contains_game_id(self):
        with patch("main.time.strftime", return_value="20260807-000000"):
            id_ab = make_match_id("model-b", "model-a", "fox-in-the-forest", 7)
            id_ba = make_match_id("model-a", "model-b", "fox-in-the-forest", 7)

        self.assertEqual(id_ab, id_ba)
        self.assertTrue(id_ab.startswith("fox-in-the-forest_model-a_vs_model-b_"))
        self.assertTrue(id_ab.endswith("_0007"))

    def test_different_games_have_different_match_ids(self):
        fox = make_match_id("a", "b", "fox-in-the-forest", 1)
        number = make_match_id("a", "b", "number-decomposition", 1)

        self.assertNotEqual(fox, number)


class SchedulerTests(unittest.TestCase):
    def test_ratings_seeded_from_v1_baseline(self):
        scheduler = MatchScheduler(
            _models("Claude-Fable-5", "Brand-New-Model"),
            ["fox-in-the-forest"], [], rng=random.Random(0),
        )

        self.assertAlmostEqual(scheduler.ratings["Claude-Fable-5"],
                               V1_ELO["Claude-Fable-5"])
        self.assertAlmostEqual(scheduler.ratings["Brand-New-Model"], DEFAULT_ELO)
        # The new model has no prior games, so it is more uncertain.
        self.assertGreater(scheduler.sigma("Brand-New-Model"),
                           scheduler.sigma("Claude-Fable-5"))

    def test_prior_results_update_live_ratings(self):
        scheduler = MatchScheduler(
            _models("a", "b"), ["fox-in-the-forest"],
            [_result("a", "b", winner=0)],
            rng=random.Random(0),
        )

        self.assertGreater(scheduler.ratings["a"], scheduler.ratings["b"])
        self.assertEqual(scheduler.games_played["a"], 1)
        self.assertEqual(scheduler.games_played["b"], 1)

    def test_sample_match_returns_valid_task(self):
        scheduler = MatchScheduler(
            _models("a", "b", "c"), ["fox-in-the-forest", "number-decomposition"],
            [], rng=random.Random(0),
        )

        task = scheduler.sample_match()

        self.assertIn(task.game_id,
                      ["fox-in-the-forest", "number-decomposition"])
        self.assertNotEqual(task.model_a["name"], task.model_b["name"])
        self.assertIn(task.model_a["name"], ["a", "b", "c"])
        self.assertIn(task.model_b["name"], ["a", "b", "c"])

    def test_uncertain_models_are_scheduled_more(self):
        # "veteran" has many recorded games; "rookie" has none.
        prior = [_result("veteran", "other", winner=0) for _ in range(30)]
        scheduler = MatchScheduler(
            _models("veteran", "other", "rookie"), ["fox-in-the-forest"],
            prior, rng=random.Random(1),
        )

        counts = Counter()
        for _ in range(300):
            task = scheduler.sample_match()
            counts[task.model_a["name"]] += 1
            counts[task.model_b["name"]] += 1
            scheduler.finish_match(task, None)

        self.assertGreater(counts["rookie"], counts["veteran"])

    def test_closer_elo_opponents_are_preferred(self):
        # Big rating gaps: top ~1900, mid ~1880, bottom ~1200. All get equal
        # pseudo-experience so pairing is driven by Elo distance only.
        scheduler = MatchScheduler(
            _models("top", "mid", "bottom"), ["fox-in-the-forest"], [],
            rng=random.Random(2),
        )
        scheduler.ratings.update({"top": 1900.0, "mid": 1880.0, "bottom": 1200.0})
        scheduler.prior_games.update({"top": 50, "mid": 50, "bottom": 50})

        pair_counts = Counter()
        for _ in range(300):
            task = scheduler.sample_match()
            pair = tuple(sorted([task.model_a["name"], task.model_b["name"]]))
            pair_counts[pair] += 1
            scheduler.finish_match(task, None)

        self.assertGreater(pair_counts[("mid", "top")],
                           pair_counts[("bottom", "top")])
        self.assertGreater(pair_counts[("mid", "top")],
                           pair_counts[("bottom", "mid")])

    def test_games_stay_balanced(self):
        scheduler = MatchScheduler(
            _models("a", "b"),
            ["fox-in-the-forest", "number-decomposition"], [],
            rng=random.Random(3),
        )

        game_counts = Counter()
        for _ in range(40):
            task = scheduler.sample_match()
            game_counts[task.game_id] += 1
            scheduler.finish_match(
                task, _result("a", "b", winner=0, game_id=task.game_id)
            )

        self.assertEqual(game_counts["fox-in-the-forest"], 20)
        self.assertEqual(game_counts["number-decomposition"], 20)

    def test_first_player_slot_is_balanced(self):
        scheduler = MatchScheduler(
            _models("a", "b"), ["fox-in-the-forest"], [],
            rng=random.Random(4),
        )

        first_counts = Counter()
        for _ in range(50):
            task = scheduler.sample_match()
            first_counts[task.model_a["name"]] += 1
            scheduler.finish_match(
                task, _result(task.model_a["name"], task.model_b["name"],
                              winner=0)
            )

        self.assertEqual(first_counts["a"], 25)
        self.assertEqual(first_counts["b"], 25)

    def test_removed_model_is_not_scheduled_but_its_games_still_count(self):
        prior = [_result("kept", "removed", winner=1)]
        scheduler = MatchScheduler(
            _models("kept", "other"), ["fox-in-the-forest"], prior,
            rng=random.Random(5),
        )

        self.assertLess(scheduler.ratings["kept"], V1_ELO.get("kept", DEFAULT_ELO))
        for _ in range(20):
            task = scheduler.sample_match()
            self.assertNotIn("removed",
                             [task.model_a["name"], task.model_b["name"]])
            scheduler.finish_match(task, None)

    def test_aborted_match_does_not_change_ratings(self):
        scheduler = MatchScheduler(
            _models("a", "b"), ["fox-in-the-forest"], [],
            rng=random.Random(6),
        )
        before = dict(scheduler.ratings)

        task = scheduler.sample_match()
        scheduler.finish_match(task, None)  # aborted (api_error) => no result

        self.assertEqual(dict(scheduler.ratings), before)
        self.assertEqual(sum(scheduler.in_flight_pairs.values()), 0)


class LoadResultsTests(unittest.TestCase):
    def test_results_are_ordered_by_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            newer = _result("a", "b", winner=0, timestamp=2000.0)
            older = _result("a", "b", winner=1, timestamp=1000.0)
            for name, result in [("newer.json", newer), ("older.json", older)]:
                with open(os.path.join(directory, name), "w",
                          encoding="utf-8") as result_file:
                    json.dump(result, result_file)

            loaded = load_all_results(directory)

        self.assertEqual([r["timestamp"] for r in loaded], [1000.0, 2000.0])


class RunSingleMatchTests(unittest.TestCase):
    def test_api_error_keeps_checkpoint_and_records_no_result(self):
        game = FakeGame({
            "winner": None,
            "reason": "api_error",
            "api_error_player": 1,
            "player_names": ["model-a", "model-b"],
            "output_tokens": [0, 0],
        })
        task = _make_task()
        with tempfile.TemporaryDirectory() as directory:
            with patch("main.get_game", return_value=game), \
                    patch("main.LLMPlayer"):
                result = run_single_match(task, directory, directory,
                                          directory, False)

            self.assertIsNone(result)
            result_path = os.path.join(directory, f"{task.match_id}.json")
            self.assertFalse(os.path.exists(result_path))
            # Checkpoint stays for resume after an infrastructure failure.
            checkpoint_path = os.path.join(directory, f"{task.match_id}.jsonl")
            self.assertTrue(os.path.exists(checkpoint_path))

    def test_forfeit_result_is_recorded_and_checkpoint_removed(self):
        game = FakeGame({
            "winner": 1,
            "reason": "forfeit",
            "player_names": ["model-a", "model-b"],
            "output_tokens": [0, 0],
        })
        task = _make_task()
        with tempfile.TemporaryDirectory() as directory:
            with patch("main.get_game", return_value=game), \
                    patch("main.LLMPlayer"):
                result = run_single_match(task, directory, directory,
                                          directory, False)

            self.assertIsNotNone(result)
            result_path = os.path.join(directory, f"{task.match_id}.json")
            with open(result_path, "r", encoding="utf-8") as result_file:
                recorded = json.load(result_file)
            checkpoint_path = os.path.join(directory, f"{task.match_id}.jsonl")
            self.assertFalse(os.path.exists(checkpoint_path))

        self.assertEqual(recorded["reason"], "forfeit")
        self.assertEqual(recorded["winner"], 1)
        self.assertEqual(recorded["match_id"], task.match_id)
        self.assertIn("timestamp", recorded)


class RunTournamentTests(unittest.TestCase):
    def test_keyboard_interrupt_stops_run_and_keeps_recorded_results(self):
        game = FakeGame({
            "winner": 0,
            "reason": "score",
            "player_names": ["a", "b"],
            "output_tokens": [0, 0],
        })
        scheduler = MatchScheduler(
            _models("a", "b"), ["fox-in-the-forest"], [],
            rng=random.Random(0),
        )
        # Simulate Ctrl+C when the 4th match would be scheduled.
        original_sample = scheduler.sample_match
        samples = {"count": 0}

        def limited_sample():
            if samples["count"] >= 3:
                raise KeyboardInterrupt
            samples["count"] += 1
            return original_sample()

        scheduler.sample_match = limited_sample

        with tempfile.TemporaryDirectory() as directory:
            with patch("main.get_game", return_value=game), \
                    patch("main.LLMPlayer"):
                completed = main.run_tournament(
                    scheduler, directory, directory, directory,
                    workers=1, verbose=False
                )

            recorded = [f for f in os.listdir(directory)
                        if f.endswith(".json")]

        self.assertEqual(completed, 3)
        self.assertEqual(len(recorded), 3)


class CheckpointResumeTests(unittest.TestCase):
    @staticmethod
    def _write_checkpoint(directory, match_id, meta_extra=None, actions=()):
        meta = {
            "type": "meta",
            "match_id": match_id,
            "game_id": "fox-in-the-forest",
            "model_a": "a",
            "model_b": "b",
            "seed": "123",
        }
        meta.update(meta_extra or {})
        path = os.path.join(directory, f"{match_id}.jsonl")
        with open(path, "w", encoding="utf-8") as checkpoint_file:
            json.dump(meta, checkpoint_file)
            checkpoint_file.write("\n")
            for action in actions:
                json.dump(action, checkpoint_file)
                checkpoint_file.write("\n")
        return path

    def test_healthy_checkpoint_is_resumed_with_replay_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint(
                directory, "match-1",
                actions=[{"player": 0, "tool": "play_card",
                          "arguments": {"card": "B3"}, "output_tokens": 10}],
            )

            tasks = main.load_resumable_tasks(
                directory, directory, _models("a", "b"), ["fox-in-the-forest"]
            )

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.match_id, "match-1")
        self.assertEqual(task.seed, "123")
        self.assertEqual(task.model_a["name"], "a")
        self.assertEqual(task.model_b["name"], "b")
        self.assertEqual(task.checkpoint.next_replay()["arguments"],
                         {"card": "B3"})

    def test_checkpoint_with_removed_model_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_checkpoint(
                directory, "match-1", meta_extra={"model_b": "removed-model"}
            )

            tasks = main.load_resumable_tasks(
                directory, directory, _models("a", "b"), ["fox-in-the-forest"]
            )

            self.assertEqual(tasks, [])
            self.assertFalse(os.path.exists(path))

    def test_checkpoint_of_finished_match_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_checkpoint(directory, "match-1")
            with open(os.path.join(directory, "match-1.json"), "w",
                      encoding="utf-8") as result_file:
                json.dump(_result("a", "b", winner=0), result_file)

            tasks = main.load_resumable_tasks(
                directory, directory, _models("a", "b"), ["fox-in-the-forest"]
            )

            self.assertEqual(tasks, [])
            self.assertFalse(os.path.exists(path))

    def test_corrupt_checkpoint_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "broken.jsonl")
            with open(path, "w", encoding="utf-8") as checkpoint_file:
                checkpoint_file.write("{not json\n")

            tasks = main.load_resumable_tasks(
                directory, directory, _models("a", "b"), ["fox-in-the-forest"]
            )

            self.assertEqual(tasks, [])
            self.assertFalse(os.path.exists(path))

    def test_resumed_match_finishes_and_removes_checkpoint(self):
        game = FakeGame({
            "winner": 0,
            "reason": "score",
            "player_names": ["a", "b"],
            "output_tokens": [0, 0],
        })
        with tempfile.TemporaryDirectory() as directory:
            self._write_checkpoint(directory, "match-1")
            tasks = main.load_resumable_tasks(
                directory, directory, _models("a", "b"), ["fox-in-the-forest"]
            )
            self.assertEqual(len(tasks), 1)

            with patch("main.get_game", return_value=game), \
                    patch("main.LLMPlayer"):
                result = run_single_match(tasks[0], directory, directory,
                                          directory, False)

            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(
                os.path.join(directory, "match-1.json")))
            self.assertFalse(os.path.exists(
                os.path.join(directory, "match-1.jsonl")))


if __name__ == "__main__":
    unittest.main()
