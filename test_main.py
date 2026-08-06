import json
import os
import tempfile
import unittest
from unittest.mock import patch

import main
from main import (
    MatchTask,
    build_match_schedule,
    get_completed_matches,
    make_match_id,
    run_single_match,
)


def _make_task():
    return MatchTask(
        model_a={"name": "model-a", "api_base": "x", "api_key": "x",
                 "model": "a"},
        model_b={"name": "model-b", "api_base": "x", "api_key": "x",
                 "model": "b"},
        game_id="fox-in-the-forest",
        game_index=0,
        match_id="fox-in-the-forest_model-a_vs_model-b_game0",
        seed="0",
    )


class FakeGame:
    def __init__(self, result):
        self._result = result

    def create_runner(self, player1, player2, verbose, seed):
        result = self._result

        class Runner:
            def run_game(self):
                return dict(result)

        return Runner()


class MatchIdTests(unittest.TestCase):
    def test_match_id_always_contains_game_id(self):
        match_id = make_match_id("model-b", "model-a", 2,
                                 "fox-in-the-forest")

        self.assertEqual(
            match_id,
            "fox-in-the-forest_model-a_vs_model-b_game2",
        )

    def test_different_games_have_different_match_ids(self):
        fox = make_match_id("a", "b", 0, "fox-in-the-forest")
        number = make_match_id("a", "b", 0, "number-decomposition")

        self.assertNotEqual(fox, number)

    def test_completed_matches_require_matching_game_id(self):
        with tempfile.TemporaryDirectory() as directory:
            results = [
                {"game_id": "number-decomposition", "match_id": "number-1"},
                {"game_id": "fox-in-the-forest", "match_id": "fox-1"},
                {"match_id": "missing-game"},
            ]
            for index, result in enumerate(results):
                with open(os.path.join(directory, f"{index}.json"), "w",
                          encoding="utf-8") as result_file:
                    json.dump(result, result_file)

            completed = get_completed_matches(directory, ["number-decomposition"])

        self.assertEqual(completed, {"number-1"})

    def test_schedule_tasks_carry_their_game_id(self):
        models = [
            {"name": "a"},
            {"name": "b"},
        ]

        tasks = build_match_schedule(
            models, 1, set(), "number-decomposition"
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].game_id, "number-decomposition")
        self.assertTrue(tasks[0].match_id.startswith("number-decomposition_"))

    def test_main_schedules_all_games_when_game_is_omitted(self):
        captured_games = []

        def capture_schedule(models, games_per_pair, completed, game_id):
            captured_games.append(game_id)
            return []

        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "main.py", "--games", "1", "--results-dir", directory,
                "--logs-dir", directory,
            ]
            with patch("sys.argv", argv), \
                    patch("main.get_completed_matches", return_value=set()), \
                    patch("main.build_match_schedule", side_effect=capture_schedule), \
                    patch("main.print_summary"):
                main.main()

        self.assertEqual(captured_games, list(main.GAMES))


class RunSingleMatchTests(unittest.TestCase):
    def test_api_error_result_is_not_recorded(self):
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
                result = run_single_match(task, directory, directory, False)

            self.assertIsNone(result)
            result_path = os.path.join(directory, f"{task.match_id}.json")
            self.assertFalse(os.path.exists(result_path))

    def test_forfeit_result_is_recorded_as_a_win(self):
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
                result = run_single_match(task, directory, directory, False)

            self.assertIsNotNone(result)
            result_path = os.path.join(directory, f"{task.match_id}.json")
            with open(result_path, "r", encoding="utf-8") as result_file:
                recorded = json.load(result_file)

        self.assertEqual(recorded["reason"], "forfeit")
        self.assertEqual(recorded["winner"], 1)
        self.assertEqual(recorded["match_id"], task.match_id)


if __name__ == "__main__":
    unittest.main()
