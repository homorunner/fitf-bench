import json
import os
import tempfile
import unittest
from unittest.mock import patch

import main
from main import build_match_schedule, get_completed_matches, make_match_id


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


if __name__ == "__main__":
    unittest.main()
