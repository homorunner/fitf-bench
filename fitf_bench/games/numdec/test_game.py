import unittest

from fitf_bench.games.numdec.game import (
    NumberDecompositionRunner,
    RoundState,
    evaluate_attack,
)


class FakePlayer:
    def __init__(self, player_id, actions):
        self.player_id = player_id
        self.player_name = f"Player {player_id + 1}"
        self.model_name = f"model-{player_id + 1}"
        self.total_output_tokens = 0
        self.actions = list(actions)
        self.logs = []
        self.rules = ""

    def reset_for_new_game(self):
        self.logs = []

    def send_rules(self, rules):
        self.rules = rules

    def add_log(self, message):
        self.logs.append(message)

    def inject_retry_error(self, error):
        raise AssertionError(f"Unexpected retry: {error}")

    def request_action(self, tool, state_info=None, action_description=None):
        expected_tool, arguments = self.actions.pop(0)
        actual_tool = tool["function"]["name"]
        if actual_tool != expected_tool:
            raise AssertionError(f"Expected {expected_tool}, got {actual_tool}")
        return arguments, ""


class AttackTests(unittest.TestCase):
    def test_subtraction_can_reduce_number_to_zero(self):
        self.assertEqual(evaluate_attack(5, "subtract", 5), (True, 0, True))

    def test_subtraction_rejects_numbers_outside_one_to_five(self):
        with self.assertRaises(ValueError):
            evaluate_attack(10, "subtract", 6)

    def test_division_by_one_is_valid_without_winning(self):
        self.assertEqual(evaluate_attack(10, "divide", 1), (True, 10, False))

    def test_indivisible_attack_fails_without_changing_number(self):
        self.assertEqual(evaluate_attack(10, "divide", 3), (False, 10, False))


class RunnerTests(unittest.TestCase):
    def test_lie_reverses_result_and_blocks_winning_attack(self):
        attacker = FakePlayer(0, [
            ("attack", {"operation": "divide", "number": 10}),
        ])
        defender = FakePlayer(1, [
            ("respond_to_attack", {"lie": True}),
        ])
        runner = NumberDecompositionRunner(attacker, defender, verbose=False)
        runner.round_number = 1
        runner.state = RoundState(numbers=[20, 10], lies_available=[True, True])

        winner = runner._run_turn()

        self.assertIsNone(winner)
        self.assertEqual(runner.state.numbers[1], 10)
        self.assertFalse(runner.state.lies_available[1])
        self.assertIn("failure", attacker.logs[-1])
        self.assertNotIn("lie", attacker.logs[-1].lower())

    def test_lie_can_report_failed_attack_as_success(self):
        attacker = FakePlayer(0, [
            ("attack", {"operation": "divide", "number": 3}),
        ])
        defender = FakePlayer(1, [
            ("respond_to_attack", {"lie": True}),
        ])
        runner = NumberDecompositionRunner(attacker, defender, verbose=False)
        runner.round_number = 1
        runner.state = RoundState(numbers=[20, 10], lies_available=[True, True])

        winner = runner._run_turn()

        self.assertIsNone(winner)
        self.assertEqual(runner.state.numbers[1], 10)
        self.assertIn("success", attacker.logs[-1])

    def test_used_lie_does_not_require_another_response(self):
        attacker = FakePlayer(0, [
            ("attack", {"operation": "subtract", "number": 2}),
        ])
        defender = FakePlayer(1, [])
        runner = NumberDecompositionRunner(attacker, defender, verbose=False)
        runner.round_number = 1
        runner.state = RoundState(numbers=[20, 10], lies_available=[True, False])

        winner = runner._run_turn()

        self.assertIsNone(winner)
        self.assertEqual(runner.state.numbers[1], 8)
        self.assertEqual(defender.actions, [])

    def test_second_player_wins_after_first_players_non_winning_sixteenth_turn(self):
        attacker = FakePlayer(0, [
            ("attack", {"operation": "divide", "number": 1}),
        ])
        defender = FakePlayer(1, [])
        runner = NumberDecompositionRunner(attacker, defender, verbose=False)
        runner.round_number = 1
        runner.state = RoundState(
            numbers=[20, 10],
            lies_available=[False, False],
            turns_taken=[15, 15],
            current_player=0,
            turn=31,
        )

        winner = runner._run_turn()

        self.assertEqual(winner, 1)
        self.assertEqual(runner.state.turns_taken, [16, 15])
        self.assertIn("16th turn", attacker.logs[-1])

    def test_first_player_can_still_win_on_sixteenth_turn(self):
        attacker = FakePlayer(0, [
            ("attack", {"operation": "divide", "number": 10}),
        ])
        defender = FakePlayer(1, [])
        runner = NumberDecompositionRunner(attacker, defender, verbose=False)
        runner.round_number = 1
        runner.state = RoundState(
            numbers=[20, 10],
            lies_available=[False, False],
            turns_taken=[15, 15],
            current_player=0,
            turn=31,
        )

        winner = runner._run_turn()

        self.assertEqual(winner, 0)
        self.assertEqual(runner.state.turns_taken, [16, 15])

    def test_lie_blocking_sixteenth_turn_win_awards_round_to_second_player(self):
        attacker = FakePlayer(0, [
            ("attack", {"operation": "divide", "number": 10}),
        ])
        defender = FakePlayer(1, [
            ("respond_to_attack", {"lie": True}),
        ])
        runner = NumberDecompositionRunner(attacker, defender, verbose=False)
        runner.round_number = 1
        runner.state = RoundState(
            numbers=[20, 10],
            lies_available=[False, True],
            turns_taken=[15, 15],
            current_player=0,
            turn=31,
        )

        winner = runner._run_turn()

        self.assertEqual(winner, 1)
        self.assertEqual(runner.state.numbers[1], 10)

    def test_best_of_three_resets_numbers_and_lies_each_round(self):
        player0 = FakePlayer(0, [
            ("choose_number", {"number": 10}),
            ("attack", {"operation": "divide", "number": 10}),
            ("choose_number", {"number": 12}),
            ("attack", {"operation": "divide", "number": 12}),
        ])
        player1 = FakePlayer(1, [
            ("choose_number", {"number": 10}),
            ("respond_to_attack", {"lie": False}),
            ("choose_number", {"number": 12}),
            ("respond_to_attack", {"lie": False}),
        ])
        runner = NumberDecompositionRunner(player0, player1, verbose=False)

        result = runner.run_game()

        self.assertEqual(result["winner"], 0)
        self.assertEqual(result["scores"], [2, 0])
        self.assertEqual(result["rounds_played"], 2)
        self.assertEqual(result["round_winners"], [0, 0])
        self.assertIn("Player 1 acts first", player0.logs[0])
        self.assertEqual(player0.actions, [])
        self.assertEqual(player1.actions, [])


if __name__ == "__main__":
    unittest.main()
