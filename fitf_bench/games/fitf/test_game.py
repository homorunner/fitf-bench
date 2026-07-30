import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from elo import load_results
from fitf_bench.games.fitf.cards import Card, Suit
from fitf_bench.game_registry import get_game
from fitf_bench.games.fitf.game import GameEngine, RoundState
from fitf_bench.llm_player import LLMPlayer, parse_card_tool_argument


def make_round(hands, leader=0, trump=Suit.MOONS):
    return RoundState(
        round_number=1,
        dealer=1,
        hands=hands,
        draw_deck=[],
        decree_card=Card(trump, 2),
        trump_suit=trump,
        tricks_won=[0, 0],
        bonus_points=[0, 0],
        current_leader=leader,
        trick_number=1,
    )


class GameEngineTests(unittest.TestCase):
    def test_treasure_scores_immediately_and_is_not_added_twice(self):
        engine = GameEngine()
        engine.current_round = make_round([[], []])
        rs = engine.current_round
        rs._lead_card = Card(Suit.BELLS, 7)
        rs._follow_card = Card(Suit.BELLS, 6)
        rs._lead_player = 0
        rs._follow_player = 1

        engine.resolve_trick()

        self.assertEqual(engine.scores, [1, 0])
        self.assertEqual(rs.bonus_points, [1, 0])

        rs.tricks_won = [7, 6]
        rs.trick_number = 14
        round_scores = engine.score_round()[:2]
        self.assertEqual(round_scores, (7, 3))
        self.assertEqual(engine.scores, [7, 3])

    def test_illegal_off_suit_follow_is_rejected_without_mutation(self):
        lead = Card(Suit.BELLS, 2)
        legal_follow = Card(Suit.BELLS, 4)
        illegal_follow = Card(Suit.KEYS, 6)
        engine = GameEngine()
        engine.current_round = make_round([[lead], [legal_follow, illegal_follow]])
        engine.play_lead_card(0, lead)

        with self.assertRaises(ValueError):
            engine.play_follow_card(1, illegal_follow)

        self.assertEqual(engine.current_round.hands[1], [legal_follow, illegal_follow])
        self.assertFalse(hasattr(engine.current_round, "_follow_card"))

    def test_fox_can_only_be_resolved_by_ability_owner(self):
        fox = Card(Suit.BELLS, 3)
        swap = Card(Suit.KEYS, 4)
        engine = GameEngine()
        engine.current_round = make_round([[fox, swap], []])
        engine.play_lead_card(0, fox)

        with self.assertRaises(ValueError):
            engine.resolve_fox_swap(1, None)

        self.assertTrue(engine.current_round.pending_fox_swap)
        engine.resolve_fox_swap(0, swap)
        self.assertFalse(engine.current_round.pending_fox_swap)
        self.assertEqual(engine.current_round.fox_player, -1)

    def test_woodcutter_can_only_be_resolved_by_ability_owner(self):
        woodcutter = Card(Suit.BELLS, 5)
        discard = Card(Suit.KEYS, 4)
        drawn = Card(Suit.MOONS, 8)
        engine = GameEngine()
        engine.current_round = make_round([[woodcutter, discard], []])
        engine.current_round.draw_deck = [drawn]
        engine.play_lead_card(0, woodcutter)

        with self.assertRaises(ValueError):
            engine.resolve_woodcutter_discard(1, discard)

        self.assertTrue(engine.current_round.pending_woodcutter)
        engine.resolve_woodcutter_discard(0, discard)
        self.assertFalse(engine.current_round.pending_woodcutter)
        self.assertEqual(engine.current_round.woodcutter_player, -1)


class ToolProtocolTests(unittest.TestCase):
    @staticmethod
    def make_player(message):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            to_dict=lambda: {"choices": [{"message": {}}]},
            usage=None,
        )
        captured = {}

        def create(**kwargs):
            captured["request"] = kwargs
            return response

        player = object.__new__(LLMPlayer)
        player.player_id = 0
        player.player_name = "Test Player"
        player.game_id = "test-game"
        player.model = "test-model"
        player.messages = []
        player._cumulative_log = []
        player._rules_text = "RULES TEXT"
        player._retry_lines = []
        player._state_info = ""
        player._action_description = ""
        player.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        player.log_path = None
        player._request_number = 0
        player.total_output_tokens = 0
        player._extra_api_params = {}
        player._captured = captured
        return player

    def test_card_argument_parser_rejects_wrong_json_types(self):
        for arguments in (None, "null", "[]", '{"card": null}', '{"card": 3}'):
            card, error = parse_card_tool_argument(arguments)
            self.assertIsNone(card)
            self.assertTrue(error)

        self.assertEqual(parse_card_tool_argument('{"card": "B3"}'), ("B3", ""))
        self.assertEqual(parse_card_tool_argument({"card": "B3"}), ("B3", ""))

    def test_generic_action_returns_decoded_arguments(self):
        tool_call = SimpleNamespace(
            id="call-id",
            function=SimpleNamespace(
                name="choose_move", arguments='{"row":2,"column":3}'
            ),
        )
        player = self.make_player(
            SimpleNamespace(content="", tool_calls=[tool_call])
        )
        tool = {
            "type": "function",
            "function": {
                "name": "choose_move",
                "description": "Choose a move",
                "parameters": {"type": "object"},
            },
        }

        arguments, error = player.request_action(tool, "state", "choose")

        self.assertEqual(arguments, {"row": 2, "column": 3})
        self.assertEqual(error, "")

    def test_default_game_is_registered(self):
        game = get_game("fox-in-the-forest")

        self.assertEqual(game.name, "The Fox in the Forest")
        self.assertEqual(game.runner_class.game_id, game.game_id)

    def test_elo_results_are_filtered_by_game(self):
        with tempfile.TemporaryDirectory() as directory:
            results = [
                {"game_id": "fox-in-the-forest", "player_names": ["a", "b"],
                 "winner": 0},
                {"game_id": "other-game", "player_names": ["a", "b"],
                 "winner": 1},
                {"player_names": ["a", "b"], "winner": 0},
            ]
            for index, result in enumerate(results):
                with open(os.path.join(directory, f"{index}.json"), "w",
                          encoding="utf-8") as result_file:
                    json.dump(result, result_file)

            loaded = load_results(directory, "other-game")

        self.assertEqual(loaded, [results[1]])

    def test_sends_exactly_three_messages_with_cumulative_log(self):
        tool_call = SimpleNamespace(
            id="call-id",
            function=SimpleNamespace(name="play_card", arguments='{"card":"B3"}'),
        )
        message = SimpleNamespace(content="", tool_calls=[tool_call])
        player = self.make_player(message)
        player._cumulative_log = ["event one", "event two"]

        with patch("fitf_bench.llm_player.time.sleep"):
            card, error = player.request_play_card(
                "==Current State==\nYour hand: B3", [Card(Suit.BELLS, 3)]
            )

        self.assertEqual(card, Card(Suit.BELLS, 3))
        self.assertEqual(error, "")
        req_messages = player._captured["request"]["messages"]
        self.assertEqual([m["role"] for m in req_messages],
                         ["system", "system", "user"])
        self.assertIn("RULES TEXT", req_messages[1]["content"])
        user_content = req_messages[2]["content"]
        self.assertIn("event one", user_content)
        self.assertIn("event two", user_content)
        self.assertIn("==Current State==", user_content)
        self.assertEqual([m["role"] for m in player.messages],
                         ["system", "system", "user"])

    def test_retry_appends_error_and_keeps_three_messages(self):
        tool_call = SimpleNamespace(
            id="call-id",
            function=SimpleNamespace(name="play_card", arguments='{"card":"B3"}'),
        )
        message = SimpleNamespace(content="", tool_calls=[tool_call])
        player = self.make_player(message)
        player._state_info = "==Current State==\nYour hand: B3"
        player._action_description = "Play a card."
        player.inject_retry_error("Illegal play")

        card, error = player.request_play_card(None, [Card(Suit.BELLS, 3)])

        self.assertEqual(card, Card(Suit.BELLS, 3))
        req_messages = player._captured["request"]["messages"]
        self.assertEqual(len(req_messages), 3)
        self.assertIn("Illegal play", req_messages[2]["content"])

        player.request_play_card("==Current State==\nYour hand: B3",
                                 [Card(Suit.BELLS, 3)])
        self.assertEqual(player._retry_lines, [])
        self.assertNotIn("Illegal play",
                         player._captured["request"]["messages"][2]["content"])

    def test_api_call_logs_request_and_response(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, "game.jsonl")
            message = SimpleNamespace(
                content="ok",
                tool_calls=None,
                to_dict=lambda: {"content": "ok", "tool_calls": None},
            )
            response = SimpleNamespace(
                choices=[SimpleNamespace(message=message)],
                usage=None,
                to_dict=lambda: {
                    "id": "response-id",
                    "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                },
            )
            create = lambda **kwargs: response
            player = object.__new__(LLMPlayer)
            player.player_id = 0
            player.player_name = "Test Player"
            player.game_id = "test-game"
            player.model = "test-model"
            player.messages = [{"role": "user", "content": "test"}]
            player.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            player.log_path = log_path
            player._request_number = 0
            player.total_output_tokens = 0
            player._extra_api_params = {}

            result = player._call_llm([], "auto")

            self.assertIs(result, message)
            with open(log_path, "r", encoding="utf-8") as log_file:
                record = json.loads(log_file.read())
            self.assertEqual(record["player_name"], "Test Player")
            self.assertEqual(record["game_id"], "test-game")
            self.assertEqual(record["request"]["tool_choice"], "auto")
            self.assertEqual(record["response"]["id"], "response-id")
            self.assertEqual(record["response"]["choices"][0]["message"]["content"], "ok")

    def test_api_call_logs_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, "game.jsonl")

            def create(**kwargs):
                raise RuntimeError("service unavailable")

            player = object.__new__(LLMPlayer)
            player.player_id = 1
            player.player_name = "Test Player"
            player.game_id = "test-game"
            player.model = "test-model"
            player.messages = []
            player.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            player.log_path = log_path
            player._request_number = 0
            player.total_output_tokens = 0
            player._extra_api_params = {}

            with patch("fitf_bench.llm_player.time.sleep") as sleep:
                with self.assertRaises(RuntimeError):
                    player._call_llm([], "auto")

            with open(log_path, "r", encoding="utf-8") as log_file:
                records = [json.loads(line) for line in log_file if line.strip()]
            self.assertEqual(len(records), 4)
            self.assertEqual([record["request_number"] for record in records],
                             [1, 2, 3, 4])
            self.assertTrue(all(record["error"]["type"] == "RuntimeError"
                                for record in records))
            self.assertTrue(all(record["error"]["message"] == "service unavailable"
                                for record in records))
            self.assertEqual([call.args[0] for call in sleep.call_args_list],
                             [1, 3, 10])

    def test_api_call_empty_choices_raises_and_logs_response(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = os.path.join(directory, "game.jsonl")
            response = SimpleNamespace(
                choices=[],
                usage=None,
                to_dict=lambda: {"id": "response-id", "choices": []},
            )

            def create(**kwargs):
                return response

            player = object.__new__(LLMPlayer)
            player.player_id = 0
            player.player_name = "Test Player"
            player.game_id = "test-game"
            player.model = "test-model"
            player.messages = []
            player.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            player.log_path = log_path
            player._request_number = 0
            player.total_output_tokens = 0
            player._extra_api_params = {}

            with patch("fitf_bench.llm_player.time.sleep"):
                with self.assertRaises(ValueError):
                    player._call_llm([], "auto")

            with open(log_path, "r", encoding="utf-8") as log_file:
                lines = [l for l in log_file.read().splitlines() if l.strip()]
            self.assertEqual(len(lines), 4)
            records = [json.loads(line) for line in lines]
            self.assertTrue(all(record["response"]["id"] == "response-id"
                                for record in records))
            self.assertTrue(all(record["error"]["type"] == "ValueError"
                                for record in records))

    def test_empty_choices_is_retryable_api_error(self):
        response = SimpleNamespace(choices=[], usage=None,
                                   to_dict=lambda: {"choices": []})
        player = self.make_player(SimpleNamespace(content="", tool_calls=[]))
        player.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: response)
            )
        )
        with patch("fitf_bench.llm_player.time.sleep"):
            card, error = player.request_play_card(
                "==Current State==\nYour hand: B3", [Card(Suit.BELLS, 3)]
            )
        self.assertIsNone(card)
        self.assertIn("API error", error)


if __name__ == "__main__":
    unittest.main()
