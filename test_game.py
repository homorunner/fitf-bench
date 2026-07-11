import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from cards import Card, Suit
from game import GameEngine, RoundState
from llm_player import LLMPlayer, parse_card_tool_argument


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
        )
        player = object.__new__(LLMPlayer)
        player.player_id = 0
        player.player_name = "Test Player"
        player.model = "test-model"
        player.messages = []
        player._pending_log_lines = []
        player.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )
        player.log_path = None
        player._request_number = 0
        return player

    def test_card_argument_parser_rejects_wrong_json_types(self):
        for arguments in (None, "null", "[]", '{"card": null}', '{"card": 3}'):
            card, error = parse_card_tool_argument(arguments)
            self.assertIsNone(card)
            self.assertTrue(error)

        self.assertEqual(parse_card_tool_argument('{"card": "B3"}'), ("B3", ""))

    def test_retry_without_tool_call_uses_user_message(self):
        player = object.__new__(LLMPlayer)
        player.messages = [{"role": "assistant", "content": "no tool call"}]

        player.inject_retry_error("missing call")

        self.assertEqual(player.messages[-1]["role"], "user")
        self.assertNotIn("tool_call_id", player.messages[-1])

    def test_retry_completes_every_tool_call(self):
        player = object.__new__(LLMPlayer)
        player.messages = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "one"}, {"id": "two"}],
        }]

        player.inject_retry_error("too many calls")

        self.assertEqual(
            [message["tool_call_id"] for message in player.messages[1:]],
            ["one", "two"],
        )

    def test_retry_reuses_request_method_without_adding_action_prompt(self):
        tool_call = SimpleNamespace(
            id="call-id",
            function=SimpleNamespace(name="play_card", arguments='{"card":"B3"}'),
        )
        message = SimpleNamespace(content="", tool_calls=[tool_call])
        player = self.make_player(message)
        player.messages = [{"role": "user", "content": "existing action"}]

        card, error = player.request_play_card(None, [Card(Suit.BELLS, 3)])

        self.assertEqual(card, Card(Suit.BELLS, 3))
        self.assertEqual(error, "")
        self.assertEqual(
            [entry["role"] for entry in player.messages],
            ["user", "assistant", "tool"],
        )

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
                to_dict=lambda: {
                    "id": "response-id",
                    "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                },
            )
            create = lambda **kwargs: response
            player = object.__new__(LLMPlayer)
            player.player_id = 0
            player.player_name = "Test Player"
            player.model = "test-model"
            player.messages = [{"role": "user", "content": "test"}]
            player.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            player.log_path = log_path
            player._request_number = 0

            result = player._call_llm([], "auto")

            self.assertIs(result, message)
            with open(log_path, "r", encoding="utf-8") as log_file:
                record = json.loads(log_file.read())
            self.assertEqual(record["player_name"], "Test Player")
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
            player.model = "test-model"
            player.messages = []
            player.client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            player.log_path = log_path
            player._request_number = 0

            with self.assertRaises(RuntimeError):
                player._call_llm([], "auto")

            with open(log_path, "r", encoding="utf-8") as log_file:
                record = json.loads(log_file.read())
            self.assertEqual(record["error"]["type"], "RuntimeError")
            self.assertEqual(record["error"]["message"], "service unavailable")


if __name__ == "__main__":
    unittest.main()
