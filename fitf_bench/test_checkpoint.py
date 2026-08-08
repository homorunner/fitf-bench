import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from fitf_bench.checkpoint import MatchCheckpoint
from fitf_bench.llm_player import LLMPlayer


class CheckpointFileTests(unittest.TestCase):
    def test_create_then_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.jsonl")
            checkpoint = MatchCheckpoint.create(path, {
                "match_id": "m", "game_id": "g",
                "model_a": "a", "model_b": "b", "seed": "7",
            })
            checkpoint.record(0, "choose_number", {"number": 42}, 10)
            checkpoint.record(1, "choose_number", {"number": 50}, 20)

            meta, loaded = MatchCheckpoint.load(path)

        self.assertEqual(meta["match_id"], "m")
        self.assertEqual(meta["seed"], "7")
        self.assertEqual(loaded.next_replay()["arguments"], {"number": 42})
        self.assertEqual(loaded.next_replay()["arguments"], {"number": 50})
        self.assertIsNone(loaded.next_replay())

    def test_replay_returns_actions_in_recorded_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.jsonl")
            checkpoint = MatchCheckpoint.create(path, {"match_id": "m"})
            checkpoint.record(0, "attack", {"operation": "subtract", "number": 3})
            checkpoint.record(1, "respond_to_attack", {"lie": False})
            _, loaded = MatchCheckpoint.load(path)

        first = loaded.next_replay()
        second = loaded.next_replay()
        third = loaded.next_replay()

        self.assertEqual(first["arguments"], {"operation": "subtract", "number": 3})
        self.assertEqual(second["arguments"], {"lie": False})
        self.assertIsNone(third)

    def test_torn_final_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.jsonl")
            checkpoint = MatchCheckpoint.create(path, {"match_id": "m"})
            checkpoint.record(0, "attack", {"number": 1})
            with open(path, "a", encoding="utf-8") as checkpoint_file:
                checkpoint_file.write('{"player": 1, "tool": "resp')  # torn

            meta, loaded = MatchCheckpoint.load(path)

        self.assertEqual(meta["match_id"], "m")
        self.assertIsNotNone(loaded.next_replay())
        self.assertIsNone(loaded.next_replay())

    def test_resumed_checkpoint_appends_after_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.jsonl")
            checkpoint = MatchCheckpoint.create(path, {"match_id": "m"})
            checkpoint.record(0, "attack", {"number": 1})
            _, resumed = MatchCheckpoint.load(path)

            resumed.next_replay()
            resumed.record(1, "respond_to_attack", {"lie": False})

            _, reloaded = MatchCheckpoint.load(path)

        self.assertIsNotNone(reloaded.next_replay())
        self.assertIsNotNone(reloaded.next_replay())

    def test_delete_removes_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.jsonl")
            checkpoint = MatchCheckpoint.create(path, {"match_id": "m"})

            checkpoint.delete()
            checkpoint.delete()  # idempotent

            self.assertFalse(os.path.exists(path))

    def test_load_missing_file_returns_empty(self):
        meta, checkpoint = MatchCheckpoint.load("/nonexistent/m.jsonl")

        self.assertIsNone(meta)
        self.assertIsNone(checkpoint.next_replay())


def _make_player(player_id, checkpoint, tool_arguments='{"number": 60}'):
    """Bare LLMPlayer whose API returns a fixed tool call."""
    tool_call = SimpleNamespace(
        id="call-id",
        function=SimpleNamespace(name="choose_number",
                                 arguments=tool_arguments),
    )
    message = SimpleNamespace(content="", tool_calls=[tool_call])
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=None,
        to_dict=lambda: {"choices": [{"message": {}}]},
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    player = object.__new__(LLMPlayer)
    player.player_id = player_id
    player.player_name = f"Player {player_id + 1}"
    player.game_id = "test-game"
    player.model = "test-model"
    player.messages = []
    player._cumulative_log = []
    player._rules_text = "RULES"
    player._retry_lines = []
    player._state_info = ""
    player._action_description = ""
    player.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    player.log_path = None
    player._request_number = 0
    player.total_output_tokens = 0
    player._extra_api_params = {}
    player.checkpoint = checkpoint
    player._api_calls = calls
    return player


CHOOSE_NUMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_number",
        "description": "d",
        "parameters": {"type": "object"},
    },
}


class PlayerReplayTests(unittest.TestCase):
    def test_live_action_is_recorded_to_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.jsonl")
            checkpoint = MatchCheckpoint.create(path, {"match_id": "m"})
            player = _make_player(0, checkpoint)

            arguments, error = player.request_action(
                CHOOSE_NUMBER_TOOL, "state", "action")

            self.assertEqual(error, "")
            self.assertEqual(arguments, {"number": 60})
            self.assertEqual(len(player._api_calls), 1)

            _, reloaded = MatchCheckpoint.load(path)
            self.assertEqual(reloaded.next_replay()["arguments"],
                             {"number": 60})

    def test_replayed_action_skips_api_and_is_not_rerecorded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.jsonl")
            checkpoint = MatchCheckpoint.create(path, {"match_id": "m"})
            checkpoint.record(0, "choose_number", {"number": 42}, 99)
            _, resumed = MatchCheckpoint.load(path)
            player = _make_player(0, resumed)

            arguments, error = player.request_action(
                CHOOSE_NUMBER_TOOL, "state", "action")

            self.assertEqual(error, "")
            self.assertEqual(arguments, {"number": 42})
            self.assertEqual(player._api_calls, [])  # no API call
            self.assertEqual(player.total_output_tokens, 99)

            # The file still holds exactly one action record.
            _, reloaded = MatchCheckpoint.load(path)
            self.assertIsNotNone(reloaded.next_replay())
            self.assertIsNone(reloaded.next_replay())

    def test_live_play_continues_after_replay_queue_is_exhausted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "m.jsonl")
            checkpoint = MatchCheckpoint.create(path, {"match_id": "m"})
            checkpoint.record(0, "choose_number", {"number": 42}, 0)
            _, resumed = MatchCheckpoint.load(path)
            player = _make_player(0, resumed)

            first, _ = player.request_action(CHOOSE_NUMBER_TOOL, "s", "a")
            second, _ = player.request_action(CHOOSE_NUMBER_TOOL, "s", "a")

            self.assertEqual(first, {"number": 42})   # replayed
            self.assertEqual(second, {"number": 60})  # live
            self.assertEqual(len(player._api_calls), 1)

            _, reloaded = MatchCheckpoint.load(path)
            self.assertEqual(reloaded.next_replay()["arguments"],
                             {"number": 42})
            self.assertEqual(reloaded.next_replay()["arguments"],
                             {"number": 60})


if __name__ == "__main__":
    unittest.main()
