"""Match checkpointing for crash-safe resume of in-flight games.

A checkpoint file is an append-only JSONL file per match:
- line 1: meta record ({"type": "meta", match_id, game_id, model_a, model_b,
  seed}) written when the match starts.
- following lines: one record per successful action
  ({"player": 0|1, "tool": name, "arguments": {...}, "output_tokens": total}).
"""

import json
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


class MatchCheckpoint:
    """Action log for one match, shared by both players."""

    def __init__(self, path: str,
                 replay_records: Optional[List[Dict[str, Any]]] = None):
        self.path = path
        self._replay = deque(replay_records or [])

    @classmethod
    def create(cls, path: str, meta: Dict[str, Any]) -> "MatchCheckpoint":
        with open(path, "w", encoding="utf-8") as checkpoint_file:
            json.dump({"type": "meta", **meta}, checkpoint_file,
                      ensure_ascii=False)
            checkpoint_file.write("\n")
        return cls(path)

    @classmethod
    def load(cls, path: str) -> Tuple[Optional[Dict[str, Any]], "MatchCheckpoint"]:
        meta = None
        records: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as checkpoint_file:
                for line in checkpoint_file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if not isinstance(record, dict):
                        break
                    if record.get("type") == "meta":
                        meta = record
                    elif (record.get("player") in (0, 1)
                          and isinstance(record.get("arguments"), dict)):
                        records.append(record)
                    else:
                        break
        except OSError as exc:
            print(f"[WARN] Failed to read checkpoint {path}: {exc}")
            return None, cls(path)
        return meta, cls(path, records)

    def next_replay(self) -> Optional[Dict[str, Any]]:
        if self._replay:
            return self._replay.popleft()
        return None

    def record(self, player_id: int, tool_name: str,
               arguments: Dict[str, Any], output_tokens: int = 0):
        with open(self.path, "a", encoding="utf-8") as checkpoint_file:
            json.dump(
                {"player": player_id, "tool": tool_name,
                 "arguments": arguments, "output_tokens": output_tokens},
                checkpoint_file, ensure_ascii=False,
            )
            checkpoint_file.write("\n")

    def delete(self):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass  # already gone; deletion is idempotent
        except OSError as exc:
            print(f"[WARN] Failed to delete checkpoint {self.path}: {exc}")
