"""Shared interfaces for two-player text games."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from fitf_bench.llm_player import LLMPlayer


class TwoPlayerGameRunner(ABC):
    """Base runner contract used by the tournament."""

    game_id: str

    def __init__(self, player1: LLMPlayer, player2: LLMPlayer,
                 verbose: bool = True, seed: Optional[Any] = None):
        self.players = [player1, player2]
        self.verbose = verbose
        self.seed = seed

    def log(self, message: str):
        if self.verbose:
            print(message)

    def broadcast_log(self, message: str):
        for player in self.players:
            player.add_log(message)

    def build_result(self, winner: int, reason: str, **game_data) -> Dict[str, Any]:
        """Build the common result shape consumed by the tournament."""
        return {
            "winner": winner,
            "reason": reason,
            "player_names": [player.model_name for player in self.players],
            "output_tokens": [player.total_output_tokens for player in self.players],
            **game_data,
        }

    @abstractmethod
    def run_game(self) -> Dict[str, Any]:
        """Return winner, reason, player_names, and output_tokens."""
