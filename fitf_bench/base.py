"""Shared interfaces for two-player text games."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from fitf_bench.llm_player import API_ERROR, LLMPlayer, error_kind


API_ERROR_LIMIT = 3
MODEL_ERROR_LIMIT = 3


class TwoPlayerGameRunner(ABC):
    """Base runner contract used by the tournament."""

    game_id: str

    def __init__(self, player1: LLMPlayer, player2: LLMPlayer,
                 verbose: bool = True, seed: Optional[Any] = None,
                 results_dir: Optional[str] = None):
        self.players = [player1, player2]
        self.verbose = verbose
        self.seed = seed
        # Directory of recorded results; games may use it for historical stats.
        self.results_dir = results_dir
        # Player who wins because the opponent forfeited (model errors).
        self.forfeit_winner: Optional[int] = None
        # Player whose API failed repeatedly; the match is aborted (no winner).
        self.abort_player: Optional[int] = None

    @property
    def stopped(self) -> bool:
        """True when the game must stop early (forfeit or abort)."""
        return self.forfeit_winner is not None or self.abort_player is not None

    def log(self, message: str):
        if self.verbose:
            print(message)

    def broadcast_log(self, message: str):
        for player in self.players:
            player.add_log(message)

    def request_with_retries(
        self, player: int, action_name: str,
        attempt_fn: Callable[[bool], Tuple[Any, str]],
    ) -> Tuple[Any, bool]:
        api_errors = 0
        model_errors = 0
        first = True
        while True:
            value, error = attempt_fn(first)
            first = False
            if not error:
                return value, True

            name = self.players[player].player_name
            if error_kind(error) == API_ERROR:
                api_errors += 1
                self.log(f"  [ERROR] {name} {action_name} "
                         f"(API error {api_errors}/{API_ERROR_LIMIT}): {error}")
                if api_errors >= API_ERROR_LIMIT:
                    self.log(f"  [ABORT] {name} hit {api_errors} API errors during "
                             f"{action_name}; match aborted.")
                    self.abort_player = player
                    return None, False
            else:
                model_errors += 1
                self.log(f"  [ERROR] {name} {action_name} "
                         f"(invalid action {model_errors}/{MODEL_ERROR_LIMIT}): {error}")
                if model_errors >= MODEL_ERROR_LIMIT:
                    self.log(f"  [FORFEIT] {name} forfeits: {model_errors} invalid "
                             f"actions during {action_name}.")
                    self.forfeit_winner = 1 - player
                    return None, False
                # Only model errors are worth feeding back to the model.
                self.players[player].inject_retry_error(error)

    def build_result(self, winner: Optional[int], reason: str,
                     **game_data) -> Dict[str, Any]:
        """Build the common result shape consumed by the tournament."""
        return {
            "winner": winner,
            "reason": reason,
            "player_names": [player.model_name for player in self.players],
            "output_tokens": [player.total_output_tokens for player in self.players],
            **game_data,
        }

    def build_stopped_result(self, **game_data) -> Dict[str, Any]:
        """Build the result for a game stopped early by abort or forfeit."""
        if self.abort_player is not None:
            return self.build_result(
                None, "api_error", api_error_player=self.abort_player, **game_data
            )
        return self.build_result(self.forfeit_winner, "forfeit", **game_data)

    @abstractmethod
    def run_game(self) -> Dict[str, Any]:
        """Return winner, reason, player_names, and output_tokens."""
