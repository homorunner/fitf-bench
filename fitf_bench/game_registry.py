"""Game registration for the benchmark."""

from dataclasses import dataclass
from typing import Dict, Type

from fitf_bench.base import TwoPlayerGameRunner
from fitf_bench.games.fitf.runner import GameRunner
from fitf_bench.games.numdec.game import NumberDecompositionRunner


@dataclass(frozen=True)
class GameDefinition:
    game_id: str
    name: str
    runner_class: Type[TwoPlayerGameRunner]

    def create_runner(self, player1, player2, *, verbose: bool, seed):
        return self.runner_class(player1, player2, verbose=verbose, seed=seed)


GAMES: Dict[str, GameDefinition] = {
    "fox-in-the-forest": GameDefinition(
        game_id="fox-in-the-forest",
        name="The Fox in the Forest",
        runner_class=GameRunner,
    ),
    "number-decomposition": GameDefinition(
        game_id="number-decomposition",
        name="Number Decomposition",
        runner_class=NumberDecompositionRunner,
    ),
}


def get_game(game_id: str) -> GameDefinition:
    try:
        return GAMES[game_id]
    except KeyError as exc:
        raise ValueError(f"Unknown game: {game_id}") from exc
