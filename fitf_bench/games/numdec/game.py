"""Number Decomposition game and LLM runner."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fitf_bench.base import TwoPlayerGameRunner
from fitf_bench.llm_player import LLMPlayer


FIRST_PLAYER_TURN_LIMIT = 16

# v2: starting number range narrowed from 2-100 to 10-90 to shorten rounds
# and reduce the first-player advantage.
MIN_STARTING_NUMBER = 10
MAX_STARTING_NUMBER = 90

# Number of recent games shown when choosing number.
RECENT_GAMES_LIMIT = 20

CHOOSE_NUMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_number",
        "description": "Secretly choose your starting number for this game.",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {"type": "integer", "minimum": MIN_STARTING_NUMBER,
                           "maximum": MAX_STARTING_NUMBER},
            },
            "required": ["number"],
        },
    },
}

ATTACK_TOOL = {
    "type": "function",
    "function": {
        "name": "attack",
        "description": "Attack the opponent's secret number with subtraction or division.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["subtract", "divide"]},
                "number": {"type": "integer", "minimum": 1},
            },
            "required": ["operation", "number"],
        },
    },
}

RESPOND_TOOL = {
    "type": "function",
    "function": {
        "name": "respond_to_attack",
        "description": "Secretly decide whether to spend your lie to block this attack.",
        "parameters": {
            "type": "object",
            "properties": {
                "lie": {"type": "boolean"},
            },
            "required": ["lie"],
        },
    },
}


@dataclass
class RoundState:
    numbers: list[int]
    lies_available: list[bool]
    turns_taken: list[int] = field(default_factory=lambda: [0, 0])
    current_player: int = 0
    turn: int = 1


def evaluate_attack(target: int, operation: str, number: int) -> Tuple[bool, int, bool]:
    """Return whether an attack succeeds, its resulting number, and whether it wins."""
    if operation == "subtract":
        if not 1 <= number <= 5:
            raise ValueError("Subtraction must use an integer from 1 to 5.")
        success = target >= number
        result = target - number if success else target
        return success, result, success and result == 0
    if operation == "divide":
        if number < 1:
            raise ValueError("Division must use a positive integer.")
        success = target % number == 0
        result = target // number if success else target
        return success, result, success and result == 1
    raise ValueError("Operation must be 'subtract' or 'divide'.")


def load_rules() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RULES.md")
    with open(path, "r", encoding="utf-8") as rules_file:
        return rules_file.read()


class NumberDecompositionRunner(TwoPlayerGameRunner):
    game_id = "number-decomposition"

    def __init__(self, player1: LLMPlayer, player2: LLMPlayer,
                 verbose: bool = True, seed: Optional[Any] = None,
                 results_dir: Optional[str] = None):
        super().__init__(player1, player2, verbose=verbose, seed=seed,
                         results_dir=results_dir)
        self.state: Optional[RoundState] = None
        self.starting_numbers: List[int] = []

    def run_game(self) -> Dict[str, Any]:
        rules = load_rules()
        for player in self.players:
            player.reset_for_new_game()
            player.send_rules(rules)

        winner = self._run_round()

        turns_taken = self.state.turns_taken.copy() if self.state else [0, 0]

        if self.stopped:
            return self.build_stopped_result(
                turns_taken=turns_taken,
                starting_numbers=self.starting_numbers.copy(),
            )

        message = f"Player {winner + 1} wins the game."
        self.log(message)
        self.broadcast_log(message)

        return self.build_result(
            winner, "win",
            turns_taken=turns_taken,
            starting_numbers=self.starting_numbers.copy(),
        )

    def _run_round(self) -> Optional[int]:
        self.broadcast_log("The game begins. Player 1 acts first.")

        history = self._format_recent_games()

        numbers = []
        for player_id in (0, 1):
            number = self._choose_number(player_id, history)
            if number is None:
                return None
            numbers.append(number)

        self.starting_numbers = numbers.copy()
        self.state = RoundState(numbers=numbers, lies_available=[True, True])
        for player_id, number in enumerate(numbers):
            self.players[player_id].add_log(
                f"Your secret starting number: {number}."
            )

        while not self.stopped:
            winner = self._run_turn()
            if winner is not None:
                return winner
        return None

    def _load_recent_games(self) -> List[Tuple[List[int], List[int], int]]:
        """Load starting numbers, turns taken, and winner of recent games.

        Returns up to RECENT_GAMES_LIMIT entries from the results directory
        (any players), most recent first.
        """
        if not self.results_dir or not os.path.isdir(self.results_dir):
            return []
        entries = []
        for filename in os.listdir(self.results_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.results_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as result_file:
                    data = json.load(result_file)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[WARN] Skipping unreadable result {path}: {exc}")
                continue
            if data.get("game_id") != self.game_id:
                continue
            if data.get("reason") != "win":
                continue
            numbers = data.get("starting_numbers")
            turns = data.get("turns_taken")
            winner = data.get("winner")
            if (not isinstance(numbers, list) or len(numbers) != 2
                    or not isinstance(turns, list) or len(turns) != 2
                    or winner not in (0, 1)):
                continue
            order = data.get("timestamp") or os.path.getmtime(path)
            entries.append((order, numbers, turns, winner))
        entries.sort(key=lambda entry: entry[0], reverse=True)
        return [(numbers, turns, winner)
                for _, numbers, turns, winner in entries[:RECENT_GAMES_LIMIT]]

    def _format_recent_games(self) -> str:
        recent = self._load_recent_games()
        if not recent:
            return ("")
        lines = [
            "== Recent Games ==",
            f"The last {len(recent)} recorded game(s) between various players, "
            "most recent first (starting numbers, turns taken, and winner):",
        ]
        for index, (numbers, turns, winner) in enumerate(recent, 1):
            winner_text = "first player won" if winner == 0 else "second player won"
            lines.append(
                f"{index}. first player started at {numbers[0]}, "
                f"second at {numbers[1]}; turns taken {turns[0]}-{turns[1]}; "
                f"{winner_text}"
            )
        return "\n".join(lines)

    def _choose_number(self, player_id: int, history: str) -> Optional[int]:
        state = (f"== Current State ==\nThe game is about to begin.\n\n"
                 f"{history}")
        action = (f"Choose your secret starting integer from "
                  f"{MIN_STARTING_NUMBER} to {MAX_STARTING_NUMBER}.")

        def attempt(first: bool):
            arguments, error = self.players[player_id].request_action(
                CHOOSE_NUMBER_TOOL, state if first else None, action
            )
            if error:
                return None, error
            number = arguments.get("number")
            if (isinstance(number, int) and not isinstance(number, bool)
                    and MIN_STARTING_NUMBER <= number <= MAX_STARTING_NUMBER):
                return number, ""
            return None, (f"The starting number must be an integer from "
                          f"{MIN_STARTING_NUMBER} to {MAX_STARTING_NUMBER}.")

        number, ok = self.request_with_retries(player_id, "choose_number", attempt)
        return number if ok else None

    def _run_turn(self) -> Optional[int]:
        state = self.state
        attacker = state.current_player
        defender = 1 - attacker
        attack = self._request_attack(attacker)
        if attack is None:
            return None
        operation, number = attack
        truthful_success, result, wins = evaluate_attack(
            state.numbers[defender], operation, number
        )

        lie = self._request_response(defender, operation, number, truthful_success)
        if lie is None:
            return None
        if lie:
            state.lies_available[defender] = False
            announced_success = not truthful_success
        else:
            announced_success = truthful_success
            if truthful_success:
                state.numbers[defender] = result

        operation_text = "subtracts" if operation == "subtract" else "divides by"
        result_text = "success" if announced_success else "failure"
        message = (f"[Turn {state.turn}] Player {attacker + 1} "
                   f"{operation_text} {number}: {result_text}.")
        self.log(message)
        self.broadcast_log(message)
        state.turns_taken[attacker] += 1

        if truthful_success and wins and not lie:
            return attacker
        if (attacker == 0
                and state.turns_taken[0] == FIRST_PLAYER_TURN_LIMIT):
            message = ("Player 1 did not win on their 16th turn. "
                       "Player 2 wins the game.")
            self.log(message)
            self.broadcast_log(message)
            return 1

        state.current_player = defender
        state.turn += 1
        return None

    def _request_attack(self, player_id: int) -> Optional[Tuple[str, int]]:
        state = self._format_state(player_id, "Your turn to attack.")
        action = ("Choose one attack. Subtraction requires an integer from 1 to 5; "
                  "division requires any positive integer.")

        def attempt(first: bool):
            arguments, error = self.players[player_id].request_action(
                ATTACK_TOOL, state if first else None, action
            )
            if error:
                return None, error
            operation = arguments.get("operation")
            number = arguments.get("number")
            if not isinstance(number, int) or isinstance(number, bool):
                return None, "The attack number must be an integer."
            try:
                evaluate_attack(2, operation, number)
            except ValueError as exc:
                return None, str(exc)
            return (operation, number), ""

        attack, ok = self.request_with_retries(player_id, "attack", attempt)
        return attack if ok else None

    def _request_response(self, player_id: int, operation: str, number: int,
                          truthful_success: bool) -> Optional[bool]:
        if not self.state.lies_available[player_id]:
            return False

        truth = "success" if truthful_success else "failure"
        state = self._format_state(
            player_id,
            f"Opponent attacks with {operation} {number}. The truthful result is {truth}.",
        )
        action = ("Secretly decide whether to spend your lie. If you lie, the opposite "
                  "result is announced and your number does not change.")

        def attempt(first: bool):
            arguments, error = self.players[player_id].request_action(
                RESPOND_TOOL, state if first else None, action
            )
            if error:
                return None, error
            lie = arguments.get("lie")
            if not isinstance(lie, bool):
                return None, "The lie field must be true or false."
            return lie, ""

        lie, ok = self.request_with_retries(player_id, "respond_to_attack", attempt)
        return lie if ok else None

    def _format_state(self, player_id: int, situation: str) -> str:
        state = self.state
        lie_status = "available" if state.lies_available[player_id] else "used"
        return (f"== Current State ==\nTurn: {state.turn}\n"
                f"Turns taken: You {state.turns_taken[player_id]} - "
                f"Opponent {state.turns_taken[1 - player_id]}\n"
                f"Your current number: {state.numbers[player_id]}\n"
                f"Your lie: {lie_status}\n{situation}")
