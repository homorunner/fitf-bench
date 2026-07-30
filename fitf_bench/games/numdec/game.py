"""Number Decomposition game and LLM runner."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from fitf_bench.base import TwoPlayerGameRunner
from fitf_bench.llm_player import LLMPlayer


MAX_RETRIES = 3
FIRST_PLAYER_TURN_LIMIT = 16

CHOOSE_NUMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_number",
        "description": "Secretly choose your starting number for this round.",
        "parameters": {
            "type": "object",
            "properties": {
                "number": {"type": "integer", "minimum": 2, "maximum": 100},
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
                 verbose: bool = True, seed: Optional[Any] = None):
        super().__init__(player1, player2, verbose=verbose, seed=seed)
        self.round_wins = [0, 0]
        self.round_number = 0
        self.state: Optional[RoundState] = None
        self.forfeit_winner: Optional[int] = None

    def run_game(self) -> Dict[str, Any]:
        rules = load_rules()
        for player in self.players:
            player.reset_for_new_game()
            player.send_rules(rules)

        round_winners = []
        while max(self.round_wins) < 2 and self.forfeit_winner is None:
            winner = self._run_round()
            if winner is not None:
                self.round_wins[winner] += 1
                round_winners.append(winner)
                message = (f"Round {self.round_number} winner: Player {winner + 1}. "
                           f"Match score: {self.round_wins[0]}-{self.round_wins[1]}.")
                self.log(message)
                self.broadcast_log(message)

        if self.forfeit_winner is not None:
            winner = self.forfeit_winner
            reason = "forfeit"
        else:
            winner = 0 if self.round_wins[0] == 2 else 1
            reason = "rounds"

        return self.build_result(
            winner,
            reason,
            scores=self.round_wins.copy(),
            rounds_played=self.round_number,
            round_winners=round_winners,
        )

    def _run_round(self) -> Optional[int]:
        self.round_number += 1
        self.broadcast_log(f"Round {self.round_number} begins. Player 1 acts first.")

        numbers = []
        for player_id in (0, 1):
            number = self._choose_number(player_id)
            if number is None:
                return None
            numbers.append(number)

        self.state = RoundState(numbers=numbers, lies_available=[True, True])
        for player_id, number in enumerate(numbers):
            self.players[player_id].add_log(
                f"Your secret starting number for round {self.round_number}: {number}."
            )

        while self.forfeit_winner is None:
            winner = self._run_turn()
            if winner is not None:
                return winner
        return None

    def _choose_number(self, player_id: int) -> Optional[int]:
        state = (f"== Current State ==\nRound: {self.round_number}\n"
                 f"Match score: You {self.round_wins[player_id]} - "
                 f"Opponent {self.round_wins[1 - player_id]}")
        action = "Choose your secret starting integer from 2 to 100."

        for attempt in range(MAX_RETRIES):
            if attempt:
                self.players[player_id].inject_retry_error(error)
            arguments, error = self.players[player_id].request_action(
                CHOOSE_NUMBER_TOOL, state if attempt == 0 else None, action
            )
            if not error:
                number = arguments.get("number")
                if isinstance(number, int) and not isinstance(number, bool) and 2 <= number <= 100:
                    return number
                error = "The starting number must be an integer from 2 to 100."

        self._forfeit(player_id, "failed to choose a valid starting number")
        return None

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
        message = (f"[Round {self.round_number}, Turn {state.turn}] Player {attacker + 1} "
                   f"{operation_text} {number}: {result_text}.")
        self.log(message)
        self.broadcast_log(message)
        state.turns_taken[attacker] += 1

        if truthful_success and wins and not lie:
            return attacker
        if (attacker == 0
                and state.turns_taken[0] == FIRST_PLAYER_TURN_LIMIT):
            message = (f"Player 1 did not win on their 16th turn. "
                       f"Player 2 wins round {self.round_number}.")
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

        for attempt in range(MAX_RETRIES):
            if attempt:
                self.players[player_id].inject_retry_error(error)
            arguments, error = self.players[player_id].request_action(
                ATTACK_TOOL, state if attempt == 0 else None, action
            )
            if not error:
                operation = arguments.get("operation")
                number = arguments.get("number")
                if not isinstance(number, int) or isinstance(number, bool):
                    error = "The attack number must be an integer."
                else:
                    try:
                        evaluate_attack(2, operation, number)
                        return operation, number
                    except ValueError as exc:
                        error = str(exc)

        self._forfeit(player_id, "failed to choose a valid attack")
        return None

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

        for attempt in range(MAX_RETRIES):
            if attempt:
                self.players[player_id].inject_retry_error(error)
            arguments, error = self.players[player_id].request_action(
                RESPOND_TOOL, state if attempt == 0 else None, action
            )
            if not error:
                lie = arguments.get("lie")
                if not isinstance(lie, bool):
                    error = "The lie field must be true or false."
                else:
                    return lie

        self._forfeit(player_id, "failed to give a valid response")
        return None

    def _format_state(self, player_id: int, situation: str) -> str:
        state = self.state
        lie_status = "available" if state.lies_available[player_id] else "used"
        return (f"== Current State ==\nRound: {self.round_number}\nTurn: {state.turn}\n"
                f"Match score: You {self.round_wins[player_id]} - "
                f"Opponent {self.round_wins[1 - player_id]}\n"
                f"Turns taken: You {state.turns_taken[player_id]} - "
                f"Opponent {state.turns_taken[1 - player_id]}\n"
                f"Your current number: {state.numbers[player_id]}\n"
                f"Your lie: {lie_status}\n{situation}")

    def _forfeit(self, player_id: int, reason: str):
        self.log(f"Player {player_id + 1} forfeits: {reason}.")
        self.forfeit_winner = 1 - player_id
