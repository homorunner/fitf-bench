"""LLM player interface for text games using OpenAI-compatible tool calls."""

import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple

from openai import OpenAI

from fitf_bench.cards import Card, card_from_str


PLAY_CARD_TOOL = {
    "type": "function",
    "function": {
        "name": "play_card",
        "description": "Play a card from your hand in short format (e.g., 'B5', 'K11')",
        "parameters": {
            "type": "object",
            "properties": {
                "card": {
                    "type": "string",
                    "description": "The card to play in short format"
                }
            },
            "required": ["card"]
        }
    }
}

FOX_SWAP_TOOL = {
    "type": "function",
    "function": {
        "name": "fox_swap",
        "description": "Use the Fox (3) ability to swap the decree card with a card from your hand, or choose not to swap. Use 'none' to skip swapping, or provide the card in short format.",
        "parameters": {
            "type": "object",
            "properties": {
                "card": {
                    "type": "string",
                    "description": "The card to swap with the decree card in short format (e.g., 'B5', 'K11'), or 'none' to skip"
                }
            },
            "required": ["card"]
        }
    }
}

WOODCUTTER_DISCARD_TOOL = {
    "type": "function",
    "function": {
        "name": "woodcutter_discard",
        "description": "Use the Woodcutter (5) ability to discard one card from your hand to the bottom of the draw deck.",
        "parameters": {
            "type": "object",
            "properties": {
                "card": {
                    "type": "string",
                    "description": "The card to discard in short format (e.g., 'B5', 'K11')"
                }
            },
            "required": ["card"]
        }
    }
}


SYSTEM_PROMPT = """You are playing a two-player game. Use the provided rules, game log, and current state to choose the best action. Always respond with exactly one call to the provided tool."""


def parse_card_tool_argument(arguments: Any) -> Tuple[Optional[str], str]:
    """Parse and type-check a card tool's JSON arguments."""
    if isinstance(arguments, dict):
        args = arguments
    else:
        try:
            args = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return None, "Invalid JSON in tool arguments."

    if not isinstance(args, dict):
        return None, "Tool arguments must be a JSON object."

    card = args.get("card")
    if not isinstance(card, str):
        return None, "Tool argument 'card' must be a string."

    return card, ""


class LLMPlayer:
    """Manages LLM interaction for a single player.

    Context assembly is stateless: every API call sends exactly 3 messages.
    - messages[0]: system prompt (brief, never changes)
    - messages[1]: system prompt with full RULES.md (never changes)
    - messages[2]: single user message containing the cumulative game log
      (from game start to now), the current state, and the action request.

    No assistant/tool history is persisted. Stable rules and historical log
    prefixes can therefore be reused by providers that support prompt caching.
    """

    def __init__(self, player_id: int, api_base: str, api_key: str, model: str,
                 model_name: Optional[str] = None,
                 log_path: Optional[str] = None, extra_api_params: Optional[Dict[str, Any]] = None):
        self.player_id = player_id
        # Keep model identities out of prompts and game logs.
        self.player_name = f"Player {player_id + 1}"
        self.model_name = model_name or self.player_name
        self.model = model
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.messages: List[Dict[str, Any]] = []
        self._cumulative_log: List[str] = []
        self._rules_text: str = ""
        self._retry_lines: List[str] = []
        self._state_info: str = ""
        self._action_description: str = ""
        self.log_path = log_path
        self._request_number = 0
        self.total_output_tokens = 0
        self._extra_api_params = extra_api_params or {}

    def reset_for_new_game(self):
        self.messages = []
        self._cumulative_log = []
        self._retry_lines = []
        self._state_info = ""
        self._action_description = ""
        self.total_output_tokens = 0

    def send_rules(self, rules_text: str):
        """Register the full rules emitted on every request."""
        self._rules_text = rules_text

    def add_log(self, text: str):
        """Append game log text to the cumulative log (kept for the whole game)."""
        self._cumulative_log.append(text)

    def _call_llm(self, tools: List[dict], tool_choice: Any = "auto") -> Any:
        """Make LLM API call and return the response message."""
        self._request_number += 1
        request = {
            "model": self.model,
            "messages": self.messages,
            "tools": tools,
            "tool_choice": tool_choice,
            **self._extra_api_params,
        }
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "player_id": self.player_id,
            "player_name": self.player_name,
            "request_number": self._request_number,
            "request": request,
        }

        try:
            response = self.client.chat.completions.create(**request)
            record["response"] = self._serialize_api_value(response)
            # Accumulate output token usage
            if hasattr(response, "usage") and response.usage is not None:
                completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
                self.total_output_tokens += completion_tokens
            choices = getattr(response, "choices", None)
            if not choices:
                raise ValueError("API returned no choices in response.")
            message = choices[0].message
            self._write_api_log(record)
            return message
        except Exception as exc:
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self._write_api_log(record)
            raise

    @staticmethod
    def _serialize_api_value(value: Any) -> Any:
        """Convert OpenAI SDK response objects into JSON-serializable values."""
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value

    def _write_api_log(self, record: Dict[str, Any]):
        """Append one complete API request/response record as JSONL."""
        if not self.log_path:
            return
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            json.dump(record, log_file, ensure_ascii=False, default=str)
            log_file.write("\n")

    def _personalize_log(self, text: str) -> str:
        my_name = self.player_name
        opp_name = "Player 2" if self.player_id == 0 else "Player 1"

        text = text.replace(f"{my_name}'s", "\x00YOU_POSS\x00")
        text = text.replace(f"{opp_name}'s", "\x00OPP_POSS\x00")
        text = text.replace(my_name, "\x00YOU\x00")
        text = text.replace(opp_name, "Opponent")
        text = text.replace("\x00YOU_POSS\x00", "Your")
        text = text.replace("\x00OPP_POSS\x00", "Opponent's")
        text = text.replace("\x00YOU\x00", "You")

        def fix_verb(match):
            verb = match.group(1).lower()
            if verb == 'ha':
                return 'You have'
            if verb == 'i':
                return 'You are'
            if verb == 'doe':
                return 'You do'
            return f'You {verb}'

        text = re.sub(r'\bYou\s+(\w+)s\b', fix_verb, text)
        return text

    def _build_messages(self, state_info: str, action_description: str) -> List[Dict[str, Any]]:
        """Build the three-message context for a request."""
        parts = []
        if self._cumulative_log:
            parts.append("==Game Log==")
            log_lines = [self._personalize_log(line) for line in self._cumulative_log]
            parts.extend(log_lines)
            parts.append("")
        parts.extend((state_info, "", action_description))
        if self._retry_lines:
            parts.append("")
            parts.extend(self._retry_lines)
        user_content = "\n".join(parts)

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system",
             "content": f"Here are the complete rules of the game:\n\n{self._rules_text}\n\n"
                        f"You are {self.player_name}."},
            {"role": "user", "content": user_content},
        ]

    def request_action(self, tool: dict, state_info: Optional[str] = None,
                       action_description: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """Request one tool call and return its decoded argument object."""
        tool_name = tool["function"]["name"]
        # For the first attempt of an action, remember the state/action so that
        # retries (which pass state_info=None) can rebuild the same 3 messages.
        if state_info is not None:
            self._state_info = state_info
            self._action_description = action_description
            self._retry_lines = []

        self.messages = self._build_messages(
            self._state_info, self._action_description
        )

        try:
            msg = self._call_llm([tool], "auto")
        except Exception as e:
            return None, f"API error: {e}"

        if not msg.tool_calls:
            return None, f"No tool call made. You must use the {tool_name} tool."
        if len(msg.tool_calls) != 1:
            return None, f"Make exactly one {tool_name} tool call."

        tc = msg.tool_calls[0]
        if tc.function.name != tool_name:
            return None, f"Wrong tool called: {tc.function.name}. Use {tool_name}."

        try:
            arguments = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            return None, "Invalid JSON in tool arguments."
        if not isinstance(arguments, dict):
            return None, "Tool arguments must be a JSON object."
        return arguments, ""

    def request_play_card(self, state_info: Optional[str],
                          legal_cards: List[Card]) -> Tuple[Optional[Card], str]:
        legal_strs = [c.short_str() for c in legal_cards]
        action = None
        if state_info is not None:
            action = (f"It's your turn to play a card.\n"
                      f"Legal plays: {', '.join(legal_strs)}\n"
                      "Please use the play_card tool to choose your card.")

        arguments, error = self.request_action(PLAY_CARD_TOOL, state_info, action)
        if error:
            return None, error
        card_str, error = parse_card_tool_argument(arguments)
        if error:
            return None, error
        card = card_from_str(card_str)
        if card is None:
            return None, f"Invalid card format: '{card_str}'. Use format like 'B3', 'K11', 'M7'."
        if card not in legal_cards:
            return None, (f"Illegal play: {card}. Legal cards are: {', '.join(legal_strs)}. "
                         f"You must follow suit if possible.")

        return card, ""

    def request_fox_swap(self, state_info: Optional[str], legal_cards: List[Card],
                         decree_card: Card) -> Tuple[Optional[Card], bool, str]:
        legal_strs = [c.short_str() for c in legal_cards]
        action = None
        if state_info is not None:
            action = (f"Your Fox (3) ability activates!\n"
                      f"Current decree card: {decree_card} (Trump suit: {decree_card.suit.value})\n"
                      f"Your hand: {', '.join(legal_strs)}\n"
                      "You may swap the decree card with any card from your hand, or choose 'none' to skip.\n"
                      "Please use the fox_swap tool.")

        arguments, error = self.request_action(FOX_SWAP_TOOL, state_info, action)
        if error:
            return None, False, error
        card_str, error = parse_card_tool_argument(arguments)
        if error:
            return None, False, error

        if card_str.strip().lower() in ("none", "skip", "pass"):
            return None, True, ""

        card = card_from_str(card_str)
        if card is None:
            return None, False, f"Invalid card format: '{card_str}'. Use format like 'B3' or 'none'."

        if card not in legal_cards:
            return None, False, f"Card {card} is not in your hand. Your hand: {', '.join(legal_strs)}."

        return card, False, ""

    def request_woodcutter_discard(self, state_info: Optional[str],
                                   legal_cards: List[Card],
                                   drawn_card: Card) -> Tuple[Optional[Card], str]:
        legal_strs = [c.short_str() for c in legal_cards]
        action = None
        if state_info is not None:
            action = (f"Your Woodcutter (5) ability activates!\n"
                      f"You drew: {drawn_card}\n"
                      f"Your current hand: {', '.join(legal_strs)}\n"
                      "You must discard 1 card to the bottom of the draw deck.\n"
                      "Please use the woodcutter_discard tool.")

        arguments, error = self.request_action(
            WOODCUTTER_DISCARD_TOOL, state_info, action
        )
        if error:
            return None, error
        card_str, error = parse_card_tool_argument(arguments)
        if error:
            return None, error
        card = card_from_str(card_str)
        if card is None:
            return None, f"Invalid card format: '{card_str}'. Use format like 'B3', 'K11'."

        if card not in legal_cards:
            return None, f"Card {card} is not in your hand. Your hand: {', '.join(legal_strs)}."

        return card, ""

    def inject_retry_error(self, error_msg: str):
        """Add error feedback to the next attempt for the current action."""
        content = f"ERROR: {error_msg}\nPlease try again with a valid choice."
        self._retry_lines.append(content)
