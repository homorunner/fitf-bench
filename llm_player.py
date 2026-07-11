"""LLM Player interface for Fox in the Forest via OpenAI-compatible API with tool calls."""

import json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple

from openai import OpenAI

from cards import Card, card_from_str


PLAY_CARD_TOOL = {
    "type": "function",
    "function": {
        "name": "play_card",
        "description": "Play a card from your hand. Use the short format: first letter of suit (B=Bells, K=Keys, M=Moons) followed by rank number. E.g., 'B3' means 3 of Bells, 'K11' means 11 of Keys.",
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
                    "description": "The card to swap with the decree card in short format (e.g., 'B5', 'K2'), or 'none' to skip"
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
                    "description": "The card to discard in short format (e.g., 'B5', 'K2')"
                }
            },
            "required": ["card"]
        }
    }
}


SYSTEM_PROMPT = """You are an excellent player playing "The Fox in the Forest", a two-player trick-taking card game. You will be asked to play cards using tool calls. Always use the appropriate tool."""


STATE_SECTION_MARKER = "=== Current State"


def parse_card_tool_argument(arguments: Any) -> Tuple[Optional[str], str]:
    """Parse and type-check a card tool's JSON arguments."""
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
    
    Context management strategy:
    - messages[0]: system prompt (always present)
    - messages[1]: user message with full RULES.md (always present)
    - messages[2]: assistant acknowledgement (always present)
    - messages[3+]: alternating user/assistant/tool messages for game actions
    
    Each user action prompt contains:
    1. Game log since last action (plays, results, events)
    2. Current state info + hand (ONLY in the latest user message)
    3. Action request with legal plays
    
    To keep prefix consistent: when a new user message is added, the previous
    user messages have their state section stripped (keeping only the game log part).
    """

    def __init__(self, player_id: int, api_base: str, api_key: str, model: str,
                 player_name: str = None, log_path: Optional[str] = None):
        self.player_id = player_id
        self.player_name = player_name or f"Player {player_id + 1}"
        self.model = model
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.messages: List[Dict[str, Any]] = []
        self._pending_log_lines: List[str] = []
        self.log_path = log_path
        self._request_number = 0

    def reset_for_new_game(self):
        """Reset conversation for a new game."""
        self.messages = []
        self._pending_log_lines = []

    def _ensure_system_prompt(self):
        """Ensure system prompt is set."""
        if not self.messages:
            self.messages.append({"role": "system", "content": SYSTEM_PROMPT})

    def send_rules(self, rules_text: str):
        """Send full rules as the first user message."""
        self._ensure_system_prompt()
        self.messages.append({
            "role": "user",
            "content": f"Here are the complete rules of the game:\n\n{rules_text}\n\nThe game will now begin. You are {self.player_name}. Good luck!"
        })
        self.messages.append({
            "role": "assistant",
            "content": "I understand the rules. I'm ready to play as " + self.player_name + ". Let's begin!"
        })

    def add_log(self, text: str):
        """Add game log text that will be included in the next prompt."""
        self._pending_log_lines.append(text)

    def _strip_state_from_old_messages(self):
        """Strip state/hand info from all previous user messages (keep only game log).
        
        This ensures only the latest user message has full state info,
        reducing context length while maintaining prefix consistency of the log portion.
        """
        for msg in self.messages[3:]:
            if msg.get("role") != "user":
                continue
            content = msg["content"]
            marker_idx = content.find(STATE_SECTION_MARKER)
            if marker_idx != -1:
                msg["content"] = content[:marker_idx].rstrip()

    def _build_action_prompt(self, state_info: str, action_description: str) -> str:
        """Build the user prompt for an action request.
        
        Structure:
        1. Game log (events since last action)
        2. Current state (will be stripped from this message later)
        3. Action request
        """
        parts = []
        if self._pending_log_lines:
            parts.append("=== Game Log ===")
            parts.extend(self._pending_log_lines)
            parts.append("")
            self._pending_log_lines = []

        parts.extend((state_info, "", action_description))
        return "\n".join(parts)

    def _call_llm(self, tools: List[dict], tool_choice: Any = "auto") -> Any:
        """Make LLM API call and return the response message."""
        self._request_number += 1
        request = {
            "model": self.model,
            "messages": self.messages,
            "tools": tools,
            "tool_choice": tool_choice,
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
            message = response.choices[0].message
            record["response"] = self._serialize_api_value(response)
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

    def _append_assistant_msg(self, msg) -> dict:
        """Append assistant message to conversation history."""
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                }
                for tc in msg.tool_calls
            ]
        self.messages.append(assistant_msg)
        return assistant_msg

    def _request_tool(self, tool: dict, tool_name: str,
                      state_info: Optional[str] = None,
                      action_description: Optional[str] = None):
        if state_info is not None:
            self._ensure_system_prompt()
            self._strip_state_from_old_messages()
            prompt = self._build_action_prompt(state_info, action_description)
            self.messages.append({"role": "user", "content": prompt})

        try:
            msg = self._call_llm([tool], "auto")
        except Exception as e:
            return None, None, f"API error: {e}"

        self._append_assistant_msg(msg)
        if not msg.tool_calls:
            return None, None, f"No tool call made. You must use the {tool_name} tool."
        if len(msg.tool_calls) != 1:
            return None, None, f"Make exactly one {tool_name} tool call."

        tc = msg.tool_calls[0]
        if tc.function.name != tool_name:
            return None, None, f"Wrong tool called: {tc.function.name}. Use {tool_name}."

        card_str, error = parse_card_tool_argument(tc.function.arguments)
        if error:
            return None, None, error
        return card_str, tc, ""

    def _add_tool_result(self, tool_call, content: str):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": content,
        })

    def request_play_card(self, state_info: Optional[str],
                          legal_cards: List[Card]) -> Tuple[Optional[Card], str]:
        legal_strs = [c.short_str() for c in legal_cards]
        action = None
        if state_info is not None:
            action = (f"It's your turn to play a card.\n"
                      f"Legal plays: {', '.join(legal_strs)}\n"
                      "Please use the play_card tool to choose your card.")

        card_str, tc, error = self._request_tool(
            PLAY_CARD_TOOL, "play_card", state_info, action
        )
        if error:
            return None, error
        card = card_from_str(card_str)
        if card is None:
            return None, f"Invalid card format: '{card_str}'. Use format like 'B3', 'K11', 'M7'."
        if card not in legal_cards:
            return None, (f"Illegal play: {card}. Legal cards are: {', '.join(legal_strs)}. "
                         f"You must follow suit if possible.")

        self._add_tool_result(tc, f"Card played successfully: {card}")
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

        card_str, tc, error = self._request_tool(
            FOX_SWAP_TOOL, "fox_swap", state_info, action
        )
        if error:
            return None, False, error

        if card_str.strip().lower() in ("none", "skip", "pass"):
            self._add_tool_result(tc, "You chose not to swap the decree card.")
            return None, True, ""

        card = card_from_str(card_str)
        if card is None:
            return None, False, f"Invalid card format: '{card_str}'. Use format like 'B3' or 'none'."

        if card not in legal_cards:
            return None, False, f"Card {card} is not in your hand. Your hand: {', '.join(legal_strs)}."

        self._add_tool_result(tc, f"Decree card swapped with {card}. New trump suit: {card.suit.value}.")
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

        card_str, tc, error = self._request_tool(
            WOODCUTTER_DISCARD_TOOL, "woodcutter_discard", state_info, action
        )
        if error:
            return None, error
        card = card_from_str(card_str)
        if card is None:
            return None, f"Invalid card format: '{card_str}'. Use format like 'B3', 'K11'."

        if card not in legal_cards:
            return None, f"Card {card} is not in your hand. Your hand: {', '.join(legal_strs)}."

        self._add_tool_result(tc, f"Discarded {card} to the bottom of the draw deck.")
        return card, ""

    def inject_retry_error(self, error_msg: str):
        """Inject error feedback for retry.
        
        After a failed attempt, the assistant message with the bad tool_call is already
        in self.messages. We add a tool result with the error, so the next API call
        will see the error and can retry.
        """
        content = f"ERROR: {error_msg}\nPlease try again with a valid choice."
        last_msg = self.messages[-1] if self.messages else None
        if last_msg and last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
            for tool_call in last_msg["tool_calls"]:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": content,
                })
        else:
            self.messages.append({"role": "user", "content": content})

    def notify_forced_play(self, text: str):
        """Notify the player of a forced play (added to pending log)."""
        self._pending_log_lines.append(text)
