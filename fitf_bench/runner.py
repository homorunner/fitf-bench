"""Fox in the Forest runner for a single match between two LLM players."""

import os
from typing import Optional

from fitf_bench.base import TwoPlayerGameRunner
from fitf_bench.cards import Card, format_hand
from fitf_bench.game import GameEngine
from fitf_bench.llm_player import LLMPlayer


MAX_RETRIES = 3


def load_rules() -> str:
    rules_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RULES.md")
    with open(rules_path, "r", encoding="utf-8") as f:
        return f.read()


class GameRunner(TwoPlayerGameRunner):
    """Orchestrates the game between two LLM players."""

    game_id = "fox-in-the-forest"

    def __init__(self, player1: LLMPlayer, player2: LLMPlayer,
                 verbose: bool = True, seed: Optional[int] = None):
        super().__init__(player1, player2, verbose=verbose, seed=seed)
        self.engine = GameEngine(seed=seed)
        self.forfeit_winner: Optional[int] = None

    def run_game(self) -> dict:
        rules = load_rules()
        for p in self.players:
            p.reset_for_new_game()
            p.send_rules(rules)

        while not self.engine.game_over:
            self.run_round()
            if self.forfeit_winner is not None:
                break

        if self.forfeit_winner is not None:
            winner = self.forfeit_winner
            reason = "forfeit"
        else:
            winner = self.engine.winner
            reason = "score"

        self.log("\n  GAME OVER")
        self.log(f"  Final scores: {self.players[0].player_name}: {self.engine.scores[0]}, "
                 f"{self.players[1].player_name}: {self.engine.scores[1]}")
        self.log(f"  Winner: {self.players[winner].player_name} ({reason})")

        return self.build_result(
            winner,
            reason,
            scores=self.engine.scores.copy(),
            rounds_played=self.engine.round_number,
        )

    def run_round(self):
        rs = self.engine.start_new_round()

        round_start_msg = (
            f"\nROUND {rs.round_number} BEGINS\n"
            f"Dealer: {self.players[rs.dealer].player_name}\n"
            f"Decree card: {rs.decree_card} (Trump suit: {rs.trump_suit.value})\n"
        )
        self.log(round_start_msg)
        self.broadcast_log(round_start_msg)
        for p in [0, 1]:
            self.players[p].add_log(f"Your hand: {format_hand(rs.hands[p])}")

        while not self.engine.is_round_over():
            if self.forfeit_winner is not None:
                return
            self.run_trick()

        _, _, details = self.engine.score_round()

        score_msg = (
            f"\n--- Round {rs.round_number} Scoring ---\n"
            f"  {self.players[0].player_name}: {details['tricks_won'][0]} tricks -> "
            f"{details['base_points'][0]} base + {details['bonus_points'][0]} bonus = {details['round_points'][0]} points\n"
            f"  {self.players[1].player_name}: {details['tricks_won'][1]} tricks -> "
            f"{details['base_points'][1]} base + {details['bonus_points'][1]} bonus = {details['round_points'][1]} points\n"
            f"  Total scores: {self.players[0].player_name}: {details['total_scores'][0]}, "
            f"{self.players[1].player_name}: {details['total_scores'][1]}"
        )
        self.log(score_msg)
        self.broadcast_log(score_msg)

    def run_trick(self):
        rs = self.engine.current_round
        leader = rs.current_leader
        follower = 1 - leader

        trick_num = rs.trick_number
        self.log(f"\n--- Trick {trick_num} ---")

        lead_card = self._get_play(leader, is_lead=True)
        if lead_card is None:
            return

        events = self.engine.play_lead_card(leader, lead_card)
        lead_msg = f"  {self.players[leader].player_name} leads: {lead_card}"
        self.log(lead_msg)

        self._emit_events(events)
        self.broadcast_log(f"[Trick {trick_num}] {self.players[leader].player_name} leads: {lead_card}")
        for event in events:
            self.broadcast_log(f"  * {event}")

        if not self._handle_pending_abilities(leader):
            return

        follow_card = self._get_play(follower, is_lead=False, lead_card=lead_card)
        if follow_card is None:
            return

        events = self.engine.play_follow_card(follower, follow_card)
        follow_msg = f"  {self.players[follower].player_name} follows: {follow_card}"
        self.log(follow_msg)

        self._emit_events(events)
        self.broadcast_log(f"  {self.players[follower].player_name} follows: {follow_card}")
        for event in events:
            self.broadcast_log(f"  * {event}")

        if not self._handle_pending_abilities(follower):
            return

        result = self.engine.resolve_trick()
        winner_msg = f"  Winner: {self.players[result.winner].player_name}"
        self.log(winner_msg)
        self._emit_events(result.events)

        self.broadcast_log(f"  Winner: {self.players[result.winner].player_name}")
        for event in result.events:
            self.broadcast_log(f"  * {event}")

        rs = self.engine.current_round
        tricks_msg = (f"  Tricks so far - {self.players[0].player_name}: {rs.tricks_won[0]}, "
                     f"{self.players[1].player_name}: {rs.tricks_won[1]}")
        self.log(tricks_msg)
        self.broadcast_log(tricks_msg)

    def _emit_events(self, events, broadcast: bool = False):
        for event in events:
            message = f"  * {event}"
            self.log(message)
            if broadcast:
                self.broadcast_log(message)

    def _handle_pending_abilities(self, player: int) -> bool:
        rs = self.engine.current_round
        if rs.pending_fox_swap and rs.fox_player == player:
            self._handle_fox_swap(player)
            if self.forfeit_winner is not None:
                return False
        if rs.pending_woodcutter and rs.woodcutter_player == player:
            self._handle_woodcutter_discard(player)
        return self.forfeit_winner is None

    def _get_play(self, player: int, is_lead: bool, lead_card: Optional[Card] = None) -> Optional[Card]:
        legal_cards = self.engine.get_legal_plays(player, lead_card if not is_lead else None)

        if len(legal_cards) == 1:
            card = legal_cards[0]
            auto_msg = f"  (Auto-play: {self.players[player].player_name} plays {card})"
            self.log(auto_msg)
            return card

        state_info = self.engine.format_game_state(player)

        for attempt in range(MAX_RETRIES):
            if attempt:
                self.players[player].inject_retry_error(error)
            card, error = self.players[player].request_play_card(
                state_info if attempt == 0 else None, legal_cards
            )

            if card is not None:
                return card

            self.log(f"  [ERROR] {self.players[player].player_name} attempt {attempt+1}/{MAX_RETRIES}: {error}")

        self.log(f"  [FORFEIT] {self.players[player].player_name} failed to make a legal play after {MAX_RETRIES} attempts.")
        self.forfeit_winner = 1 - player
        return None

    def _handle_fox_swap(self, player: int):
        rs = self.engine.current_round
        legal_cards = self.engine.get_legal_plays_for_fox_swap(player)

        if not legal_cards:
            events = self.engine.resolve_fox_swap(player, None)
            self._emit_events(events, broadcast=True)
            return

        state_info = self.engine.format_game_state(player)

        for attempt in range(MAX_RETRIES):
            if attempt:
                self.players[player].inject_retry_error(error)
            card, skipped, error = self.players[player].request_fox_swap(
                state_info if attempt == 0 else None, legal_cards, rs.decree_card
            )

            if skipped:
                events = self.engine.resolve_fox_swap(player, None)
                self._emit_events(events, broadcast=True)
                return
            if card is not None:
                events = self.engine.resolve_fox_swap(player, card)
                self._emit_events(events, broadcast=True)
                return

            self.log(f"  [ERROR] {self.players[player].player_name} fox_swap attempt {attempt+1}/{MAX_RETRIES}: {error}")

        self.log(f"  [FORFEIT] {self.players[player].player_name} failed fox_swap after {MAX_RETRIES} attempts.")
        self.forfeit_winner = 1 - player

    def _handle_woodcutter_discard(self, player: int):
        rs = self.engine.current_round
        legal_cards = self.engine.get_legal_plays_for_woodcutter_discard(player)
        drawn_card = rs.woodcutter_drawn_card

        if len(legal_cards) == 1:
            card = legal_cards[0]
            events = self.engine.resolve_woodcutter_discard(player, card)
            self.log(f"  (Auto-discard: {self.players[player].player_name} discards {card})")
            self._emit_events(events, broadcast=True)
            hand_after = format_hand(self.engine.current_round.hands[player])
            self.players[player].add_log(f"Your hand after Woodcutter: {hand_after}")
            return

        state_info = self.engine.format_game_state(player)

        for attempt in range(MAX_RETRIES):
            if attempt:
                self.players[player].inject_retry_error(error)
            card, error = self.players[player].request_woodcutter_discard(
                state_info if attempt == 0 else None, legal_cards, drawn_card
            )

            if card is not None:
                events = self.engine.resolve_woodcutter_discard(player, card)
                self._emit_events(events, broadcast=True)
                hand_after = format_hand(self.engine.current_round.hands[player])
                self.players[player].add_log(f"Your hand after Woodcutter: {hand_after}")
                return

            self.log(f"  [ERROR] {self.players[player].player_name} woodcutter attempt {attempt+1}/{MAX_RETRIES}: {error}")

        self.log(f"  [FORFEIT] {self.players[player].player_name} failed woodcutter_discard after {MAX_RETRIES} attempts.")
        self.forfeit_winner = 1 - player
