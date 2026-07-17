"""Game engine for Fox in the Forest."""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from fitf_bench.cards import Card, Suit, create_deck, format_hand


@dataclass
class TrickResult:
    trick_number: int
    leader: int  # 0 or 1
    leader_card: Card
    follower: int  # 0 or 1
    follower_card: Card
    winner: int  # 0 or 1
    bonus_points: Dict[int, int] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)


@dataclass
class RoundState:
    round_number: int
    dealer: int  # 0 or 1
    hands: List[List[Card]]  # hands[0], hands[1]
    draw_deck: List[Card]
    decree_card: Card
    trump_suit: Suit
    tricks_won: List[int]  # tricks_won[0], tricks_won[1]
    bonus_points: List[int]  # bonus from Treasure cards
    current_leader: int  # who leads the current trick
    trick_number: int  # 1..13
    trick_results: List[TrickResult] = field(default_factory=list)

    pending_fox_swap: bool = False
    fox_player: int = -1

    pending_woodcutter: bool = False
    woodcutter_player: int = -1
    woodcutter_drawn_card: Optional[Card] = None

    monarch_constraint: bool = False


class GameEngine:
    def __init__(self, seed: Optional[int] = None):
        self.scores: List[int] = [0, 0]
        self.round_number: int = 0
        self._rng = random.Random(seed)
        self.dealer: int = self._rng.randint(0, 1)
        self.current_round: Optional[RoundState] = None
        self.game_over: bool = False
        self.winner: Optional[int] = None

    def start_new_round(self) -> RoundState:
        """Deal cards and set up a new round."""
        self.round_number += 1
        deck = create_deck()
        self._rng.shuffle(deck)

        hands = [sorted(deck[:13]), sorted(deck[13:26])]
        draw_deck = deck[26:]
        decree_card = draw_deck.pop(0)
        trump_suit = decree_card.suit

        first_leader = 1 - self.dealer

        self.current_round = RoundState(
            round_number=self.round_number,
            dealer=self.dealer,
            hands=hands,
            draw_deck=draw_deck,
            decree_card=decree_card,
            trump_suit=trump_suit,
            tricks_won=[0, 0],
            bonus_points=[0, 0],
            current_leader=first_leader,
            trick_number=1,
        )
        return self.current_round

    def get_legal_plays(self, player: int, lead_card: Optional[Card] = None) -> List[Card]:
        """Get legal cards a player can play.
        
        If lead_card is None, the player is leading (any card is legal).
        If lead_card is set, the player must follow suit if possible.
        """
        rs = self.current_round
        hand = rs.hands[player]

        if lead_card is None:
            return hand.copy()

        lead_suit = lead_card.suit

        if rs.monarch_constraint and lead_card.rank == 11:
            same_suit_cards = [c for c in hand if c.suit == lead_suit]
            if same_suit_cards:
                options = []
                one_card = Card(suit=lead_suit, rank=1)
                if one_card in same_suit_cards:
                    options.append(one_card)
                highest = max(same_suit_cards, key=lambda c: c.rank)
                if highest not in options:
                    options.append(highest)
                return options
            return hand.copy()

        same_suit_cards = [c for c in hand if c.suit == lead_suit]
        return same_suit_cards or hand.copy()

    def get_legal_plays_for_fox_swap(self, player: int) -> List[Card]:
        """Get legal cards to swap with decree card (any card in hand, or None to skip)."""
        return self.current_round.hands[player].copy()

    def get_legal_plays_for_woodcutter_discard(self, player: int) -> List[Card]:
        """Get legal cards to discard after drawing (any card in hand)."""
        return self.current_round.hands[player].copy()

    def play_lead_card(self, player: int, card: Card) -> List[str]:
        """Leader plays a card. Returns list of events/messages.
        
        May trigger Fox(3) or Woodcutter(5) abilities.
        """
        rs = self.current_round
        assert player == rs.current_leader
        assert card in rs.hands[player]

        rs.hands[player].remove(card)
        rs._lead_card = card
        rs._lead_player = player
        events = []
        if card.rank == 11:
            rs.monarch_constraint = True
            events.append(f"Player {player+1} leads with Monarch (11). "
                         f"Opponent must play the 1 or highest card of {card.suit.value} if they have any.")

        events.extend(self._activate_card_ability(player, card))
        return events

    def _activate_card_ability(self, player: int, card: Card) -> List[str]:
        rs = self.current_round
        events = []
        if card.rank == 3:
            rs.pending_fox_swap = True
            rs.fox_player = player
            events.append(f"Player {player+1} plays Fox (3). "
                         f"They may exchange the decree card with a card from their hand.")

        if card.rank == 5:
            if rs.draw_deck:
                drawn = rs.draw_deck.pop(0)
                rs.hands[player].append(drawn)
                rs.hands[player].sort()
                rs.pending_woodcutter = True
                rs.woodcutter_player = player
                rs.woodcutter_drawn_card = drawn
                events.append(f"Player {player+1} plays Woodcutter (5). "
                             f"They draw 1 card and must discard 1 card.")
            else:
                events.append(f"Player {player+1} plays Woodcutter (5), "
                             f"but the draw deck is empty.")

        return events

    def play_follow_card(self, player: int, card: Card) -> List[str]:
        """Follower plays a card. Returns list of events/messages.
        
        May trigger Fox(3) or Woodcutter(5) abilities.
        """
        rs = self.current_round
        assert player != rs.current_leader
        assert card in rs.hands[player]

        legal_cards = self.get_legal_plays(player, rs._lead_card)
        if card not in legal_cards:
            raise ValueError(f"Illegal follow play: {card}")

        rs.hands[player].remove(card)
        rs._follow_card = card
        rs._follow_player = player
        rs.monarch_constraint = False
        return self._activate_card_ability(player, card)

    def resolve_fox_swap(self, player: int, swap_card: Optional[Card]) -> List[str]:
        """Resolve Fox(3) ability: swap decree card with a hand card, or None to skip."""
        rs = self.current_round
        events = []

        if not rs.pending_fox_swap:
            raise ValueError("No Fox swap is pending")
        if player != rs.fox_player:
            raise ValueError("Only the player who played the Fox may resolve its ability")
        if swap_card is not None and swap_card not in rs.hands[player]:
            raise ValueError("Fox swap card is not in the player's hand")

        if swap_card is None:
            events.append(f"Player {player+1} chooses not to swap the decree card.")
        else:
            old_decree = rs.decree_card
            rs.hands[player].remove(swap_card)
            rs.hands[player].append(old_decree)
            rs.hands[player].sort()
            rs.decree_card = swap_card
            rs.trump_suit = swap_card.suit
            events.append(f"Player {player+1} swaps the decree card. "
                         f"Old decree: {old_decree}, New decree: {swap_card}. "
                         f"Trump suit is now {rs.trump_suit.value}.")

        rs.pending_fox_swap = False
        rs.fox_player = -1

        return events

    def resolve_woodcutter_discard(self, player: int, discard_card: Card) -> List[str]:
        """Resolve Woodcutter(5) ability: discard a card to bottom of deck."""
        rs = self.current_round
        events = []
        if not rs.pending_woodcutter:
            raise ValueError("No Woodcutter discard is pending")
        if player != rs.woodcutter_player:
            raise ValueError("Only the player who played the Woodcutter may discard")
        if discard_card not in rs.hands[player]:
            raise ValueError("Woodcutter discard card is not in the player's hand")

        rs.hands[player].remove(discard_card)
        rs.draw_deck.append(discard_card)
        rs.pending_woodcutter = False
        rs.woodcutter_player = -1
        rs.woodcutter_drawn_card = None
        events.append(f"Player {player+1} discards a card to the bottom of the draw deck.")
        return events

    def resolve_trick(self) -> TrickResult:
        """After both cards are played and abilities resolved, determine trick winner."""
        rs = self.current_round
        lead_card = rs._lead_card
        follow_card = rs._follow_card
        leader = rs._lead_player
        follower = rs._follow_player
        trump = rs.trump_suit
        lead_suit = lead_card.suit

        events = []

        one_nine = (lead_card.rank == 9) != (follow_card.rank == 9)

        lead_is_trump = lead_card.suit == trump
        follow_is_trump = follow_card.suit == trump

        if one_nine:
            if lead_card.rank == 9 and lead_card.suit != trump:
                lead_is_trump = True
                events.append(f"Witch (9 of {lead_card.suit.value}) is treated as trump!")
            if follow_card.rank == 9 and follow_card.suit != trump:
                follow_is_trump = True
                events.append(f"Witch (9 of {follow_card.suit.value}) is treated as trump!")

        if lead_is_trump and follow_is_trump:
            if lead_card.rank > follow_card.rank:
                trick_winner = leader
            else:
                trick_winner = follower
        elif lead_is_trump:
            trick_winner = leader
        elif follow_is_trump:
            trick_winner = follower
        else:
            if follow_card.suit == lead_suit:
                if follow_card.rank > lead_card.rank:
                    trick_winner = follower
                else:
                    trick_winner = leader
            else:
                trick_winner = leader

        rs.tricks_won[trick_winner] += 1

        bonus = {}
        treasure_count = sum(1 for c in [lead_card, follow_card] if c.rank == 7)
        if treasure_count > 0:
            bonus[trick_winner] = treasure_count
            rs.bonus_points[trick_winner] += treasure_count
            self.scores[trick_winner] += treasure_count
            events.append(f"Treasure bonus: Player {trick_winner+1} gains {treasure_count} point(s)!")

        swan_override = None
        lead_has_swan = lead_card.rank == 1
        follow_has_swan = follow_card.rank == 1

        if lead_has_swan and follow_has_swan:
            swan_override = 1 - trick_winner
            events.append("Both players played Swan (1). The loser leads the next trick.")
        elif lead_has_swan and trick_winner == follower:
            swan_override = leader
            events.append(f"Player {leader+1}'s Swan (1) lost the trick. Player {leader+1} leads next.")
        elif follow_has_swan and trick_winner == leader:
            swan_override = follower
            events.append(f"Player {follower+1}'s Swan (1) lost the trick. Player {follower+1} leads next.")

        next_leader = swan_override if swan_override is not None else trick_winner

        result = TrickResult(
            trick_number=rs.trick_number,
            leader=leader,
            leader_card=lead_card,
            follower=follower,
            follower_card=follow_card,
            winner=trick_winner,
            bonus_points=bonus,
            events=events,
        )
        rs.trick_results.append(result)

        rs.trick_number += 1
        rs.current_leader = next_leader

        del rs._lead_card
        del rs._follow_card
        del rs._lead_player
        del rs._follow_player

        return result

    def score_round(self) -> Tuple[int, int, Dict]:
        """Score the round based on tricks won. Returns (p0_score, p1_score, details)."""
        rs = self.current_round

        def tricks_to_points(tricks: int) -> int:
            if tricks <= 3:
                return 6  # Humble
            elif tricks == 4:
                return 1
            elif tricks == 5:
                return 2
            elif tricks == 6:
                return 3
            elif tricks <= 9:
                return 6  # Victorious
            else:
                return 0  # Greedy (10-13)

        p0_base = tricks_to_points(rs.tricks_won[0])
        p1_base = tricks_to_points(rs.tricks_won[1])

        p0_total = p0_base + rs.bonus_points[0]
        p1_total = p1_base + rs.bonus_points[1]

        self.scores[0] += p0_base
        self.scores[1] += p1_base

        details = {
            "tricks_won": [rs.tricks_won[0], rs.tricks_won[1]],
            "base_points": [p0_base, p1_base],
            "bonus_points": [rs.bonus_points[0], rs.bonus_points[1]],
            "round_points": [p0_total, p1_total],
            "total_scores": [self.scores[0], self.scores[1]],
        }

        if self.scores[0] >= 35 or self.scores[1] >= 35:
            self.game_over = True
            if self.scores[0] > self.scores[1]:
                self.winner = 0
            elif self.scores[1] > self.scores[0]:
                self.winner = 1
            else:
                if p0_base > p1_base:
                    self.winner = 0
                else:
                    self.winner = 1

        self.dealer = 1 - self.dealer

        return p0_total, p1_total, details

    def is_round_over(self) -> bool:
        """Check if the current round is over (all 13 tricks played)."""
        return self.current_round.trick_number > 13

    def format_trick_result(self, result: TrickResult) -> str:
        """Format a trick result as human-readable text."""
        lines = []
        lines.append(f"--- Trick {result.trick_number} ---")
        lines.append(f"  Player {result.leader+1} leads: {result.leader_card}")
        lines.append(f"  Player {result.follower+1} follows: {result.follower_card}")
        for event in result.events:
            lines.append(f"  * {event}")
        lines.append(f"  Winner: Player {result.winner+1}")
        return "\n".join(lines)

    def format_game_state(self, player: int) -> str:
        """Format current game state visible to a player."""
        rs = self.current_round
        lines = []
        lines.append(f"=== Current State (Your perspective: Player {player+1}) ===")
        lines.append(f"Round: {rs.round_number} | Trick: {rs.trick_number}/13")
        lines.append(f"Trump suit: {rs.trump_suit.value} (Decree card: {rs.decree_card})")
        lines.append(f"Tricks won - You: {rs.tricks_won[player]}, Opponent: {rs.tricks_won[1-player]}")
        lines.append(f"Scores - You: {self.scores[player]}, Opponent: {self.scores[1-player]}")
        if rs.current_leader == player:
            lines.append(f"You are leading this trick.")
        else:
            lines.append(f"You are following this trick.")
        lines.append(f"Your hand: {format_hand(rs.hands[player])}")
        return "\n".join(lines)
