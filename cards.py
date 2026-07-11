"""Card and deck system for Fox in the Forest."""

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Suit(Enum):
    """Three suits in Fox in the Forest."""
    BELLS = "Bells"
    KEYS = "Keys"
    MOONS = "Moons"


SUIT_SYMBOLS = {
    Suit.BELLS: "🔔",
    Suit.KEYS: "🔑",
    Suit.MOONS: "🌙",
}


@dataclass(frozen=True)
class Card:
    """A single card with suit and rank (1-11)."""
    suit: Suit
    rank: int

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit.value}"

    def short_str(self) -> str:
        return f"{self.suit.value[0]}{self.rank}"

    def __lt__(self, other: "Card") -> bool:
        if self.suit != other.suit:
            return self.suit.value < other.suit.value
        return self.rank < other.rank


ABILITY_NAMES = {
    1: "Swan",
    3: "Fox",
    5: "Woodcutter",
    7: "Treasure",
    9: "Witch",
    11: "Monarch",
}


def create_deck() -> List[Card]:
    """Create a full 33-card deck (3 suits x 11 ranks)."""
    deck = []
    for suit in Suit:
        for rank in range(1, 12):
            deck.append(Card(suit=suit, rank=rank))
    return deck


def shuffle_deck(deck: List[Card]) -> List[Card]:
    """Return a shuffled copy of the deck."""
    shuffled = deck.copy()
    random.shuffle(shuffled)
    return shuffled


def card_from_str(s: str) -> Optional[Card]:
    """Parse a card from string like 'B3', 'K11', 'M1', or '3 of Bells'."""
    s = s.strip()
    # Try short format: first char is suit, rest is rank
    if len(s) >= 2 and s[0] in ('B', 'K', 'M'):
        suit_map = {'B': Suit.BELLS, 'K': Suit.KEYS, 'M': Suit.MOONS}
        try:
            rank = int(s[1:])
            if 1 <= rank <= 11:
                return Card(suit=suit_map[s[0]], rank=rank)
        except ValueError:
            pass
    # Try long format: "rank of suit"
    parts = s.lower().split(" of ")
    if len(parts) == 2:
        try:
            rank = int(parts[0])
            for suit in Suit:
                if suit.value.lower() == parts[1]:
                    if 1 <= rank <= 11:
                        return Card(suit=suit, rank=rank)
        except ValueError:
            pass
    return None


def format_hand(hand: List[Card]) -> str:
    """Format a hand of cards for display."""
    sorted_hand = sorted(hand)
    return ", ".join(str(c) for c in sorted_hand)
