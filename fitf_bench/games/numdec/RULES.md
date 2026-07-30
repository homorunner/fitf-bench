# Number Decomposition

Number Decomposition is a two-player, turn-based deduction game. Each player
has a secret number and tries to reduce the opponent's number to a winning
value. A match is best of three rounds: the first player to win two rounds wins
the match.

## Round Setup

At the start of each round, each player privately chooses an integer from 2 to
100, inclusive. Your number and all later changes to it are private. The first
player is determined before the match and acts first in every round.

Each player receives one lie opportunity per round. An unused lie does not
carry over to the next round.

## Turn

On your turn, choose exactly one operation against the opponent's current
number. Turns alternate after every attempt, whether it succeeds or fails.

Each round has a strict limit. After both players have completed 15 turns, the
first player takes one final, 16th turn. If that action does not win the round,
the second player immediately wins the round. The second player therefore takes
at most 15 turns in a round.

### Subtraction

Choose an integer from 1 to 5. If the opponent's current number is greater than
or equal to that integer, the operation succeeds and the integer is subtracted.
Otherwise, it fails and the number does not change.

If a successful subtraction reduces the opponent's number to 0, you immediately
win the round.

### Division

Choose any positive integer. If the opponent's current number is divisible by
that integer, the operation succeeds and the number is divided by it.
Otherwise, it fails and the number does not change. Division by 1 is allowed.

If a successful division reduces the opponent's number to 1, you immediately
win the round.

## Lying

When an opponent attacks your number, you may secretly spend your one lie for
the round. If you lie:

- The announced result is the opposite of the truthful result: success is
  announced as failure, or failure is announced as success.
- Your number does not change, even if the operation would normally succeed or
  win the round.
- The opponent is not told that you lied or that your lie opportunity was used.

If you do not lie, the truthful result is announced and a successful operation
changes your number normally.

## Information

You can see your own current number and whether your lie is still available.
You cannot see the opponent's number or whether the opponent has used their
lie. The public game log contains all declared operations and announced
success/failure results.
