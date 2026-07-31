# Two-Player Game LLM Benchmark

A benchmark that evaluates LLMs in round-robin tournaments across two-player text games.

## Usage

```bash
python main.py              # Run tournament
python elo.py               # Compute Elo ratings
```

Configure models in `main.py` `MODELS` list. Supports parallel execution, deterministic seeds, and auto-resume.

## Adding a Game

New games use the shared `TwoPlayerGameRunner` and `LLMPlayer.request_action()` interfaces:

1. Add a package under `fitf_bench/games/` containing its runner and `RULES.md`.
2. Implement a runner subclass that maintains the game log and current state and validates tool-call actions.
3. Return results with `build_result(winner, reason, **game_data)`.
4. Register a `GameDefinition` in `fitf_bench/game_registry.py`.

The tournament runner, model setup, API logging, retries, and result metadata are shared across games.
Use `python main.py --game <game-id>` and `python elo.py --game <game-id>` to run and analyze a registered game.

## Results

### Elo Ratings

| Rank | Model | Elo | 95% CI | Games | WinRate |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Claude-Fable-5 | 1844.0 | [1761.0, 1912.3] | 76 | 93.4% |
| 2 | GLM-5.2-Max | 1578.5 | [1492.6, 1680.4] | 97 | 66.0% |
| 3 | Kimi-K3 | 1578.0 | [1478.5, 1676.9] | 55 | 61.8% |
| 4 | DS-V4-Flash-GA | 1523.0 | [1421.9, 1622.7] | 55 | 52.7% |
| 5 | GPT-5.6-Sol | 1479.3 | [1382.0, 1574.7] | 72 | 47.2% |
| 6 | Gemini-3.5-Flash | 1454.8 | [1357.1, 1554.3] | 74 | 44.6% |
| 7 | GPT-5.6-Terra | 1412.3 | [1312.3, 1513.3] | 53 | 37.7% |
| 8 | DS-V4-Pro | 1339.2 | [1239.0, 1440.4] | 98 | 29.6% |
| 9 | DS-V4-Flash | 1290.9 | [1202.0, 1380.2] | 92 | 23.9% |

### Head-to-Head (W-L)

| Model | Claude-Fable-5 | GLM-5.2-Max | Kimi-K3 | DS-V4-Flash-GA | GPT-5.6-Sol | Gemini-3.5-Flash | GPT-5.6-Terra | DS-V4-Pro | DS-V4-Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude-Fable-5 | -- | 9-1 | 8-1 | 8-1 | 10-0 | 10-0 | 6-2 | 10-0 | 10-0 |
| GLM-5.2-Max | 1-9 | -- | 4-6 | 4-5 | 5-5 | 9-1 | 8-0 | 16-4 | 17-3 |
| Kimi-K3 | 1-8 | 6-4 | -- | 1-0 | 3-3 | 3-2 | 2-3 | 9-1 | 9-0 |
| DS-V4-Flash-GA | 1-8 | 5-4 | 0-1 | -- | 6-4 | 4-6 | 0-0 | 8-2 | 5-1 |
| GPT-5.6-Sol | 0-10 | 5-5 | 3-3 | 4-6 | -- | 4-6 | 6-2 | 6-4 | 6-2 |
| Gemini-3.5-Flash | 0-10 | 1-9 | 2-3 | 6-4 | 6-4 | -- | 4-4 | 6-4 | 8-3 |
| GPT-5.6-Terra | 2-6 | 0-8 | 3-2 | 0-0 | 2-6 | 4-4 | -- | 3-5 | 6-2 |
| DS-V4-Pro | 0-10 | 4-16 | 1-9 | 2-8 | 4-6 | 4-6 | 5-3 | -- | 9-11 |
| DS-V4-Flash | 0-10 | 3-17 | 0-9 | 1-5 | 2-6 | 3-8 | 2-6 | 11-9 | -- |

### Token Usage

Output tokens per game:

| Model | Average | Min | Max | Total |
| --- | ---: | ---: | ---: | ---: |
| Claude-Fable-5 | 224,816 | 52,309 | 569,900 | 17,086,007 |
| GLM-5.2-Max | 313,999 | 88,986 | 739,191 | 30,457,858 |
| Kimi-K3 | 617,610 | 163,472 | 1,028,096 | 33,968,534 |
| DS-V4-Flash-GA | 1,148,119 | 67,562 | 2,067,844 | 63,146,532 |
| GPT-5.6-Sol | 39,993 | 14,504 | 107,669 | 2,879,476 |
| Gemini-3.5-Flash | 143,616 | 33,786 | 391,466 | 10,627,592 |
| GPT-5.6-Terra | 52,112 | 12,927 | 122,080 | 2,761,959 |
| DS-V4-Pro | 167,863 | 61,898 | 335,779 | 16,450,577 |
| DS-V4-Flash | 143,938 | 47,732 | 286,750 | 13,242,335 |
