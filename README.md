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
| 1 | Claude-Fable-5 | 1823.2 | [1751.4, 1885.7] | 63 | 93.7% |
| 2 | GLM-5.2-Max | 1592.4 | [1503.1, 1688.2] | 84 | 67.9% |
| 3 | Kimi-K3 | 1570.9 | [1471.4, 1662.4] | 52 | 59.6% |
| 4 | GPT-5.6-Sol | 1486.0 | [1382.7, 1582.5] | 54 | 48.1% |
| 5 | Gemini-3.5-Flash | 1450.4 | [1351.4, 1541.7] | 56 | 44.6% |
| 6 | GPT-5.6-Terra | 1412.6 | [1322.3, 1516.1] | 53 | 37.7% |
| 7 | DS-V4-Pro | 1358.0 | [1264.8, 1456.0] | 82 | 31.7% |
| 8 | DS-V4-Flash | 1306.5 | [1215.9, 1402.4] | 86 | 24.4% |

### Head-to-Head (W-L)

| Model | Claude-Fable-5 | GLM-5.2-Max | Kimi-K3 | GPT-5.6-Sol | Gemini-3.5-Flash | GPT-5.6-Terra | DS-V4-Pro | DS-V4-Flash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude-Fable-5 | -- | 9-1 | 8-1 | 8-0 | 8-0 | 6-2 | 10-0 | 10-0 |
| GLM-5.2-Max | 1-9 | -- | 4-6 | 4-4 | 7-1 | 8-0 | 16-4 | 17-3 |
| Kimi-K3 | 1-8 | 6-4 | -- | 3-3 | 3-2 | 2-3 | 7-1 | 9-0 |
| GPT-5.6-Sol | 0-8 | 4-4 | 3-3 | -- | 3-5 | 6-2 | 4-4 | 6-2 |
| Gemini-3.5-Flash | 0-8 | 1-7 | 2-3 | 5-3 | -- | 4-4 | 5-3 | 8-3 |
| GPT-5.6-Terra | 2-6 | 0-8 | 3-2 | 2-6 | 4-4 | -- | 3-5 | 6-2 |
| DS-V4-Pro | 0-10 | 4-16 | 1-7 | 4-4 | 3-5 | 5-3 | -- | 9-11 |
| DS-V4-Flash | 0-10 | 3-17 | 0-9 | 2-6 | 3-8 | 2-6 | 11-9 | -- |

### Token Usage

Output tokens per game:

| Model | Average | Min | Max | Total |
| --- | ---: | ---: | ---: | ---: |
| Claude-Fable-5 | 227,352 | 52,309 | 532,454 | 14,323,172 |
| GLM-5.2-Max | 314,596 | 106,604 | 739,191 | 26,426,041 |
| Kimi-K3 | 638,536 | 203,181 | 1,028,096 | 33,203,850 |
| GPT-5.6-Sol | 35,338 | 14,504 | 107,669 | 1,908,250 |
| Gemini-3.5-Flash | 136,475 | 33,786 | 391,466 | 7,642,585 |
| GPT-5.6-Terra | 52,112 | 12,927 | 122,080 | 2,761,959 |
| DS-V4-Pro | 163,580 | 61,898 | 335,779 | 13,413,569 |
| DS-V4-Flash | 144,205 | 57,984 | 286,750 | 12,401,657 |
