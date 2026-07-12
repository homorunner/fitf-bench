# Fox in the Forest LLM Benchmark

A benchmark that evaluates LLMs by having them play *The Fox in the Forest*, a two-player trick-taking card game, against each other in a round-robin tournament.

## Usage

```bash
python main.py              # Run tournament
python elo.py               # Compute Elo ratings
```

Configure models in `main.py` `MODELS` list. Supports parallel execution, deterministic seeds, and auto-resume.

## Results

```
  ELO RATINGS:
  #    Model                         Elo           95% CI   Games
------------------------------------------------------------------------
  1    GPT-5.6-Sol                1660.7 [1557.3, 1766.2]     120
  2    GLM-5.2-Max                1612.6 [1519.0, 1714.9]     120
  3    GPT-5.6-Terra              1539.2 [1437.1, 1647.1]     120
  4    DS-V4-Pro-NonThinking      1498.6 [1408.5, 1587.2]     120
  5    DS-V4-Pro-Thinking         1422.8 [1321.1, 1523.2]     120
  6    DS-V4-Flash-Thinking       1401.7 [1308.4, 1507.9]     120
  7    DS-V4-Flash-NonThinking    1364.4 [1261.9, 1453.8]     120

  Token Usage (output tokens per game):
  Model                            Avg        Min        Max        Total
------------------------------------------------------------------------
  GPT-5.6-Sol                     9958       5612      16144      1194931
  GLM-5.2-Max                    57290       8979     192349      6874777
  GPT-5.6-Terra                  16414       8485      29355      1969731
  DS-V4-Pro-NonThinking           9965       1816      14995      1195750
  DS-V4-Pro-Thinking             23447      11336      43744      2813596
  DS-V4-Flash-Thinking           18920       8702      32357      2270430
  DS-V4-Flash-NonThinking         9602       4202      16304      1152199
```