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
  1    Claude-Fable-5             1749.1 [1656.3, 1843.0]      80
  2    GPT-5.6-Sol                1625.5 [1523.8, 1728.6]     140
  3    GLM-5.2-Max                1573.7 [1481.6, 1671.4]     140
  4    GPT-5.6-Terra              1485.2 [1391.2, 1592.2]     140
  5    Claude-Sonnet-5            1481.9 [1378.5, 1584.6]      80
  6    DS-V4-Pro-NonThinking      1452.6 [1353.2, 1543.7]     140
  7    DS-V4-Pro-Thinking         1402.2 [1291.5, 1500.8]     140
  8    DS-V4-Flash-Thinking       1371.8 [1263.7, 1464.3]     140
  9    DS-V4-Flash-NonThinking    1358.1 [1244.3, 1457.2]     140

  Head-to-Head (W-L):
                           Claude-Fable-5GPT-5.6-Sol   GLM-5.2-Max   GPT-5.6-Terra Claude-Sonnet-DS-V4-Pro-NonTDS-V4-Pro-ThinDS-V4-Flash-ThDS-V4-Flash-No
  Claude-Fable-5           --            6-4           8-2           10-0          10-0          9-1           9-1           9-1           6-4           
  GPT-5.6-Sol              4-6           --            10-10         13-7          6-4           18-2          14-6          15-5          18-2          
  GLM-5.2-Max              2-8           10-10         --            10-10         6-4           16-4          13-7          14-6          17-3          
  GPT-5.6-Terra            0-10          7-13          10-10         --            4-6           9-11          16-4          13-7          12-8          
  Claude-Sonnet-5          0-10          4-6           4-6           6-4           --            7-3           4-6           7-3           6-4           
  DS-V4-Pro-NonThinking    1-9           2-18          4-16          11-9          3-7           --            12-8          16-4          15-5          
  DS-V4-Pro-Thinking       1-9           6-14          7-13          4-16          6-4           8-12          --            9-11          12-8          
  DS-V4-Flash-Thinking     1-9           5-15          6-14          7-13          3-7           4-16          11-9          --            10-10         
  DS-V4-Flash-NonThinking  4-6           2-18          3-17          8-12          4-6           5-15          8-12          10-10         --            

  Token Usage (output tokens per game):
  Model                            Avg        Min        Max        Total
------------------------------------------------------------------------
  Claude-Fable-5                 77620      27075     181351      6209627
  GPT-5.6-Sol                    10114       5612      16144      1415907
  GLM-5.2-Max                    51930       8979     192349      7270236
  GPT-5.6-Terra                  16208       8485      29355      2269101
  Claude-Sonnet-5                44969      17417     105209      3597522
  DS-V4-Pro-NonThinking           9998       1816      16369      1399671
  DS-V4-Pro-Thinking             23164      11336      43744      3242998
  DS-V4-Flash-Thinking           18893       8702      32357      2645056
  DS-V4-Flash-NonThinking         9627       4202      16304      1347788
```