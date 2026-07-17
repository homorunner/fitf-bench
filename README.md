# Fox in the Forest LLM Benchmark

A benchmark that evaluates LLMs by having them play *The Fox in the Forest*, a two-player trick-taking card game, against each other in a round-robin tournament.

## Usage

```bash
python main.py              # Run tournament
python elo.py               # Compute Elo ratings
```

Configure models in `main.py` `MODELS` list. Supports parallel execution, deterministic seeds, and auto-resume.

## Results

### Elo Ratings

| Rank | Model | Elo | 95% CI | Games |
| ---: | --- | ---: | ---: | ---: |
| 1 | Claude-Fable-5 | 1748.5 | [1680.9, 1803.8] | 30 |
| 2 | GLM-5.2-Max | 1619.3 | [1529.8, 1702.9] | 50 |
| 3 | GPT-5.6-Sol | 1480.1 | [1390.2, 1568.1] | 30 |
| 4 | Gemini-3.5-Flash | 1448.4 | [1359.0, 1540.2] | 30 |
| 5 | DS-V4-Pro | 1437.6 | [1344.8, 1532.1] | 50 |
| 6 | DS-V4-Flash | 1403.8 | [1312.0, 1500.9] | 50 |
| 7 | GPT-5.6-Terra | 1362.4 | [1275.2, 1449.3] | 30 |

### Head-to-Head (W-L)

| Model | Claude-Fable-5 | GLM-5.2-Max | GPT-5.6-Sol | Gemini-3.5-Flash | DS-V4-Pro | DS-V4-Flash | GPT-5.6-Terra |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Claude-Fable-5 | -- | 5-0 | 5-0 | 5-0 | 5-0 | 5-0 | 4-1 |
| GLM-5.2-Max | 0-5 | -- | 3-2 | 4-1 | 11-4 | 13-2 | 5-0 |
| GPT-5.6-Sol | 0-5 | 2-3 | -- | 2-3 | 3-2 | 3-2 | 4-1 |
| Gemini-3.5-Flash | 0-5 | 1-4 | 3-2 | -- | 2-3 | 2-3 | 4-1 |
| DS-V4-Pro | 0-5 | 4-11 | 2-3 | 3-2 | -- | 7-8 | 4-1 |
| DS-V4-Flash | 0-5 | 2-13 | 2-3 | 3-2 | 8-7 | -- | 2-3 |
| GPT-5.6-Terra | 1-4 | 0-5 | 1-4 | 1-4 | 1-4 | 3-2 | -- |

### Token Usage

Output tokens per game:

| Model | Average | Min | Max | Total |
| --- | ---: | ---: | ---: | ---: |
| Claude-Fable-5 | 321,258 | 160,053 | 532,454 | 9,637,739 |
| GLM-5.2-Max | 318,411 | 173,581 | 538,039 | 15,920,541 |
| GPT-5.6-Sol | 22,841 | 14,504 | 29,978 | 685,232 |
| Gemini-3.5-Flash | 103,368 | 64,499 | 153,165 | 3,101,031 |
| DS-V4-Pro | 153,248 | 100,445 | 198,622 | 7,662,391 |
| DS-V4-Flash | 128,320 | 87,912 | 190,453 | 6,416,021 |
| GPT-5.6-Terra | 53,479 | 36,358 | 67,277 | 1,604,368 |
