"""V1 baseline Elo ratings.

These are the output of running `python elo.py` on the archived v1 results
(git tag `v1`; raw data preserved locally under `backup/`, see README).

They are used as:
- the initial ratings for the v2 continuous matchmaking scheduler (main.py)
- the initial ratings when computing v2 Elo standings (elo.py)

Models not listed here (e.g. newly added models) start at DEFAULT_ELO.
"""

DEFAULT_ELO = 1500.0

V1_ELO = {
    "Claude-Fable-5": 1845.3,
    "Kimi-K3": 1584.2,
    "GLM-5.2-Max": 1583.0,
    "DS-V4-Pro-GA": 1516.0,  # only 1 game in v1; treat as near-default
    "DS-V4-Flash-GA": 1496.8,
    "GPT-5.6-Sol": 1471.5,
    "Gemini-3.5-Flash": 1469.4,
    "GPT-5.6-Terra": 1411.4,
    "DS-V4-Pro": 1329.2,
    "DS-V4-Flash": 1293.2,
}
