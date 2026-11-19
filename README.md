# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/leosweet/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/leosweet/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26 00:00 UTC, as five quantiles (5% / 25% / 50% / 75% / 95%) per trophy. A logit-trajectory + denominator-growth model: earners and players tracked are fit separately in log time and re-rated at the target, with tier shrinkage and a horizon-widening spread that carries Exophase's refresh risk. Early tracked players are hardcore launch-night players, so early rates are inflated; when Exophase refreshes, the tracked base can jump faster than earners (638 → 3,868 players in the first refresh), so rates can fall even as more people earn the trophy — the model forecasts the counts, not just the rate.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case to the output directory. Python 3.13, standard library only, no network. Deterministic: the same input always writes byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=zhhxa80s&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
