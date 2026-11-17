# GTA VI Twitch CCU Forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jamesgan09/gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jamesgan09/gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

A program that forecasts, hour by hour, how many people watch Grand Theft Auto VI live on
Twitch over the 24 UTC hours of its release day (2026-11-19), from everything recorded up to the
end of its first hour — as five quantiles per hour (5%, 25%, 50%, 75%, 95%).

Built for [make.fun's GTA VI Twitch CCU forecast bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
($250 in USDC, entries close 2026-11-19 01:00 UTC).

## The model

`main.py` (Python 3.13, standard library only, deterministic, offline) reads one case
`/input/<id>.in` and writes `/output/<id>.out` with 23 rows (hours 2–24).

For each hour h it fits, on the case's own comparables (past big game launches), a regression in
log space anchored on hour 1's print `p1`:

```
log(v_h / p1) = a_h + b_h * log(p1) + noise
```

The training set is the comparables with a complete launch day and an hour-1 print of at least
50,000 viewers (falling back to 20k, then 5k, then all, if fewer than 8 qualify). The target
game's own pre-release history (its premiere) is excluded — it is not a launch. Quantiles are the
fitted median times `exp(z_tau * sd_h)`, where `sd_h` is the per-hour residual spread, widened a
little when `p1` sits above the training range. Hours with too few training points fall back to
the house baseline (persistence, spread `0.15 * sqrt(h-1)`).

The key fact the model exploits: the launch-day curve is not flat, and its shape depends almost
entirely on hour 1 itself. Viewers dip in the small hours (~0.75x hour 1), then climb as regions
unlock, peaking around hours 19–20. But the climb's size correlates −0.93 with `log(p1)` (R² =
0.87 at hour 19): a huge hour 1 means the launch is already near its ceiling, so it barely
grows; a small hour 1 means it grows several-fold. In absolute terms the late-day level is nearly
flat in `p1` — the day's peak is set by how many people Twitch can bring, not by how many were
awake at midnight UTC.

## Validation

Leave-one-out on the kit's 37 usable comparables (pinball loss on the log scale, lower is better):

| subset | house baseline | this model |
|---|---|---|
| big launches, hour-1 ≥ 20k (n=27) | 0.321 | 0.250 |
| mega launches, hour-1 ≥ 300k (n=6, the GTA VI regime) | 0.226 | 0.128 |
| all launches (n=37) | 0.930 | 0.719 |

Coverage of the 5–95% band on big launches: 0.90, as scored (baseline: 0.79). Features tested and
left out: hour-1 channel count, and pre-midnight (NZ-window) viewership.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in well under a second per case, far inside the 8-second limit.
