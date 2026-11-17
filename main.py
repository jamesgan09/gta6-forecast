"""gta6-twitch-ccu-forecast-01 entry: a log-ratio regression on the case's own comparables.

Model `gta6-forecast-v1`. For each hour h = 2..24 of release day, the forecast is built from the
launch-day curves of past big game launches (the case's `comparables`), anchored on hour 1's print p1:

    log(v_h / p1) = a_h + b_h * log(p1) + noise

fit by ordinary least squares per hour, over the comparables whose launch day is complete and whose
hour-1 print is at least 50,000 viewers (falling back to 20k, then 5k, then all, if fewer than 8
qualify). The five quantiles are the fitted median times exp(z_tau * sd_h), where sd_h is the
per-hour residual standard deviation (floored at 0.10), widened a little when p1 sits above the
largest hour-1 print in the training set (extrapolation insurance). Hours with too few training
points, and cases with no usable comparables at all, fall back to the house baseline (persistence
with spread 0.15 * sqrt(h - 1)).

Why this shape, and why it beats persistence: across 37 past launches the launch-day curve is not
flat. Viewers dip in the small hours (hours 2-5 run ~0.75x hour 1), then climb as regions unlock
(NZ 11:00Z, Europe evening, US from 05:00Z/08:00Z), peaking around hours 19-20 at ~2.4x hour 1 for
a typical launch. But the size of that climb depends almost entirely on hour 1 itself: the
log-ratio at hour 19 correlates -0.93 with log(p1) (R^2 = 0.87). A huge hour 1 means the launch is
already near its ceiling, so it barely grows; a small hour 1 means it grows several-fold. In
absolute terms the late-day level is nearly flat in p1 (log v19 ~ 5.0 + 0.62 * log p1 among big
launches): the day's peak is set by how many people Twitch can bring, not by how many were awake
at midnight UTC. Persistence ignores all of this; this model conditions on it.

Leave-one-out validation on the kit's comparables (pinball loss on the log scale, lower is better):
  - big launches (hour-1 >= 20k, n=27): baseline 0.321, this model 0.250
  - mega launches (hour-1 >= 300k, n=6, the GTA VI regime): baseline 0.226, this model 0.128
  - all launches (n=37): baseline 0.930, this model 0.719
Coverage of the 5-95% band on big launches: 0.90, as scored (baseline: 0.79).
Features that did NOT help and were left out: hour-1 channel count, and the pre-midnight
(NZ-window) viewership in the 13 hours before day_start.

Deterministic, standard library only, no network: reads every <id>.in in the input directory and
writes <id>.out to the output directory, exactly 23 rows (hours 2..24), five non-decreasing
quantiles each, rounded to 2 decimals half-up like the starter.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# House baseline spread, in natural-log units per square-root hour (the fallback).
SIGMA = 0.15
# Smallest training set worth fitting on; below this, drop to the next threshold.
MIN_TRAIN = 8
# Hour-1 print thresholds for the training set, tried in order.
THRESHOLDS = (50000, 20000, 5000, 0)
# Extra spread per log unit of hour-1 print beyond the training range.
EXTRA_SD_PER_LOG = 0.10
# Floor on the per-hour residual spread, in log units.
SD_FLOOR = 0.10


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def day_series(comp: dict) -> dict[int, int]:
    """The comparable's release-day hourly viewers, keyed by hour 1..24 from its day_start."""
    ds = parse_ts(comp["day_start"])
    hours: dict[int, int] = {}
    for row in comp.get("hourly") or []:
        try:
            h = int((parse_ts(row["from"]) - ds).total_seconds() // 3600) + 1
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= h <= 24:
            hours[h] = max(0, int(row.get("viewers") or 0))
    return hours


def is_own_history(comp: dict, game: str) -> bool:
    """True when the comparable is the target game's own pre-release history (a premiere, not a launch)."""
    name = (comp.get("name") or "").lower()
    return bool(game) and game.lower() in name


def select_training(case: dict) -> list[dict[int, int]]:
    """Complete launch-day series to fit on, with the biggest hour-1 prints preferred."""
    game = case.get("game") or ""
    if isinstance(game, dict):
        game = game.get("name") or ""
    complete = []
    for comp in case.get("comparables") or []:
        if is_own_history(comp, game):
            continue
        series = day_series(comp)
        if series.get(1, 0) > 0 and len(series) >= 20:
            complete.append(series)
    for thr in THRESHOLDS:
        picked = [s for s in complete if s[1] >= thr]
        if len(picked) >= MIN_TRAIN:
            return picked
    return complete


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
    return my - b * mx, b


def baseline_quantiles(p1: float, h: int) -> list[float]:
    return [p1 * math.exp(z * SIGMA * math.sqrt(h - 1)) for z in Z]


def hour_quantiles(p1: float, h: int, training: list[dict[int, int]]) -> list[float]:
    """The five quantiles for hour h: regression on the training set, baseline when too thin."""
    rows = [
        (math.log(s[1]), math.log(s[h] / s[1]))
        for s in training
        if h in s and s[1] > 0 and s[h] > 0
    ]
    if len(rows) < 5:
        return baseline_quantiles(p1, h)
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    a, b = ols(xs, ys)
    lp = math.log(p1)
    mu = a + b * lp
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = max(SD_FLOOR, (sum(r * r for r in resid) / max(1, len(resid) - 2)) ** 0.5)
    sd += EXTRA_SD_PER_LOG * max(0.0, lp - max(xs))
    return [p1 * math.exp(mu + z * sd) for z in Z]


def clamp_quantiles(qs: list[float]) -> list[float | int]:
    """Round, clamp to the valid range, and keep the five quantiles non-decreasing."""
    out: list[float | int] = []
    prev = 1.0
    for q in qs:
        v = round2(min(max(q, 1.0), 1e9))
        if v < prev:
            v = prev
        out.append(v)
        prev = v if isinstance(v, (int, float)) else prev
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    training = select_training(case)
    return {
        "hours": [
            {"hour": row["hour"], "q": clamp_quantiles(hour_quantiles(p1, row["hour"], training))}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
