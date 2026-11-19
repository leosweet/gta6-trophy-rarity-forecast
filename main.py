"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts each PlayStation 5 trophy's earn rate on Exophase at the case's
target, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95), in percent.

The model
----------
Exophase prints a rate = earners / players_tracked. Early tracked players are
hardcore launch-night players, so early rates are inflated, and when Exophase
refreshes the page the tracked base can jump (638 -> 3,868 players in the first
refresh) while earners grow more slowly, so rates can FALL on a refresh even
though more people earned the trophy. A model that only extrapolates the rate
itself misses that. This program therefore models the two counts separately,
in log time, and re-rates at the target:

  * players:  ln(players) fit by least squares against x = ln(1 + hours since
    2026-11-19T00:00:00Z), over every read of the case.
  * earners:  for each trophy, earners ~= rate/100 * players_tracked, and
    ln(earners) fit by least squares against the same x. A trophy's slope is
    shrunk toward its tier's median slope (platinum / gold / silver / bronze),
    weight n/(n+2) on its own fit, so trophies with few reads borrow strength
    from trophies of the same difficulty instead of chasing noise.
  * median rate at the target = 100 * exp(ln earners) / exp(ln players),
    clamped to [0.01, 99.99].

The spread is honest about what the history cannot see. It combines, in
quadrature on the logit scale:

  * a random-walk term, 0.5 * sqrt(horizon in days) (the house baseline's);
  * a refresh step of 0.45 logit units: Exophase refreshes on its own
    schedule, and one refresh can move a rate by more than a day of drift;
    0.85 when the whole history is one frozen page (players and rates
    unchanged across every read), because then the next refresh's dilution
    is entirely unobserved and pending: the first refresh moved the tracked
    base 638 -> 3,868 players, a dilution of up to ~1.8 logit units if
    earners had stood still, so a frozen page warrants a spread near
    1 logit unit even on a 90-minute horizon;
  * the trophy's own fit residual (0 when its history is flat, up to 0.8),
    or 0.25 when it has too few reads to fit.

Quantiles are logit(median) + z_tau * sigma, converted back to percent and
rounded to 4 decimals, half up. Everything is pure arithmetic on the case
file: no randomness, no wall clock, no network, standard library only, so
two runs on the same input write byte-identical output.

How the grader runs it: `python3 /bot/main.py`, reading every <id>.in in
$MAKEFUN_INPUT_DIR (else /input/) and writing <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output/): {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
one row per trophy, quantiles non-decreasing, each 0..100.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry ships main.py only.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Log-time origin: the house curve baseline's epoch. Rates and counts grow
# fastest right after launch and slow down, so log time, not clock time, is
# the axis the trends are fit on.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Spread pieces, in logit units (see the module docstring).
SIGMA_RW_PER_SQRT_DAY = 0.5
SIGMA_STEP = 0.45
SIGMA_STEP_FROZEN = 0.85
SIGMA_FIT_MAX = 0.8
SIGMA_FIT_DEFAULT = 0.25
SIGMA_MAX = 2.5

# A trophy with fewer earners than this in a read is at the mercy of the
# page's 2-decimal rounding; floor the count so its log stays finite and sane.
MIN_EARNERS = 0.5


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's approximation, ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"quantile level {p} outside (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_epoch(when: datetime) -> float:
    return (when - EPOCH).total_seconds() / 3600.0


def log_time(when: datetime) -> float:
    """x for a read: ln(1 + hours since the epoch). Never below 0."""
    return math.log1p(max(0.0, hours_since_epoch(when)))


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares y = a + b*x. With < 2 distinct x, slope 0 at the mean."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if n < 2 or sxx <= 0.0:
        return mean_y, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels", LEVELS)
    z_values = [norm_ppf(p) for p in levels]
    history = case.get("history") or []
    trophies = case["trophies"]
    target = parse_time(case["target"])
    x_target = log_time(target)
    horizon_days = case["horizon_minutes"] / 1440.0

    if not history:
        # No reads at all: validity first. A flat 1% median, very wide.
        sigma = SIGMA_MAX
        rows = []
        for trophy in trophies:
            lg = logit(1.0)
            rows.append({"id": trophy["id"],
                         "q": [round4(inv_logit(lg + z * sigma)) for z in z_values]})
        return {"trophies": rows}

    # One (x, players) point per read, oldest first. read_xs stays aligned
    # with history (None where a read has no usable players count), so the
    # per-trophy fits below can never pair a rate with another read's x.
    xs: list[float] = []
    log_players: list[float] = []
    read_xs: list[float | None] = []
    players_last = 0.0
    for read in history:
        players = read.get("players_tracked")
        x_read = log_time(parse_time(read["at"]))
        read_xs.append(x_read)
        if players is None or players <= 0:
            continue
        xs.append(x_read)
        log_players.append(math.log(float(players)))
        players_last = float(players)
    if not xs:
        xs = [0.0]
        log_players = [0.0]
    a_players, b_players = ols(xs, log_players)
    log_players_target = a_players + b_players * x_target

    # Was the whole history one frozen page? Then the next refresh's
    # dilution is unobserved, and the spread must carry that.
    frozen = len(set(log_players)) == 1 and all(
        read.get("rates") == history[0].get("rates") for read in history)
    sigma_step = SIGMA_STEP_FROZEN if frozen else SIGMA_STEP

    # Per-trophy earners fits, and each tier's median slope for shrinkage.
    own_slopes: list[float | None] = []
    fit_residuals: list[float | None] = []
    intercepts: list[float | None] = []
    counts: list[int] = []
    for i in range(len(trophies)):
        ex: list[float] = []
        ey: list[float] = []
        for j, read in enumerate(history):
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            players = read.get("players_tracked")
            if players is None or players <= 0 or read_xs[j] is None:
                continue
            earners = max(float(rates[i]) * float(players) / 100.0, MIN_EARNERS)
            ex.append(read_xs[j])
            ey.append(math.log(earners))
        counts.append(len(ey))
        if len(ey) >= 2:
            a_i, b_i = ols(ex, ey)
            intercepts.append(a_i)
            own_slopes.append(b_i)
            if len(ey) >= 3:
                resid = [y - (a_i + b_i * x) for x, y in zip(ex, ey)]
                mean_sq = sum(r * r for r in resid) / len(resid)
                fit_residuals.append(math.sqrt(mean_sq))
            else:
                fit_residuals.append(None)
        else:
            intercepts.append(None)
            own_slopes.append(None)
            fit_residuals.append(None)

    tier_slopes: dict[str, float] = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [b for t, b in zip(trophies, own_slopes)
                  if t.get("tier") == tier and b is not None]
        tier_slopes[tier] = median(slopes) if slopes else 0.0
    global_slope = median([b for b in own_slopes if b is not None])

    # Last non-null rate per trophy, for the fallback and the frozen page.
    last_rates: list[float | None] = []
    for i in range(len(trophies)):
        last_rate: float | None = None
        for read in history:
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last_rate = float(rates[i])
        last_rates.append(last_rate)

    sigma_rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(max(0.0, horizon_days))
    rows = []
    for i, trophy in enumerate(trophies):
        n = counts[i]
        if intercepts[i] is not None and own_slopes[i] is not None:
            tier_slope = tier_slopes.get(trophy.get("tier"), global_slope)
            w = n / (n + 2.0)
            slope = w * own_slopes[i] + (1.0 - w) * tier_slope
            log_earners_target = intercepts[i] + slope * x_target
            rate_hat = 100.0 * math.exp(log_earners_target - log_players_target)
            rate_hat = clamp_pct(rate_hat)
            if fit_residuals[i] is not None:
                sigma_fit = min(SIGMA_FIT_MAX, fit_residuals[i])
            else:
                sigma_fit = SIGMA_FIT_DEFAULT
        elif last_rates[i] is not None:
            rate_hat = clamp_pct(last_rates[i])
            sigma_fit = SIGMA_FIT_DEFAULT
        else:
            rate_hat = 1.0
            sigma_fit = SIGMA_FIT_DEFAULT
        sigma = math.sqrt(sigma_rw ** 2 + sigma_step ** 2 + sigma_fit ** 2)
        sigma = min(SIGMA_MAX, sigma)
        lg = logit(rate_hat)
        qs = [round4(inv_logit(lg + z * sigma)) for z in z_values]
        # Rounding can only tie, never invert, but enforce it outright.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
