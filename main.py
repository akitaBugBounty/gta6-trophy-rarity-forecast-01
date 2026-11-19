#!/usr/bin/env python3
"""GTA 6 trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads one case per *.in file from $MAKEFUN_INPUT_DIR (default /input) and
writes one forecast per case to $MAKEFUN_OUTPUT_DIR (default /output) as
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}: five quantiles of
each trophy's Exophase earn rate (percent) at the case's target.

Model, per trophy (all work in logit space; rates are clamped to
[0.01, 99.99] before any logit, exactly as the scorer clamps them):

* The intro trophy sitting at 100% is exempt from every other regime: its
  median is 99.6% with a narrow spread. A trophy everybody already holds
  cannot be forecast by movement, and a point forecast at 100% has no
  logit at all.
* Flat regime (the trophy's history does not move): a two-component
  mixture in logit space. With probability p the page has refreshed since
  the last read and every rate is shifted by REFRESH_SHIFT (-1.39 logit,
  odds x0.25, the launch-night refresh the house recorded: 638 tracked
  players became 3,868 and every rate but the 100% one fell); with
  probability 1-p nothing has refreshed and the rate sits where it is.
  p = 1 - exp(-(staleness + horizon) / REFRESH_TAU_H): a refresh is a
  Poisson event, so the longer the data is stale and the further the
  target, the likelier a refresh lands before it. Quantiles come from
  inverting the mixture CDF numerically, so the bimodality itself widens
  the band instead of a fatter single normal.
* Moving regime (the trophy's history moves): the median is the
  logit-average of two fits, 0.7 saturating curve + 0.3 damped trend.
  The saturating curve S_inf * (1 - exp(-t / tau)) is fit per trophy by
  grid search over tau (S_inf by least squares in rate space, tau picked
  by logit-space residuals): earn rates saturate as the player pool
  finishes the content that grants the trophy. The trend is a
  recency-weighted least-squares line of logit(rate) on ln(1 + hours)
  over the last 48h of reads, its slope shrunk halfway to its tier's mean
  slope (rare trophies have too few reads for their own slope), anchored
  at the last read and read at half the distance to the target: a
  straight log-linear line overshoots to 78-99.99% over a week, so it is
  damped twice, by the shrinkage and by the half-distance read.

Spread: sigma^2 = fit residuals + component disagreement + a random-walk
term (0.5 * sqrt(horizon in days), the house baseline's walk), clamped to
[0.25, 2.0]. Two things are deliberately NOT used: E(t)/N(t) player
counts (they overshoot systematically: the tracked-player count is not
the earning population) and the undamped log-linear trend (it runs away
over a week).

Stdlib only. Deterministic: no randomness, no wall-clock input, trophies
processed in id order, so two runs on the same case write identical bytes.
"""

import glob
import json
import math
import os
from datetime import datetime, timezone

# --- constants (fixed design numbers) -------------------------------------
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)  # normal quantiles, 5/25/50/75/95
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
REFRESH_SHIFT = -1.39          # logit shift of an Exophase refresh
REFRESH_TAU_H = 2.0            # hours: Poisson timescale of a refresh
INTRO_MEDIAN = 99.6            # median for the trophy at 100%
INTRO_MIN_RATE = 99.99         # a trophy at/above this is the intro one
SIGMA_MIN = 0.25
SIGMA_MAX = 2.0
RW_PER_SQRT_DAY = 0.5          # random-walk sigma per sqrt(day)
SAT_TAUS_H = (4.0, 8.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0)
TREND_WINDOW_H = 48.0          # only the last 48h of reads feed the trend
SLOPE_SHRINK = 0.5             # slope shrunk this far toward the tier mean
TREND_HALF = 0.5               # read the trend at half the distance out
MOVING_LOGIT_RANGE = 0.05      # history moves if its logit range exceeds this
RATE_LO = 0.01
RATE_HI = 99.99


# --- logit helpers ----------------------------------------------------------
def clamp_rate(r):
    return min(max(float(r), RATE_LO), RATE_HI)


def logit_pct(r):
    p = clamp_rate(r) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt):
    return max(0.0, (dt - RELEASE).total_seconds() / 3600.0)


# --- quantile helpers -------------------------------------------------------
def normal_quantiles(median_logit, sigma):
    """Five quantiles of N(median_logit, sigma^2), back in percent."""
    out = []
    for z in Z:
        out.append(round(inv_logit_pct(median_logit + z * sigma), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def mixture_quantiles(mu0, mu1, weight1, sigma):
    """Quantiles of (1-w)*N(mu0,s^2) + w*N(mu1,s^2), in percent.

    Bisection on the mixture CDF; the CDF is monotone so the quantiles
    are monotone by construction.
    """
    lo = min(mu0, mu1) - 6.0 * sigma - 1.0
    hi = max(mu0, mu1) + 6.0 * sigma + 1.0

    def cdf(x):
        return ((1.0 - weight1) * phi((x - mu0) / sigma)
                + weight1 * phi((x - mu1) / sigma))

    out = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(80):
            m = 0.5 * (a + b)
            if cdf(m) < tau:
                a = m
            else:
                b = m
        out.append(round(inv_logit_pct(0.5 * (a + b)), 4))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


# --- case handling ----------------------------------------------------------
def trophy_series(case, idx):
    """[(hours_since_release, rate)] for one trophy, nulls dropped."""
    pts = []
    for read in case["history"]:
        rate = read["rates"][idx]
        if rate is None:
            continue
        pts.append((hours_since_release(parse_time(read["at"])),
                    float(rate)))
    return pts


def last_rate(case, idx):
    for read in reversed(case["history"]):
        rate = read["rates"][idx]
        if rate is not None:
            return float(rate)
    return None


def forecast_flat(last_logit, stale_h, horizon_h, sigma):
    """Flat-regime forecast: refresh mixture around the last rate."""
    p_refresh = 1.0 - math.exp(-(stale_h + horizon_h) / REFRESH_TAU_H)
    return mixture_quantiles(last_logit, last_logit + REFRESH_SHIFT,
                             p_refresh, sigma)


def rw_sigma(horizon_h):
    return RW_PER_SQRT_DAY * math.sqrt(max(0.0, horizon_h) / 24.0)


# --- moving regime: saturating curve ---------------------------------------
def fit_saturating(pts):
    """Fit S_inf*(1-exp(-t/tau)) per trophy; return (pred_logit, resid_sd).

    For each tau on the grid, S_inf comes from least squares in rate
    space; tau is picked by logit-space residuals, the scale the score
    is measured on. pred is the curve read at the target hour.
    """
    best = None
    for tau in SAT_TAUS_H:
        f = [1.0 - math.exp(-t / tau) for t, _ in pts]
        num = sum(r * fi for (_, r), fi in zip(pts, f))
        den = sum(fi * fi for fi in f)
        if den <= 0.0:
            continue
        s_inf = min(max(num / den, RATE_LO), RATE_HI)
        sse = 0.0
        for (t, r), fi in zip(pts, f):
            pred = s_inf * fi
            sse += (logit_pct(pred) - logit_pct(r)) ** 2
        if best is None or sse < best[0]:
            best = (sse, s_inf, tau)
    if best is None:
        return None, 0.0
    sse, s_inf, tau = best
    resid_sd = math.sqrt(sse / max(1, len(pts)))
    return s_inf, tau, resid_sd


def saturating_at(s_inf, tau, t_hours):
    return clamp_rate(s_inf * (1.0 - math.exp(-t_hours / tau)))


# --- moving regime: damped log-linear trend ---------------------------------
def wls_slope(pts, t_last):
    """Recency-weighted slope of logit(rate) on ln(1+hours), last 48h."""
    window = [(t, r) for (t, r) in pts if t >= t_last - TREND_WINDOW_H]
    if len(window) < 2:
        window = pts[-2:] if len(pts) >= 2 else pts
    if len(window) < 2:
        return 0.0
    xs = [math.log1p(t) for t, _ in window]
    ys = [logit_pct(r) for _, r in window]
    ws = [0.5 ** ((t_last - t) / 24.0) for t, _ in window]
    sw = sum(ws)
    xm = sum(w * x for w, x in zip(ws, xs)) / sw
    ym = sum(w * y for w, y in zip(ws, ys)) / sw
    den = sum(w * (x - xm) ** 2 for w, x in zip(ws, xs))
    if den <= 0.0:
        return 0.0
    return sum(w * (x - xm) * (y - ym)
               for w, x, y in zip(ws, xs, ys)) / den


def trend_at(pts, slope, t_last, t_target):
    """Trend line anchored at the last read, read at half the distance out."""
    x_last = math.log1p(t_last)
    x_tgt = math.log1p(t_target)
    x_mid = x_last + TREND_HALF * (x_tgt - x_last)
    y = logit_pct(pts[-1][1]) + slope * (x_mid - x_last)
    return clamp_rate(inv_logit_pct(y))


# --- per-trophy forecast -----------------------------------------------------
def forecast_trophy(case, idx, tier_slopes, stale_h, horizon_h, t_target):
    pts = trophy_series(case, idx)
    r_last = last_rate(case, idx)
    if r_last is None:
        r_last = 1.0
        pts = []
    sigma_rw = rw_sigma(horizon_h)

    # The intro trophy at 100%: exempt, narrow band around 99.6%.
    if r_last >= INTRO_MIN_RATE:
        return normal_quantiles(logit_pct(INTRO_MEDIAN), SIGMA_MIN)

    if not pts:
        return normal_quantiles(logit_pct(r_last),
                                min(max(sigma_rw, SIGMA_MIN), SIGMA_MAX))

    logits = [logit_pct(r) for _, r in pts]
    moving = (max(logits) - min(logits)) > MOVING_LOGIT_RANGE

    if not moving:
        sigma = min(max(sigma_rw, SIGMA_MIN), SIGMA_MAX)
        return forecast_flat(logit_pct(r_last), stale_h, horizon_h, sigma)

    # Moving: 0.7 saturating curve + 0.3 damped trend, averaged in logit.
    t_last = pts[-1][0]
    fit = fit_saturating(pts)
    if fit[0] is None:
        s_logit = logit_pct(r_last)
        resid_sd = 0.0
    else:
        s_inf, tau, resid_sd = fit
        s_logit = logit_pct(saturating_at(s_inf, tau, t_target))

    tier = case["trophies"][idx]["tier"]
    raw_slope = wls_slope(pts, t_last)
    tier_mean = tier_slopes.get(tier)
    slope = raw_slope if tier_mean is None else (
        (1.0 - SLOPE_SHRINK) * raw_slope + SLOPE_SHRINK * tier_mean)
    trend_logit = logit_pct(trend_at(pts, slope, t_last, t_target))

    median_logit = 0.7 * s_logit + 0.3 * trend_logit
    disagree = abs(s_logit - trend_logit)
    sigma = math.sqrt(resid_sd ** 2 + (0.5 * disagree) ** 2
                      + sigma_rw ** 2)
    sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
    return normal_quantiles(median_logit, sigma)


def forecast_case(case):
    horizon_h = float(case["horizon_minutes"]) / 60.0
    t_target = hours_since_release(parse_time(case["target"]))
    at = parse_time(case["at"])
    last_read_at = None
    if case["history"]:
        last_read_at = parse_time(case["history"][-1]["at"])
    stale_h = 0.0
    if last_read_at is not None:
        stale_h = max(0.0, (at - last_read_at).total_seconds() / 3600.0)

    # Tier-mean slopes first (the trend shrinkage target per tier).
    tier_slopes = {}
    by_tier = {}
    for idx, trophy in enumerate(case["trophies"]):
        pts = trophy_series(case, idx)
        if len(pts) >= 2:
            by_tier.setdefault(trophy["tier"], []).append(
                wls_slope(pts, pts[-1][0]))
    for tier, slopes in by_tier.items():
        if slopes:
            tier_slopes[tier] = sum(slopes) / len(slopes)

    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        q = forecast_trophy(case, idx, tier_slopes, stale_h, horizon_h,
                            t_target)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
            fh.write("\n")


if __name__ == "__main__":
    main()
