"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn
rate at the case target as five quantiles (5/25/50/75/95%), scored by pinball
loss on the logit scale.

Model (written from scratch for this competition, stdlib only, deterministic):

  Rate identity:  rate = earned_count / players_tracked.  Two forces move a
  rate: players progress (earned counts grow) and new players sync (the
  denominator jumps at each Exophase refresh, diluting every rate).

  Moving regime (the final case: two days of reads that actually move).
  The median is an ensemble of three forecasts, averaged in logit space:

  (a) saturating curve on the rate itself:  S(t) = S_inf * (1 - exp(-t/tau)),
      tau chosen by grid search, amplitude by weighted least squares;
      progression saturates, so a log-linear extrapolation (which would send
      a fast early riser to ~100%) is not trusted on its own;
  (b) damped log-linear trend: WLS of logit(rate) on ln(1 + hours since
      release) over the last 48 h, slope shrunk toward the tier mean,
      extrapolated only half the log-time distance (the curve is concave).
  The two are averaged in logit space (0.7 / 0.3).  (A counts-based
  E(t)/N(t) component was tried and dropped: the earned-counts curve is a
  product of two saturating curves, not a saturating curve itself, so its
  fit systematically overshoots.)

  The spread (logit sigma) combines the saturating fit's RECENT residual
  (last 48 h only, per-point residuals capped -- early near-zero rates make
  logit residuals meaningless), the disagreement between the two components,
  and a random-walk term growing with the horizon.

  Flat-history regime (practice cases): every read identical, so no trend
  exists to fit.  Exophase refreshes every few hours and a refresh dilutes
  rates: a launch-night cohort holds ~1 trophy per player, concentrated on
  the intro trophy, so a refresh drops non-intro trophies hard.  The
  forecast is then a two-component mixture per trophy: no-refresh (rate
  stays) with probability 1-p, refresh (logit shift -1.39, i.e. odds x0.25)
  with probability p, where p grows with how stale the page already is and
  with the horizon.  Quantiles come from inverting the mixture CDF.  The
  100% intro trophy is exempt (new players earn it first): median 99.6%.
  The mixture applies only for short horizons (<= 12 h) on flat histories;
  long-horizon cases always use the trend/counts model.

Usage: reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Grid of saturation timescales (hours) for the saturating-curve fits.
TAU_GRID = [4.0, 6.0, 9.0, 13.0, 19.0, 28.0, 42.0, 63.0, 95.0, 140.0,
            210.0, 320.0, 480.0, 720.0, 1080.0]


# --- logit helpers -------------------------------------------------------

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l > 20.0:
        return 99.99
    if l < -20.0:
        return 0.01
    return 100.0 / (1.0 + math.exp(-l))


def phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


# --- time helpers --------------------------------------------------------

def parse_ts(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since(t, ref):
    return max(0.0, (t - ref).total_seconds() / 3600.0)


# --- weighted least squares ----------------------------------------------

def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b, resid_var, se_b)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n > 2:
        rss = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        dof = max(1, n - 2)
        var = rss / dof / (sw / n)
        se_b = math.sqrt(max(var, 0.0) / sxx) if sxx > 0 else 0.0
    else:
        var = 0.0
        se_b = 0.0
    return (a, b, var, se_b)


def sat_fit(ts, ys, ws):
    """Fit y(t) = amp * (1 - exp(-t/tau)) by grid search over tau.

    Returns (amp, tau, weighted_sse) for the best tau, or None.
    Amplitude is the weighted least-squares estimate at fixed tau.
    """
    if len(ts) < 5:
        return None
    best = None
    for tau in TAU_GRID:
        us = [1.0 - math.exp(-t / tau) for t in ts]
        num = sum(w * u * y for w, u, y in zip(ws, us, ys))
        den = sum(w * u * u for w, u in zip(ws, us))
        if den <= 1e-12:
            continue
        amp = num / den
        if amp <= 0:
            continue
        sse = sum(w * (y - amp * u) ** 2 for w, u, y in zip(ws, us, ys))
        if best is None or sse < best[2]:
            best = (amp, tau, sse)
    return best


# --- mixture quantiles (flat-history regime) ------------------------------

def mixture_quantile(mu0, s0, mu1, s1, p, tau):
    lo, hi = -14.0, 14.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        cdf = (1.0 - p) * phi((mid - mu0) / s0) + p * phi((mid - mu1) / s1)
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- per-case forecast ----------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    hist = case.get("history") or []
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_d = case["horizon_minutes"] / 1440.0

    ref = None
    try:
        ref = parse_ts(case["game"]["release_date"])
    except Exception:
        ref = None
    if ref is None and hist:
        ref = parse_ts(hist[0]["at"])
    if ref is None:
        ref = parse_ts(case["at"])
    at_t = parse_ts(case["at"])
    target_t = parse_ts(case["target"])
    t_target = hours_since(target_t, ref)
    t_last = hours_since(parse_ts(hist[-1]["at"]), ref) if hist else 0.0
    x_target = math.log1p(t_target)
    x_last = math.log1p(t_last)

    last_rates = list(hist[-1]["rates"]) if hist else [None] * len(trophies)

    # Is the history flat? (max logit move of any trophy, and N movement)
    flat = True
    if len(hist) >= 2:
        for i in range(len(trophies)):
            vals = [h["rates"][i] for h in hist if h["rates"][i] is not None]
            if len(vals) >= 2:
                lg = [logit(v) for v in vals]
                if max(lg) - min(lg) > 0.05:
                    flat = False
                    break
        ns = [h["players_tracked"] for h in hist]
        if max(ns) != min(ns):
            flat = False
    elif len(hist) < 2:
        flat = True

    # ---- flat regime, short horizon: refresh-mixture ---------------------
    if flat and horizon_h <= 12.0 and hist:
        stale_h = max(0.0, hours_since(at_t, ref)
                      - hours_since(parse_ts(hist[0]["at"]), ref))
        # p(refresh before target): page already stale `stale_h`, refreshes
        # roughly every ~3.5 h on Exophase's own schedule.
        p_ref = 1.0 - math.exp(-(stale_h + horizon_h) / 3.5)
        p_ref = min(0.90, max(0.15, p_ref))
        drop = -1.39   # logit shift of a refresh for a non-intro trophy
        s_comp = 0.45  # spread of the refreshed component
        s_flat = 0.12  # spread of the no-refresh component
        rows = []
        for i, tr in enumerate(trophies):
            r = last_rates[i]
            if r is None:
                r = 1.0
            if r >= 99.99:
                mu, sg = logit(99.6), 0.5
                qs = [round4(clamp_pct(inv_logit(mu + z * sg))) for z in Z]
            else:
                mu0 = logit(r)
                qs = [round4(clamp_pct(inv_logit(
                    mixture_quantile(mu0, s_flat, mu0 + drop, s_comp, p_ref, tau))))
                    for tau in LEVELS]
            rows.append({"id": tr["id"], "q": qs})
        return {"trophies": rows}

    # ---- moving regime: saturating + damped trend -----------------------
    # Per-trophy series (recency weights, 24 h half-life).
    per = []
    for i, tr in enumerate(trophies):
        ts, rs, ws = [], [], []
        for h in hist:
            r = h["rates"][i]
            if r is None:
                continue
            t = hours_since(parse_ts(h["at"]), ref)
            w = 0.5 ** ((t_last - t) / 24.0)
            ts.append(t)
            rs.append(r)
            ws.append(w)
        per.append((ts, rs, ws))

    # First pass: per-trophy saturating rate fits.
    rate_fits = []
    for i, tr in enumerate(trophies):
        ts, rs, ws = per[i]
        f = sat_fit(ts, rs, ws) if len(ts) >= 5 else None
        rate_fits.append(f)

    # Tier slope pools for the damped trend component.
    tier_slopes = {}
    global_slopes = []
    trend_fits = []
    for i, tr in enumerate(trophies):
        ts, rs, ws = per[i]
        xs, ys, w2 = [], [], []
        for t, r, w in zip(ts, rs, ws):
            if t >= t_last - 48.0:
                xs.append(math.log1p(t))
                ys.append(logit(r))
                w2.append(w)
        f = wls(xs, ys, w2) if len(xs) >= 3 else None
        trend_fits.append(f)
        if f is not None:
            global_slopes.append(f[1])
            tier_slopes.setdefault(tr["tier"], []).append(f[1])
    global_slope = (sum(global_slopes) / len(global_slopes)) if global_slopes else 0.0
    tier_mean = {k: sum(v) / len(v) for k, v in tier_slopes.items()}

    rows = []
    for i, tr in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 1.0
        ts, rs, ws = per[i]

        comps = []  # (logit median, weight)
        resid_var = 0.04

        # (a) saturating rate fit (grid tau, no shrinkage: pooling tau
        # across trophies biases fast saturators upward).
        f = rate_fits[i]
        if f is not None:
            amp, tau, sse = f
            pred = amp * (1.0 - math.exp(-t_target / tau))
            comps.append((logit(pred), 0.70))
            # Residual variance from the RECENT window only, in logit
            # units, per-point residuals capped: early near-zero rates
            # (0.00% after rounding) make whole-history logit residuals
            # meaningless.
            res = []
            for t, r in zip(ts, rs):
                if t >= t_last - 48.0:
                    pr = amp * (1.0 - math.exp(-t / tau))
                    res.append(max(-1.5, min(1.5, logit(r) - logit(pr))))
            if len(res) >= 3:
                m = sum(res) / len(res)
                resid_var = max(0.01, sum((v - m) ** 2 for v in res) / len(res))

        # (b) damped log-linear trend (half the log-time distance).
        tf = trend_fits[i]
        if tf is not None:
            a, b, _, _ = tf
            tier_b = tier_mean.get(tr["tier"], global_slope)
            n_dist = len(set(round(math.log1p(t), 6) for t in ts))
            b_s = (n_dist * b + 3.0 * tier_b) / (n_dist + 3.0)
            b_s = max(-3.0, min(3.0, b_s))
            med_c = logit(r_last) + b_s * 0.5 * (x_target - x_last)
            comps.append((med_c, 0.30))

        if not comps:
            med = logit(r_last) + max(-3.0, min(3.0, global_slope)) * (x_target - x_last)
            disagree = 0.0
        else:
            wsum = sum(w for _, w in comps)
            med = sum(m * w for m, w in comps) / wsum
            disagree = (max(m for m, _ in comps) - min(m for m, _ in comps)) / 2.0 \
                if len(comps) > 1 else 0.0

        # The 100% intro trophy: keep near the top; (nearly) every new
        # player earns it first, so dilution barely touches it.
        if r_last >= 99.99:
            med = max(med, logit(99.3))
            med = min(med, logit(99.9))
        med = max(logit(0.01), min(logit(99.5), med))

        dx = max(0.0, x_target - x_last)
        rw = 0.22 * math.sqrt(max(horizon_d, 0.0))
        var = resid_var + rw ** 2 + disagree ** 2 + 0.15 ** 2
        sigma = math.sqrt(var)
        sigma = max(0.25, min(2.0, sigma))
        if r_last >= 99.99:
            sigma = min(sigma, 0.6)

        qs = []
        prev = -1.0
        for z in Z:
            q = round4(clamp_pct(inv_logit(med + z * sigma)))
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
