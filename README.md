# gta6-trophy-rarity-forecast-01

Forecast each Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy, from the
house's recorded reads up to the case checkpoint. Scored by mean pinball loss
on the logit scale.

## Program

`main.py` (Python 3.13, standard library only, deterministic, ~0.1 s/case):

- Reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`), writes
  `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`):
  `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.
- **Moving history (the final case):** median = 0.7 saturating-curve fit
  `S(t) = S_inf * (1 - exp(-t/tau))` on the rate itself (tau by grid search)
  + 0.3 damped log-linear trend (WLS of logit(rate) on `ln(1 + hours)`,
  last 48 h, slope shrunk to the tier mean, half the log-time distance).
  Spread combines the recent fit residual, component disagreement, and a
  random-walk term in horizon. A counts-based E(t)/N(t) component was tried
  and dropped: earned counts are a product of two saturating curves, not a
  saturating curve, so its fit systematically overshoots.
- **Flat history (practice cases):** no trend exists to fit, so the forecast
  is a two-component refresh mixture per trophy: no-refresh (rate stays)
  vs refresh (logit shift -1.39, odds x0.25), with the refresh probability
  growing with page staleness and horizon. The 100% intro trophy is exempt
  (median 99.6%) since new players earn it first.

## Local validation

Practice cases (answers not published in the kit; reconstructed from the
published aggregates — 638 -> 3,868 players, 943 -> 4,276 trophies earned —
under five scenarios from proportional to intro-heavy; the house's stated
practice baseline loss of ~1.51 matches the intro-heavy end):

| forecast | weighted pinball (5 scenarios) |
|---|---|
| house flat / trend baseline | 1.35 |
| this entry | **0.28-0.31** |

Synthetic final-like backtests (2.5 days of 10-min reads, cut at the
deadline, scored at day 7; three dynamics families — saturating-exponential,
hyperbolic, log-linear — plus observation-noise and shock variants;
6-18 cases):

| forecast | mean pinball |
|---|---|
| house flat baseline | 0.31-0.36 |
| house trend baseline | 0.21-0.27 |
| this entry | **0.12-0.14** |

Determinism: two runs on the practice cases are byte-identical.
Runtime: ~0.07 s per practice case, ~0.25 s per synthetic final-size case
(357 reads), far under the 8 s cap.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
