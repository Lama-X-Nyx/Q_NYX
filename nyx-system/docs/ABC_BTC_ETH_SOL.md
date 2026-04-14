# A/B/C Study — BTC vs BTC+ETH vs BTC+ETH+SOL

> Choose a portfolio. Numbers regenerated via `python scripts/validate_abc.py`.

Three portfolios, same pipeline (`NYXPipeline v0.3.2`), same post-only
maker-first execution, same OOS window (2022 bear + 2023 bull).

* **Portfolio A** — BTC only (reference, already shipped as baseline)
* **Portfolio B** — BTC + ETH (validated in the earlier A/B doc)
* **Portfolio C** — BTC + ETH + SOL (this iteration)

Per-asset standalone (ETH-only, SOL-only) and the alternate pair
combinations (BTC+SOL, ETH+SOL) are reported for full transparency.

Raw data: `reports/ABC_trio_validation.json`.

---

## TL;DR

| Portfolio | n_trades | MC median return | MC % profitable | BS p5 return | BS p5 Sharpe | BS prob_loss |
|---|---:|---:|---:|---:|---:|---:|
| A — BTC           | 121 | +35.2 % | 100.0 % | +23.7 % | 3.32 | 0.0 % |
| B — BTC + ETH     | 284 | +82.4 % | 100.0 % | +63.2 % | 5.55 | 0.0 % |
| **C — BTC+ETH+SOL** | **365** | **+122.8 %** | **100.0 %** | **+98.1 %** | **6.60** | **0.0 %** |

Adding SOL on top of B:
* **+28 % more trades** (284 → 365)
* **+40 pp MC median return** (+82 % → +123 %)
* **+35 pp BS p5 return** (+63 % → +98 %) — the 5th-percentile case of
  the trio now returns nearly as much as the median case of BTC+ETH
* **+19 % BS p5 Sharpe** (5.55 → 6.60)
* **Prob_loss stays at 0.0 %** across 4 500 bootstrap sims

Portfolio C strictly dominates B on every bootstrap-derived metric.

---

## OOS walk-forward — by asset × year

| Year   | Asset | Trades | PnL ($10k) | Sharpe | Max DD | Win rate |
|---|---|---:|---:|---:|---:|---:|
| 2022 (bear) | BTC |  77 | **+$2 155** | +3.47 | 3.4 % | 70.1 % |
| 2022 (bear) | ETH |  67 | **+$2 322** | +3.62 | 2.4 % | 67.2 % |
| 2022 (bear) | **SOL** |  38 | **+$2 474** | **+5.35** | 1.2 % | 78.9 % |
| 2023 (bull) | BTC |  44 | **+$1 367** | +5.86 | 0.6 % | 81.8 % |
| 2023 (bull) | ETH |  96 | **+$2 399** | +5.06 | 1.4 % | 69.8 % |
| 2023 (bull) | **SOL** |  43 | **+$1 566** | +3.66 | 1.0 % | 76.7 % |

Notables:
* **All 6 combos (2 years × 3 assets) are PnL-positive**.
* **SOL 2022**: +24.7 % on $10k during a year SOL spot dropped ~94 %.
  The bear dial + post-only maker-first is doing exactly what it was
  designed to do, even on the most nervous asset of the trio.
* SOL trade count is lowest (38 and 43) because the session filter is
  most selective (`session_profile: momentum_fast` → tight `max_wait=2`),
  but the per-trade quality is the highest (win rate 77-79 %).

---

## Monte Carlo — trade shuffle (n_sims = 2 000)

| Pool | n_trades | % profitable | median return | dd_95 | dd_max |
|---|---:|---:|---:|---:|---:|
| A  BTC only        | 121 | 100.0 % | +35.21 % | 3.5 % | 5.4 % |
| ETH only (ref)     | 163 | 100.0 % | +47.20 % | 3.5 % | 6.6 % |
| SOL only (ref)     |  81 | 100.0 % | +40.40 % | 2.4 % | 4.5 % |
| B  BTC + ETH       | 284 | 100.0 % | +82.42 % | 3.8 % | 7.0 % |
| BTC + SOL (pair)   | 202 | 100.0 % | +75.62 % | 3.7 % | 6.1 % |
| ETH + SOL (pair)   | 244 | 100.0 % | +87.61 % | 3.8 % | 6.5 % |
| **C  BTC+ETH+SOL** | **365** | **100.0 %** | **+122.82 %** | **4.5 %** | **7.9 %** |

Not a single simulation across **14 000 permutations** (7 pools × 2 000)
produced a losing outcome. `dd_95` stays below 5 % for every pool.

---

## Bootstrap — worst-case across standard/block/regime (n_sims = 1 500)

| Pool | prob_loss | median ret | median Sharpe | p5 return | p5 Sharpe | dd_p95 |
|---|---:|---:|---:|---:|---:|---:|
| A  BTC only        | 0.0 % | +35.9 % | 5.40 | +23.7 % | 3.32 | 4.7 % |
| ETH only (ref)     | 0.0 % | +47.0 % | 5.93 | +33.7 % | 4.17 | 3.7 % |
| SOL only (ref)     | 0.0 % | +40.8 % | 7.20 | +26.7 % | 4.61 | 2.4 % |
| B  BTC + ETH       | 0.0 % | +82.6 % | 7.52 | +63.2 % | 5.55 | 5.0 % |
| BTC + SOL (pair)   | 0.0 % | +75.9 % | 7.77 | +58.8 % | 6.20 | 3.7 % |
| ETH + SOL (pair)   | 0.0 % | +87.8 % | 8.16 | +66.7 % | 6.54 | 3.8 % |
| **C  BTC+ETH+SOL** | **0.0 %** | **+122.8 %** | **8.20** | **+98.1 %** | **6.60** | **4.5 %** |

Every single row passes the institutional bars
(`prob_loss < 5 %`, `p5_return > 0`, `p5_sharpe > 1`). The trio passes
them with the largest margin.

---

## Why C > B > A (the empirical explanation)

1. **Sample size.** Bootstrap and Monte Carlo confidence both shrink
   with √n. Going from 121 → 365 trades tightens the distribution and
   pushes the 5th-percentile return up *even before* any edge change.

2. **Regime coverage.** SOL bear 2022 has the highest per-trade Sharpe
   of the 6 year×asset combos (5.35). Adding SOL does NOT add a poor
   source — it adds a distinct strong one.

3. **Low inter-asset correlation at trade level.** Trades from the three
   assets don't fire at the same bars (each strategy uses its own vol /
   regime gates). The union bootstrap therefore mixes less-correlated
   PnLs, which diversifies variance.

4. **Bear dial.** The conditional bear dial is trained per asset. SOL
   (nervous, high slippage) activates it most aggressively — which is
   exactly why SOL bear-year DD stays at 1.2 %.

---

## TDD suites backing these numbers

All GREEN as of the commit this doc ships with.

| Suite | Tests | File |
|---|---:|---|
| SOL OOS              | 5 | `tests/test_sol_oos.py` |
| SOL Monte Carlo      | 4 | `tests/test_sol_monte_carlo.py` |
| SOL Bootstrap        | 6 | `tests/test_sol_bootstrap.py` |
| BTC + SOL OOS        | 3 | `tests/test_combined_btc_sol_oos.py` |
| BTC + SOL MC         | 2 | `tests/test_combined_btc_sol_monte_carlo.py` |
| BTC + SOL Bootstrap  | 4 | `tests/test_combined_btc_sol_bootstrap.py` |
| ETH + SOL OOS        | 3 | `tests/test_combined_eth_sol_oos.py` |
| ETH + SOL MC         | 2 | `tests/test_combined_eth_sol_monte_carlo.py` |
| ETH + SOL Bootstrap  | 2 | `tests/test_combined_eth_sol_bootstrap.py` |
| **Trio OOS**         | 4 | `tests/test_combined_trio_oos.py` |
| **Trio MC**          | 3 | `tests/test_combined_trio_monte_carlo.py` |
| **Trio Bootstrap**   | 5 | `tests/test_combined_trio_bootstrap.py` |

Permanent guardrails still running on every CI:
* `tests/test_oos_final.py::TestOOSFullPipeline` (BTC baseline)
* `tests/test_eth_oos.py` + `tests/test_eth_bootstrap.py` + …
* `tests/test_mtf_feature_coverage.py` (the full MTF feature set)

---

## Recommendation

**Ship Portfolio C (BTC + ETH + SOL).**

* Strictly dominates B on every bootstrap-derived metric.
* No losing year on any asset in the 2022 + 2023 OOS window.
* Diversification shows up: trio p5 return is close to 3× BTC-only p5.
* All three assets individually pass their own MC + BS stress tests.

Operationally:
* `HubSpokeRunner` already supports N pods; use the allocator with
  `max_total_risk=0.04`, `max_asset_risk=0.015`,
  `max_cluster_risk={'alts': 0.02}` (SOL and ETH are both in `alts`).
* Keep BTC, ETH, SOL regression test suites GREEN in CI.
* If any single asset's bootstrap `prob_loss` exceeds 10 % on a new OOS
  window, drop that asset back to standby until retrained.

---

## What this study does NOT prove

* **Live execution parity.** Post-only paper broker is honest about miss
  rate but not about queue position, exchange borrow cost, or latency.
  Expect some PnL decay vs. these numbers.
* **Range-bound years.** 2022 bear + 2023 bull cover only the
  directional regimes. A flat 2019-Q2 style year is not in the bootstrap
  pool.
* **Portfolio allocator sizing.** Each asset's paper run assumed a full
  `initial_capital = $10k`. A production deployment with
  `max_asset_risk = 1.5 %` would scale these PnL numbers down; prob_loss
  / Sharpe / pct_profitable shape is unchanged (those are scale-
  invariant).
* **Cross-asset correlation in stress.** Bootstrap assumes per-trade
  IID. Real positions could be correlated during a crypto-wide flash
  event.
