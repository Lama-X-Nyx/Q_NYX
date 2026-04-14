# A/B/C Study — BTC vs BTC+ETH vs BTC+ETH+SOL (4-timeframe training)

> Choose a portfolio. Numbers regenerated via `python scripts/validate_abc.py`.

All assets trained with the **permanent 4-TF rule** — 15m + 1h + 4h + 1d.
See `docs/OPERATING_RULES.md` for the rule, guardrails, and how to add a
new asset without violating it.

* **Portfolio A** — BTC only
* **Portfolio B** — BTC + ETH
* **Portfolio C** — BTC + ETH + SOL

Raw data: `reports/ABC_trio_validation.json`.

---

## TL;DR (4TF)

| Portfolio | n_trades | MC median return | MC % profitable | BS p5 return | BS p5 Sharpe | BS prob_loss |
|---|---:|---:|---:|---:|---:|---:|
| A — BTC           | 132 | +40.1 % | 100.0 % | +29.0 % | 4.14 | 0.0 % |
| B — BTC + ETH     | 302 | +103.3 % | 100.0 % | +86.3 % | 7.22 | 0.0 % |
| **C — BTC+ETH+SOL** | **396** | **+147.8 %** | **100.0 %** | **+126.0 %** | **7.78** | **0.0 %** |

C strictly dominates B, which strictly dominates A.

---

## 3-TF vs 4-TF uplift

Adding the 4h timeframe to the feature block produced measurable
improvements on every portfolio:

| Metric | Portfolio | 3-TF | 4-TF | Δ |
|---|---|---:|---:|---:|
| MC median return   | A     | +35.2 % | **+40.1 %** | +4.9 pp |
| MC median return   | B     | +82.4 % | **+103.3 %** | +20.9 pp |
| MC median return   | C     | +122.8 % | **+147.8 %** | +25.0 pp |
| BS p5 return       | A     | +23.7 % | **+29.0 %** | +5.3 pp |
| BS p5 return       | B     | +63.2 % | **+86.3 %** | +23.1 pp |
| BS p5 return       | C     | +98.1 % | **+126.0 %** | +27.9 pp |
| BS p5 Sharpe       | A     | 3.30 | **4.14** | +25 % |
| BS p5 Sharpe       | B     | 5.55 | **7.22** | +30 % |
| BS p5 Sharpe       | C     | 6.60 | **7.78** | +18 % |

The uplift gets larger as more assets are pooled — the 4h block gives
the model a better regime signal, which pays off more when the model is
deciding across multiple assets.

---

## OOS walk-forward — 4TF — by asset × year

| Year | Asset | Trades | PnL ($10k) | Sharpe | Max DD | Win rate |
|---|---|---:|---:|---:|---:|---:|
| 2022 (bear) | BTC | 78 | **+$1 837** | +3.08 | 2.8 % | 67.9 % |
| 2022 (bear) | ETH | 72 | **+$3 330** | **+4.98** | 1.7 % | 73.6 % |
| 2022 (bear) | SOL | 46 | **+$2 690** | +5.09 | 1.3 % | 76.1 % |
| 2023 (bull) | BTC | 54 | **+$2 171** | **+9.96** | 0.4 % | 90.7 % |
| 2023 (bull) | ETH | 98 | **+$2 992** | +6.09 | 1.5 % | 74.5 % |
| 2023 (bull) | SOL | 48 | **+$1 754** | +4.21 | 1.0 % | 79.2 % |

Every single year × asset combination is PnL-positive, even bear 2022
on SOL (-94 % spot) — strategy makes **+27 % on $10k**.
BTC 2023 bull Sharpe jumped from 5.86 (3-TF) to **9.96 (4-TF)** — the
h4 block filters out weak trend signals.

---

## Monte Carlo — trade shuffle (n_sims = 2 000)

| Pool | n_trades | % profitable | median return | dd_95 | dd_max |
|---|---:|---:|---:|---:|---:|
| A  BTC only        | 132 | 100.0 % | +40.09 % | 3.2 % | 5.9 % |
| ETH only (ref)     | 170 | 100.0 % | +63.22 % | 3.4 % | 5.8 % |
| SOL only (ref)     |  94 | 100.0 % | +44.44 % | 2.5 % | 4.1 % |
| B  BTC + ETH       | 302 | 100.0 % | +103.31 % | 3.4 % | 5.8 % |
| BTC + SOL (pair)   | 226 | 100.0 % | +84.53 % | 3.1 % | 4.9 % |
| ETH + SOL (pair)   | 264 | 100.0 % | +107.66 % | 3.1 % | 5.0 % |
| **C  BTC+ETH+SOL** | **396** | **100.0 %** | **+147.75 %** | **3.4 %** | **5.8 %** |

14 000 total permutations, zero losing outcome. `dd_95` stays ≤ 3.4 %
for every pool.

---

## Bootstrap — worst-case across standard/block/regime (n_sims = 1 500)

| Pool | prob_loss | median ret | median Sharpe | p5 return | p5 Sharpe | dd_p95 |
|---|---:|---:|---:|---:|---:|---:|
| A  BTC only        | 0.0 % | +40.1 % | 6.15 | +29.0 % | 4.14 | 4.3 % |
| ETH only (ref)     | 0.0 % | +63.2 % | 7.83 | +48.9 % | 5.70 | 3.5 % |
| SOL only (ref)     | 0.0 % | +44.4 % | 7.17 | +31.8 % | 4.88 | 2.4 % |
| B  BTC + ETH       | 0.0 % | +103.3 % | 9.35 | +86.3 % | 7.22 | 4.0 % |
| BTC + SOL (pair)   | 0.0 % | +84.5 % | 9.12 | +68.6 % | 7.06 | 3.6 % |
| ETH + SOL (pair)   | 0.0 % | +107.7 % | 10.13 | +86.8 % | 7.88 | 3.4 % |
| **C  BTC+ETH+SOL** | **0.0 %** | **+147.8 %** | **10.65** | **+126.0 %** | **7.78** | **3.9 %** |

Every pool passes the institutional bars (`prob_loss < 5 %`,
`p5_return > 0`, `p5_sharpe > 1`). **Trio p5 return alone is > 3× BTC-
only median return.**

---

## Permanent rules enforced

These three guardrails make sure no future change can regress to a
reduced-TF training:

1. **Code guardrail** in `src/ml/train_asset_model.py` —
   `train_and_save()` raises `MTFCoverageError` if any of
   `h1_*`, `h4_*`, `d1_*` is missing or `n_features < 65`.
2. **Test parameterised over every asset** — `tests/test_mtf_4tf_coverage.py`
   runs on `BTCUSDT`, `ETHUSDT`, `SOLUSDT` (extend tuple when adding a
   new asset) and asserts both the feature block and saved metadata.
3. **Documented rule** — `docs/OPERATING_RULES.md` section "Rule 2 —
   Training is multi-timeframe — ALWAYS, FOR EVERY ASSET".

---

## Recommendation

**Ship Portfolio C (BTC + ETH + SOL).**

Every bootstrap-derived metric is strictly better than B, which was
strictly better than A. No losing year on any asset. p5 return 3× the
most conservative asset alone.

Operational setup:
* `HubSpokeRunner` with the 3 pods.
* Allocator: `max_total_risk=0.04`, `max_asset_risk=0.015`,
  `max_cluster_risk={'alts': 0.02}` (ETH + SOL are both `alts`).
* Keep BTC, ETH, SOL OOS + MC + BS suites GREEN on every CI run.

---

## Caveats (same as 3-TF study)

* Live parity — post-only paper broker is honest about miss rate but
  not about exchange queue position, latency, or borrow cost.
* Range-bound year not yet in the bootstrap pool.
* Per-asset paper runs assume full `initial_capital = $10k`. Real
  allocator sizing (1.5 % per asset) scales absolute PnL down;
  percentage metrics are scale-invariant.
