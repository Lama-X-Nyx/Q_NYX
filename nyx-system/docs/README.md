# NYX — Documentation index

> Navigation map for everything shipped on the
> `claude/run-pyright-system-qroCy` branch (≈ 60 commits).

---

## 📖 Start here

| Document | What you get |
|---|---|
| [`BRANCH_STORY.md`](BRANCH_STORY.md) | **The full chronological journey** — what we built, what we learned, what's honestly true |
| [`OPERATING_RULES.md`](OPERATING_RULES.md) | **The 6 permanent rules** (TDD, 4-TF MTF, post-only honest, etc.) — must stay GREEN on every CI |
| [`REALITY_CHECK.md`](REALITY_CHECK.md) | **The honest numbers** — what the headline backtest *really* means after 7 corrections |

---

## 🧱 Architecture

| Document | What you get |
|---|---|
| [`PAPER_LIVE_INFRASTRUCTURE.md`](PAPER_LIVE_INFRASTRUCTURE.md) | Paper-live stack: DecisionLogger, StateManager, PostOnlyPaperBroker, Heartbeat, EventAlerter, MultiPairRunner |
| [`HUB_AND_SPOKE_ARCHITECTURE.md`](HUB_AND_SPOKE_ARCHITECTURE.md) | Hub-and-spoke: AssetRegistry, ExecutionProfile per-asset, Signal contract, PortfolioAllocator, HubSpokeRunner |

---

## 🔬 Validation studies

| Document | Scope | Key number |
|---|---|---|
| [`AB_BTC_vs_BTC_ETH.md`](AB_BTC_vs_BTC_ETH.md) | BTC-only vs BTC+ETH portfolio | B p5 Sharpe 7.22 |
| [`ABC_BTC_ETH_SOL.md`](ABC_BTC_ETH_SOL.md) | A/B/C adding SOL | C p5 Sharpe 7.78 |
| [`WALK_FORWARD_TRIO.md`](WALK_FORWARD_TRIO.md) | Annualized trio walk-forward 2020-2023 | CAGR +49 % (headline, see reality check for the honest number) |
| [`REALITY_CHECK.md`](REALITY_CHECK.md) | 7 corrections to the headline numbers | Realistic CAGR live: +15-25 % |

---

## 🗂 Historical docs (pre-branch, kept for context)

Pipeline internals, HSMM investigation, SMC audit, feature engineering:

- `PIPELINE_V031_RESULTS.md`
- `MC_FINAL_V031.md`
- `BOOTSTRAP_REPORT.md`
- `MONTE_CARLO_REPORT.md`
- `JESSE_ARCHITECTURE.md`
- `HSMM_DEEP_DIVE.md`, `HSMM_INPUT_AUDIT.md`, `HSMM_VERDICT_REFINEMENT.md`
- `SETUP_INVESTIGATION*.md` (7 files — setup agent deep dives)
- `SMC_AUTOPSY.md`, `SMC_VISUAL_TRUTH_AUDIT.md`, `SMC_WIRING_FIX.md`
- `FRACTAL_*.md` (7 files — fractal architecture)
- `MTF_BASELINE.md`, `MTF_RESTORATION_NOTES.md`
- `REGIME_FEATURE_ALIGNMENT.md`, `REGIME_TUNING.md`
- `BULLISH_AUDIT.md`, `OOS_BOUNDED_LOOKBACK.md`

Keep them for pipeline internals reference. They are not the "current state" docs.

---

## 📊 Reports (raw numbers)

| File | What |
|---|---|
| `reports/AB_BTC_vs_BTC_ETH.json` | A/B full data |
| `reports/ABC_trio_validation.json` | A/B/C full data (4-TF) |
| `reports/walk_forward_trio.json` | Walk-forward annualized trio |
| `reports/walk_forward_trio_summary.txt` | Human-readable summary |
| `reports/reality_check_numbers.json` | 7 reality-check measurements |
| `reports/ETHUSDT_oos_report.json` | ETH 2023 OOS report |
| `reports/SOLUSDT_oos_report.json` | SOL 2023 OOS report |

All reports are regeneratable via `python scripts/*.py`. See
[`BRANCH_STORY.md`](BRANCH_STORY.md) §tooling for the full list.

---

## 🧪 Tests at a glance

(See `tests/test_*.py` and [`OPERATING_RULES.md`](OPERATING_RULES.md)
for the permanent guardrails.)

| Suite | Count | Subject |
|---|---:|---|
| Paper-live (decision log, state, broker, heartbeat, telegram, discord, multi-alerter, event-alerter, data validator, resync, post-only, missed, runner, multi-pair, runner-events, runner-post-only, hub-spoke) | 132 | Paper-live infrastructure |
| Hub-and-spoke (AssetRegistry, ExecutionProfile, Signal, PortfolioAllocator) | 39 | Multi-asset architecture |
| Pipeline (MTF coverage 3-TF + 4-TF) | 23 | Feature block integrity |
| Per-asset training (ETH, SOL + trio + pairs) | 43 | OOS + MC + BS per portfolio |
| Walk-forward trio + reality checks | 9 + 15 | Annualized, honest corrections |
| BTC baseline (OOS final) | 4 | Regression guard |
| **Total on this branch** | **265+ GREEN** | — |

---

## 🔧 Tooling / scripts

| Script | What |
|---|---|
| `scripts/train_eth_model.py` | Train ETH model artefact under `models/ETHUSDT/` |
| `scripts/train_sol_model.py` | Same for SOL |
| `scripts/compute_4h_features.py` | Cache 4h stationary features for BTC/ETH/SOL |
| `scripts/validate_ab.py` | A/B BTC vs BTC+ETH — writes `reports/AB_BTC_vs_BTC_ETH.json` |
| `scripts/validate_abc.py` | A/B/C with SOL — writes `reports/ABC_trio_validation.json` |
| `scripts/walk_forward_trio.py` | Annualized walk-forward 2020-2023 |
| `scripts/run_reality_checks.py` | 7 reality checks + forced-stop B&H |

---

## 🚦 Red-flag rules (can't be silently broken)

From [`OPERATING_RULES.md`](OPERATING_RULES.md):

1. **TDD strict** — RED before GREEN, no module ships without tests.
2. **4-TF always** — every training uses 15m + 1h + 4h + 1d, for every asset.
3. **Honest execution** — PostOnlyPaperBroker only, no taker fallback.
4. **No silent data loss** — append-only DBs, atomic state, fan-out alerts.
5. **Every operator-relevant event is alerted** — 6 semantic events.
6. **Pyright clean** — 0 errors on new code.

Permanent guardrail tests (must stay GREEN):

- `test_oos_final.py::TestOOSFullPipeline` — BTC baseline
- `test_mtf_4tf_coverage.py` — 4-TF rule, parameterised on every asset
- `test_post_only_broker.py::TestTimeout::test_no_taker_fallback`
- `test_paper_live_runner.py` — wire-up integrity
- `test_walk_forward_trio.py` — annualized walk-forward invariants
- `test_reality_checks.py` — 7 honesty corrections
