# NYX PHASE 1 - BASELINE DOCUMENTATION

## 🎯 Purpose

This document defines the **FROZEN BASELINE** for NYX Phase 1 validation.

**Critical Rule:** No trading logic changes during Phase 1.

---

## 📋 What is Frozen

### 1. Strategy Logic
- NYXEngine code (`src/core/nyx_engine.py`)
- HSMM implementation (`src/core/hsmm.py`)
- SMC detector (`src/core/smc.py`)
- Macro engine (`src/macro/real_macro_engine.py`)

**NO CHANGES ALLOWED** to these files during Phase 1.

### 2. Parameters
All parameters in `config/validation_baseline.yaml` are frozen:
- SdC threshold: 3.5
- Prob threshold: 0.55
- Base size: 5%
- Stop loss: 5%
- Take profit: 15%

### 3. Data
- Pairs: BTCUSDT, ETHUSDT only
- Timeframe: 1h only
- Period: 2020-01-01 to 2024-12-31

### 4. Execution Model
- Maker fee: 0.04%
- Taker fee: 0.10%
- Slippage: 5 bps

---

## ✅ What Can Change

### Validation Code (NEW in Phase 1)
- `src/validation/*` - All validation modules
- `src/risk/*` - Risk engine and controls
- `scripts/run_validation.py` - Validation CLI
- `tests/test_validation_*.py` - Validation tests

### Reports & Analysis
- `reports/validation/*` - All validation outputs
- Analysis scripts
- Visualization tools

---

## 🚫 What is Prohibited

### During Phase 1, you CANNOT:
1. ❌ Modify signal generation logic
2. ❌ Add new indicators or features
3. ❌ Change HSMM/SMC/Macro algorithms
4. ❌ Optimize parameters based on results
5. ❌ Add new pairs or timeframes
6. ❌ Change the baseline config

### Why?

**You cannot validate a moving target.**

If you change the strategy mid-validation:
- All prior results become invalid
- You introduce look-ahead bias
- Walk-forward becomes meaningless
- You're curve-fitting, not validating

---

## 📊 Phase 1 Objectives

### What We're Proving:
1. **Statistical validity** - Does NYX hold up beyond single backtest?
2. **Robustness** - Does it survive parameter variations?
3. **Regime awareness** - When does it work/fail?
4. **Risk control** - Can we stop it when needed?
5. **Benchmark comparison** - Is it better than simple alternatives?

### What We're NOT Doing:
- ❌ Making NYX "smarter"
- ❌ Adding more alpha sources
- ❌ Optimizing for best returns
- ❌ Feature engineering

---

## 📁 Baseline Code Version

**Base:** NYX v0.8.2-FINAL

**Archive:** `nyx-v0.8.2-FINAL.tar.gz`

**Status at Phase 1 start:**
- ✅ 71/71 tests passing
- ✅ Bug-free integration
- ✅ Paper trading operational
- ✅ 8/8 GPT conditions met

---

## 🔒 Validation Integrity

### Data Manifest
Every validation run must generate:
- `data/manifests/validation_manifest.json`
- Contains: file hashes, row counts, date ranges
- Ensures reproducibility

### Configuration Tracking
Every report must include:
- Baseline version used
- Parameter snapshot
- Code commit/version

### Reproducibility
With the baseline frozen:
- Same data + same config = same results
- Results are comparable across time
- Findings are defensible

---

## ✅ Phase 1 Gate Criteria

Phase 1 is complete ONLY when you can answer:

1. **Does NYX beat simple benchmarks?**
   - vs Buy & Hold
   - vs SMA crossover
   - vs Random entry

2. **Does NYX hold in out-of-sample?**
   - OOS Sharpe > 0.5?
   - OOS max DD < 30%?
   - Performance degradation < 50%?

3. **Does NYX survive Monte Carlo?**
   - 95th percentile DD < 40%?
   - Positive median return?
   - Tail risk acceptable?

4. **What parameters are sensitive?**
   - Which cause >30% metric change?
   - Where is the system fragile?

5. **When does NYX have an edge?**
   - Which market regimes?
   - Which volatility levels?
   - Trending vs ranging?

6. **When should it be stopped?**
   - Daily DD limit?
   - Consecutive losses?
   - Error thresholds?

**If you can't answer all 6 clearly → Phase 1 not done.**

---

## 📝 Change Log

### 2025-03-28 - Baseline Frozen
- Created from v0.8.2-FINAL
- Parameters locked
- Data scope defined
- Phase 1 objectives set

---

## 🎯 Next Steps

1. ✅ Baseline frozen (this doc)
2. ⏳ Create dataset manifest (P1-01)
3. ⏳ Build metrics library (P1-02)
4. ⏳ Create validation CLI (P1-03)
5. ⏳ Implement benchmarks (P1-04)

---

**Remember:** Phase 1 = Validation, NOT optimization.

**Goal:** Prove NYX works. Don't make it "better" yet.
