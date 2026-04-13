# Speed Benchmarks

## Backtest Engine Comparison

| Engine | Speed | 8,640 bars | Bottleneck |
|--------|-------|------------|------------|
| run_backtest.py (v0.8) | 70ms/bar | ~10 min | HSMM per-bar, DataFrame.iloc[:idx+1] O(n²) |
| backtest_mtf.py (precomputed) | ~1ms/bar | ~9 sec | O(1) numpy lookup per bar |
| **FastBacktester (jesse)** | **0.38ms/bar** | **3.3 sec** | Vectorized predictions + numpy loop |
| **5-Agent vectorized** | **0.03ms/bar** | **0.1 sec** (loop) | Batch predict_proba + numpy hot loop |

## Bottleneck Analysis (original run_backtest.py)

| Component | Time/bar | % | Fix |
|-----------|----------|---|-----|
| HSMM initialize_params | 22ms | 31% | Precompute once |
| SMC detect_all | 21ms | 30% | Precompute rolling window |
| _prepare_data (×2) | 10ms | 14% | Vectorize features |
| DataFrame copy (iloc[:idx+1]) | 7ms | 10% | Use numpy arrays |
| Other overhead | 10ms | 15% | — |

## FastBacktester Breakdown

| Step | Time | Notes |
|------|------|-------|
| precompute() | 0.5s | Features + labels + ATR for 8,640 bars |
| RF training | 0.8s | 200 trees, 5,500 samples |
| Batch predict_proba | 0.1s | Single call for all test bars |
| Numpy hot loop | 0.1s | Position management, 2,976 iterations |
| **Total** | **1.5s** | |

## 5-Agent Pipeline Breakdown

| Step | Time | Notes |
|------|------|-------|
| Train 4 agents | 3.7s | RF×4 on different timeframes/periods |
| Batch predict (4 models) | 0.5s | 4× predict_proba vectorized |
| Context/Regime cache | 0.1s | Pre-analyze daily + hourly |
| Numpy hot loop | 0.1s | 8,580 iterations, pure numpy |
| **Total** | **4.3s** | 128,612 bars/sec effective |
