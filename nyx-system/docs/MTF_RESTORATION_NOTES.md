# NYX 4-TF Baseline Restoration - Notes

**Date:** 2025-03-29  
**Version:** 0.8.2-MTF  
**Status:** Baseline Restored

---

## 🎯 **POURQUOI CETTE RESTAURATION?**

### **Problème identifié**

Tous les diagnostics Phase 1 (funnel, SMC, OB, pattern quality) ont été exécutés sur une **version simplifiée 1-TF** de NYX.

**Limitations:**
- HSMM s'exécutait sur 1h uniquement
- SMC patterns détectés sur 1h uniquement
- Pas de contexte Daily (Intent)
- Pas de stabilité 4H
- Pas d'alignement fractal

**Résultat:** Conclusions partiellement biaisées car système incomplet.

---

## ✅ **CE QUI A ÉTÉ RESTAURÉ**

### **1. Documentation Baseline (MTF_BASELINE.md)**

Définit explicitement:
- Les 5 timeframes (1D, 4H, 1H, 15M, 5M)
- Le rôle de chaque TF
- Le mapping Module→TF
- Les règles de synchronisation
- Les conditions d'entrée MTF

**Basé sur:** Prompt Maître v5.5M

---

### **2. Config MTF (validation_baseline.yaml)**

Ajouté:
```yaml
mtf:
  enabled: false  # Toggle pour diagnostics
  timeframes:
    context: "1d"
    regime: "4h"
    bridge: "1h"
    setup: "15m"
    entry: "5m"
  weights:
    "1d": 3.0
    "4h": 2.0
    "1h": 1.5
    "15m": 1.0
    "5m": 0.5
  mtf_conditions:
    sdc_min: 5.0
    stability_4h_min: 0.60
    alignment_15m_min: 0.60
    rr_min: 2.0
```

---

### **3. MTF Data Loader (src/data/mtf_loader.py)**

**Responsabilités:**
- Charge 4-5 TFs simultanément
- Aligne temporellement
- **Garantit closed-candle rule** (pas de HTF non close)
- Valide alignment

**Fonctions clés:**
```python
load(pair, tfs)  # Charge MTF data
align_at_timestamp(mtf_data, target_time, target_tf)  # Aligne avec closed rule
validate_alignment(mtf_data)  # Vérifie cohérence
```

**Tested:** ✅ Aucun look-ahead bias

---

### **4. MTF Engine (src/core/nyx_engine_mtf.py)**

**Implémente:**
- Fractal state computation (chaque TF)
- Intent_Daily (projection 1D→4H HSMM)
- Stability_4H (persistence)
- Alignment_15M (setup alignment avec Intent)
- Score_FL (fractal scoring pondéré)
- SdC (Score de Confiance)
- MTF entry conditions (6 checks)

**Entry Conditions (ALL required):**
1. SdC > 5
2. Stability_4H ≥ 0.60
3. Alignment_15M ≥ 0.60
4. RR ≥ 2:1
5. macro_block = OFF
6. P(hit -0.10) ≤ 0.20

---

### **5. CLI Mode: mtf_baseline_check**

```bash
python scripts/run_validation.py --mode mtf_baseline_check --pair BTCUSDT --sample-bars 1000
```

**Outputs:**
- Validation report (alignment, closed-candle rule)
- Sample signal avec MTF analysis
- JSON: `reports/validation/mtf/<PAIR>_mtf_baseline_check.json`

**Tested:** ✅ Fonctionne

---

## 📊 **VALIDATION RESULTS**

**Données chargées:**
- BTCUSDT: 1d (1,577 bars), 4h (9,453), 1h (37,808), 15m (151,226)
- Période: 2019-09-08 → 2024-01-01

**Closed-candle alignment vérifié:**
```
Test @ 2023-12-30 23:15:00:
  1d:  2023-12-29 00:00:00 (Δ=2835m) ✅
  4h:  2023-12-30 16:00:00 (Δ=435m)  ✅
  1h:  2023-12-30 22:00:00 (Δ=75m)   ✅
  15m: 2023-12-30 23:15:00 (Δ=0m)    ✅
```

**Aucun look-ahead bias** ✅

---

## 🚫 **CE QUI N'A PAS ÉTÉ FAIT**

### **Hors scope de cette restauration:**

1. **Optimisation stratégie** - Non
2. **Tuning thresholds** - Non
3. **Nouvelle alpha logic** - Non
4. **Filtres additionnels** - Non
5. **Performance enhancement** - Non

**Cette restauration est purement structurelle.**

Le but: remettre NYX dans sa forme native pour diagnostics corrects.

---

## 🔄 **CE QUI CHANGE**

### **Avant (1-TF):**
```python
engine.generate_signal(pair, df_1h, date)
# HSMM → 1h
# SMC → 1h
# Pas de contexte Daily
# Pas de stability check
```

### **Après (4-TF):**
```python
engine.generate_signal_mtf(pair, mtf_data, date)
# mtf_data = {
#   '1d': df_1d,   # Intent_Daily
#   '4h': df_4h,   # Stability_4H, HSMM primary
#   '1h': df_1h,   # Bridge, flip detection
#   '15m': df_15m  # SMC patterns, Alignment
# }
# Fractal scoring across TFs
# 6 MTF conditions checked
```

---

## ⚠️ **LIMITATIONS ACTUELLES**

### **Simplifications temporaires:**

1. **HSMM Integration**
   - Actuellement: régime simplifié (SMA-based)
   - TODO: Intégrer vrai HSMM 4H

2. **SMC Patterns**
   - Actuellement: pas encore appelé depuis MTF engine
   - TODO: Brancher SMCDetector sur 15M data

3. **Risk Manager**
   - Actuellement: RR/hitting-probs = placeholder
   - TODO: Calculer vrais hitting-probs MTF

4. **5M Data**
   - Manquant pour BTCUSDT
   - Disponible pour ETHUSDT

**Ces limitations seront levées en Phase 2.**

---

## 📋 **PROCHAINES ÉTAPES**

### **1. Re-run diagnostics sur MTF baseline:**

```bash
# Funnel avec MTF
python scripts/run_validation.py --mode funnel --pair BTCUSDT --sample-bars 1000 --mtf

# SMC diagnostics avec MTF
python scripts/run_validation.py --mode smc_diagnostics --pair BTCUSDT --sample-bars 1000 --mtf

# Pattern quality avec MTF
python scripts/run_validation.py --mode pattern_quality --pair BTCUSDT --sample-bars 1000 --mtf
```

### **2. Comparer 1-TF vs 4-TF:**

- Coverage change?
- Quality amélioration?
- Bottleneck shift?

### **3. Intégrations Phase 2:**

- HSMM 4H complet
- SMC 15M integration
- Risk MTF hitting-probs
- Trailing rules (5M aggressive / 1H conservative)

---

## ✅ **VALIDATION CHECKLIST**

- [✅] 4 TFs définis et documentés
- [✅] Rôles TF explicites
- [✅] Config MTF ajoutée
- [✅] MTF loader opérationnel
- [✅] Closed-candle rule enforced
- [✅] Alignment validation
- [✅] MTF engine baseline créé
- [✅] CLI mode mtf_baseline_check
- [✅] JSON output
- [✅] Aucun look-ahead bias
- [⏳] Tests unitaires (TODO)
- [⏳] Full HSMM/SMC integration (Phase 2)

---

## 🎯 **CONCLUSION**

**Baseline 4-TF restaurée.**

Diagnostics futurs pourront s'appuyer sur le vrai NYX multi-timeframe, pas sur une version simplifiée 1-TF.

**Prochaine étape recommandée:**
Re-run funnel/SMC diagnostics avec MTF engine pour comparer résultats.

---

**Archive:** `nyx-phase1-mtf-baseline-RESTORED.tar.gz`  
**Status:** Baseline structurelle complète ✅
