# TICKET: SETUP INVESTIGATION V2 FLOW FIX

## STATUS: ✅ CORRIGÉ

**Date:** 2026-04-03
**Objectif:** Corriger la conservation du flux dans setup_investigation_v2

---

## PROBLÈME IDENTIFIÉ

Setup Investigation V2 initiale avait une **incohérence mathématique**:

```
patterns_kept = 0
patterns_rejected = 19
failed_alignment = 42
pre_alignment_failures = -23  ← IMPOSSIBLE!
```

**Root cause:** Mélange de deux niveaux de comptage incompatibles:
- Barres bloquées (setup_deep_dive)
- Patterns formés (comptage réel)

---

## SOLUTION IMPLÉMENTÉE

Flux canonique avec **conservation stricte**:

```
Niveau 1: Candidats vus
Niveau 2: Rejets pré-pattern (structure, gap, threshold)
Niveau 3: Patterns valides formés ⭐ (ground truth)
Niveau 4: Rejets post-pattern (alignment)
Niveau 5: Setup pass final ⭐ (ground truth)
```

### Règles de Conservation

1. `total_candidates = fvg + ob`
2. `valid_patterns = candidates - pre_failures`
3. `setup_pass = valid_patterns - post_failures`
4. Toutes valeurs >= 0 (JAMAIS négatives)
5. `failed_alignment <= valid_patterns`

---

## CORRECTIONS APPORTÉES

**Fonctions modifiées:**
- `_estimate_candidate_flow()` → `_estimate_candidate_flow_corrected()`
  - Part des ground truths (patterns observés, setup passed)
  - Dérive les autres métriques avec conservation
  
- `_estimate_rejection_ladder()` → `_estimate_rejection_ladder_corrected()`
  - Sépare pré-pattern vs post-pattern
  - Assure `failed_alignment <= valid_patterns`

**Fonctions ajoutées:**
- `_check_flow_conservation()` - Vérifie toutes les règles
- `_determine_root_cause_v2_corrected()` - Requiert flux valide

---

## EXEMPLE: FLUX CORRIGÉ

```
Candidats vus:              14
Pre-pattern failures:       10
  Structure rules:           5
  Gap threshold:             2
  Range threshold:           1
  Followthrough:             2
Valid patterns formés:       4  ← Ground truth
Post-pattern failures:       3
  Alignment:                 2  ← ≤ 4 ✅
Final setup passes:          1  ← Ground truth

Conservation:                ✅
```

**Vérification:**
- 14 - 10 = 4 ✅
- 4 - 3 = 1 ✅
- 2 ≤ 4 ✅
- Pas de négatifs ✅

---

## CONSISTENCY CHECKS

Chaque période inclut maintenant:

```json
"consistency_checks": {
  "flow_conservation_ok": true,
  "notes": []
}
```

Si conservation échoue → `diagnostic_invalid`

---

## FICHIERS MODIFIÉS/CRÉÉS

**Modifiés:**
- scripts/setup_investigation_v2.py (corrections conservation)

**Créés:**
- tests/test_setup_investigation_v2_flow.py (6 tests conservation)
- docs/SETUP_INVESTIGATION_V2_FLOW_FIX.md

**JSON produit:**
- Version mise à jour: `"version": "v2_corrected"`
- Inclut `candidate_flow`, `consistency_checks`

---

## TESTS

```bash
python tests/test_setup_investigation_v2_flow.py
```

**Tests validés:**
- ✅ Conservation rules enforced
- ✅ No negative values
- ✅ Alignment never exceeds patterns
- ✅ Rejection ladder sums correctly
- ✅ Verdict requires valid flow
- ✅ Zero patterns case handled

---

## VERDICT POSSIBLE

Maintenant que le flux est cohérent, on peut dire:

**Cas A:** "Setup sees very few candidate structures"
**Cas B:** "Setup sees candidates, but too many die before valid pattern formation"
**Cas C:** "Setup forms patterns, but alignment kills most of them"
**Cas D:** "Setup is blocked by both internal filtering and alignment"
**Cas E:** "Diagnostic invalid" (si conservation échoue)

---

## DEFINITION OF DONE ✅

- [✅] Conservation du flux respectée
- [✅] Plus de compteurs négatifs
- [✅] failed_alignment <= valid_patterns
- [✅] Verdict dépend d'un flux valide
- [✅] Tests passent (6/6)
- [✅] Documentation complète

---

## KEY INSIGHT

**Le fix transforme Setup Investigation V2 d'une "histoire incohérente" en "analyse root-cause mathématiquement cohérente" en appliquant une conservation stricte entre les niveaux du flux.**

Maintenant quand on dit "alignment is the blocker", les chiffres le supportent vraiment.
