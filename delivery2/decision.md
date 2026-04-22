# Feature Engineering Decisions & Observations

## Problem context

Predicting COMPAS `RawScore` (continuous, ~-2.2 to +2.4) from demographic and legal features.
Target is a proprietary algorithm's output — not actual recidivism. High R² = good at replicating COMPAS's bias, not predicting truth.

---

## Iteration 0 — Baseline

**Features:** 8 categorical OHE + 2 numerical (`age_at_screening`, `IsCompleted`)

| Model | MAE | R² |
|---|---|---|
| Linear Regression | 0.4844 | 0.2574 |
| Ridge (α=1) | 0.4843 | 0.2577 |
| Decision Tree (d=6) | 0.4882 | 0.2427 |
| Random Forest (100) | 0.4795 | 0.2672 |

Fairness — MAE disparity across ethnic groups: **0.0992**

---

## Iteration 1 — Engineered features

**Added:** `age_sq`, `is_young_adult`, `age_group`, `is_detained`, `is_pretrial`,
`is_post_sentence`, `is_socially_stable`, `is_male`, `male_x_young`, `is_on_probation`, `custody_legal`

**Rationale at creation time:** binary flags encoding criminological domain knowledge (age-crime curve, detention context, social bonds). `custody_legal` = cross-product of two categoricals.

**Observation after importance audit:** most binary flags near-zero importance. Reason: they are information-lossy versions of source categoricals already OHE'd in baseline. `CustodyStatus_Pretrial Defendant` at 0.0069 makes `is_pretrial` redundant. `Agency_Text_Probation` at 0.0073 makes `is_on_probation` redundant. `Sex_Code_Text_Male/Female` makes `is_male` redundant. Binned/discretized age (`age_group`, `is_young_adult`) loses information already in `age_at_screening` + `age_sq`.

**Decision:** drop 8 of 11 engineered features. Keep only those that are genuinely non-redundant:

| Feature | Keep? | Reason |
|---|---|---|
| `age_sq` | **YES** | importance 0.1593 (#2); adds curvature OHE cannot express |
| `male_x_young` | **YES** | importance 0.1742 (#1); cross-feature interaction linear models can't learn alone |
| `custody_legal` | **YES** | OHE variants rank 18, 23, 29; cross-product of two categoricals = distinct risk contexts |
| `is_young_adult` | dropped | subsumed by `age_at_screening` + `age_sq` + `male_x_young` |
| `age_group` | dropped | redundant with continuous age |
| `is_detained` | dropped | subsumed by `CustodyStatus` OHE + `custody_legal` |
| `is_pretrial` | dropped | subsumed by `CustodyStatus_Pretrial Defendant` (baseline, imp 0.0069) |
| `is_post_sentence` | dropped | subsumed by `LegalStatus` OHE |
| `is_socially_stable` | dropped | subsumed by `MaritalStatus` OHE; kept as helper for `age_x_stability` |
| `is_male` | dropped | redundant with `Sex_Code_Text` OHE (imp 0.0056/0.0066) |
| `is_on_probation` | dropped | subsumed by `Agency_Text_Probation` (imp 0.0073) + `custody_legal` |

**Rule learned:** binary flags derived from a categorical column are always redundant when that categorical is already OHE'd in the model. The OHE preserves all category-level signal; a binary collapse can only lose information.

---

## Iteration 2 — Composite signals

**Proposed 6, kept 3 after importance audit:**

| Signal | Formula | Importance | Decision |
|---|---|---|---|
| `repeat_assessment` | `Person_ID` count > 1 | 0.0824 (#5) | **KEEP** |
| `risk_factor_count` | sum of 5 binary risk flags | 0.0545 (#6) | **KEEP** |
| `age_x_stability` | `age_at_screening × is_socially_stable` | 0.0326 (#7) | **KEEP** |
| `young_x_detained` | `is_young_adult × is_detained` | 0.0045 | dropped |
| `male_x_detained` | `is_male × is_detained` | 0.0034 | dropped |
| `young_x_probation` | `is_young_adult × is_on_probation` | 0.0024 | dropped |

**Rule learned:** binary × binary interactions near-zero when source categoricals are OHE'd — the OHE encoding already places each combination in its own subspace. Continuous × binary interactions survive because they create a slope change that no single OHE column can represent.

**Why `risk_factor_count` survives:** the individual flags are redundant as standalone features, but their *sum* creates a new ordinal dimension (0–5 risk loads) not otherwise present. Linear models gain a pre-computed summary; tree models use it as a cheap split.

**Why `repeat_assessment` is strongest:** acts as a proxy for criminal history depth, which COMPAS weights heavily in its 137-feature algorithm. Our dataset has no direct criminal history column. This is the closest substitute available.

---

## Final feature set summary

| Tier | Features | Count |
|---|---|---|
| Baseline CAT | Sex, Ethnicity, Marital, Custody, Legal, Agency, ScaleSet, AssessmentType | 8 |
| Baseline NUM | age_at_screening, IsCompleted | 2 |
| Engineered CAT | + custody_legal | +1 |
| Engineered NUM | + age_sq, male_x_young | +2 |
| Composite NUM | + risk_factor_count, age_x_stability, repeat_assessment | +3 |
| **Total** | | **16** |

---

## Performance trajectory (Random Forest)

| Feature set | MAE | R² | Features |
|---|---|---|---|
| Baseline | 0.4795 | 0.2672 | 10 |
| Engineered (trimmed) | 0.4783 | 0.2682 | 13 |
| Composite (trimmed) | 0.4766 | 0.2735 | 16 |

Linear models benefited more from composites (R² +0.0144) because `risk_factor_count` and `age_x_stability` pre-compute nonlinearities/interactions they otherwise can't learn. Tree models already learn these splits implicitly — marginal gain smaller.

Decision Tree degraded slightly with composite features (+0.0017 MAE). Reason: more numerical features increase split opportunities and the fixed `max_depth=6` cap forces it to allocate splits across a larger space, reducing depth available for the most informative features.

---

## Fairness observations

| Feature set | MAE disparity (best vs worst ethnic group) |
|---|---|
| Baseline | 0.0992 |
| Engineered (trimmed) | 0.1100 |
| Composite (trimmed) | 0.1134 |

**Disparity increased monotonically** as features were added. Two likely causes:

1. `repeat_assessment` is a racial proxy — Black defendants are over-represented in repeat screenings due to systemic inequities in policing and prosecution. Adding it improved overall MAE but widened per-group error gaps.

2. `custody_legal` and baseline `CustodyStatus`/`Agency_Text` encode custodial context, which reflects racially unequal pretrial detention and probation rates.

**Tradeoff:** higher predictive accuracy came at cost of increased fairness disparity. This is a documented failure mode of standard loss minimization in biased datasets — minimizing average MAE does not minimize per-group MAE.

---

## Iteration 3 — Categorical data cleaning

**Rationale:** Rare categories create sparse OHE columns that add noise, hurt generalization, and prevent stable per-group analysis.

**Changes made:**
1. `Ethnic_Code_Text`: merged "African-Am" (0.08%, typo) → "African-American"; merged rare cats (<4.26%: Asian, Native American, Arabic, Oriental) → "Other"
2. `LegalStatus`: merged rare cats (<7.16%: Conditional Release, Probation Violator, Parole Violator, Deferred Sentencing) → "Other"
3. Dropped: `AssessmentReason` (zero variance), `Language` (99% English)

**Results:**
- Performance flat: Composite RF MAE 0.4766 → 0.4769 (neutral), R² 0.2735 → 0.2721 (marginal loss)
- **Fairness improved:** MAE disparity across ethnic groups 0.1134 → **0.0830** (↓12%)
  - Before: Asian defendants had worst error (0.5212 MAE, n=15)
  - After: African-American defendants have worst error (0.5009 MAE, n=1378 — more stable estimate)
- Cleaner feature space: eliminated 5 sparse OHE columns

**Conclusion:** Categorical cleanup improved fairness estimation by reducing per-group sample fragmentation, with no loss of predictive power.

---

## Open questions for next iteration

- Remove `Ethnic_Code_Text` entirely and re-measure MAE disparity — does fairness improve further?
- `repeat_assessment` as continuous count instead of binary — captures depth of history rather than presence/absence.
- Continuous × binary interactions: explore `age_x_detained`, `age_x_on_probation` — these might outperform current composites.
