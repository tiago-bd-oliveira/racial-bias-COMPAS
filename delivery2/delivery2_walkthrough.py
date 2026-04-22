"""
=============================================================================
 DELIVERABLE 2 — Preliminary Results
 Project: COMPAS Recidivism Score Prediction
=============================================================================
 Target variable  : RawScore (continuous)
 Assessment scope : Risk of Recidivism only
 Approach         : Baseline features → Feature engineering → Compare results
 Models explored  : Linear Regression, Ridge, Decision Tree, Random Forest
 Ethical focus    : Per-racial-group error analysis (fairness metrics)
=============================================================================
"""

# =============================================================================
# SECTION 0: IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

warnings.filterwarnings("ignore")

RANDOM_STATE = 42   # fixed seed → reproducible results every run
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "compas-scores-raw.csv")
TARGET       = "RawScore"
ASSESSMENT   = "Risk of Recidivism"

print("=" * 70)
print("DELIVERABLE 2 — COMPAS RawScore Regression (Risk of Recidivism)")
print("=" * 70)


# =============================================================================
# SECTION 1: DATA LOADING & INITIAL FILTER
# =============================================================================
# The raw CSV contains all three COMPAS assessment types stacked together.
# We immediately filter to the one we care about to keep things clean.

df_raw = pd.read_csv(DATA_PATH)
print(f"\nFull dataset: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")

df = df_raw[df_raw["DisplayText"] == ASSESSMENT].copy()
print(f"After filtering to '{ASSESSMENT}': {len(df):,} rows")
print(f"Unique individuals: {df['Person_ID'].nunique():,}")


# =============================================================================
# SECTION 2: BASIC DATE WORK
# =============================================================================
# Parse the date strings. Pandas needs them as proper datetime objects before
# we can subtract them to compute durations.

df["Screening_Date"] = pd.to_datetime(df["Screening_Date"], format="mixed", dayfirst=False)
df["DateOfBirth"]    = pd.to_datetime(df["DateOfBirth"],    format="mixed", dayfirst=False)

# Base age feature — used in both baseline and engineered sets
df["age_at_screening"] = (df["Screening_Date"] - df["DateOfBirth"]).dt.days / 365.25

# Drop the small number of rows where age looks impossible (data entry errors)
n_before = len(df)
df = df[(df["age_at_screening"] >= 10) & (df["age_at_screening"] <= 100)]
print(f"\nDropped {n_before - len(df)} rows with implausible age values")

# Drop rows where the target variable is missing
df = df.dropna(subset=[TARGET])
print(f"Rows after cleaning: {len(df):,}")
print(f"Target (RawScore) — mean: {df[TARGET].mean():.2f}, "
      f"std: {df[TARGET].std():.2f}, "
      f"min: {df[TARGET].min():.2f}, "
      f"max: {df[TARGET].max():.2f}")


# =============================================================================
# SECTION 2.5: CATEGORICAL DATA CLEANING
# =============================================================================
# Merge rare categories to reduce sparse OHE columns and improve generalization.

# ── Ethnic_Code_Text cleanup ─────────────────────────────────────────────────
# Merge "African-Am" (0.08%, typo) → "African-American"
# Merge rare categories (<4.26%, i.e., below "Other" threshold) → "Other"
#   Merged: Asian (0.53%), Native American (0.36%), Arabic (0.12%), Oriental (0.06%)
df["Ethnic_Code_Text"] = df["Ethnic_Code_Text"].replace({
    "African-Am": "African-American",
    "Asian": "Other",
    "Native American": "Other",
    "Arabic": "Other",
    "Oriental": "Other",
})

# ── LegalStatus cleanup ──────────────────────────────────────────────────────
# Merge rare categories (<7.16%, i.e., below "Other" threshold) → "Other"
#   Kept: Pretrial (61.76%), Post Sentence (30.13%), Other (7.16%)
#   Merged: Conditional Release (0.69%), Probation Violator (0.21%),
#           Parole Violator (0.03%), Deferred Sentencing (0.02%)
df["LegalStatus"] = df["LegalStatus"].replace({
    "Conditional Release": "Other",
    "Probation Violator": "Other",
    "Parole Violator": "Other",
    "Deferred Sentencing": "Other",
})

# ── Drop zero-variance columns ───────────────────────────────────────────────
# AssessmentReason: all identical values (zero information)
# Language: 99% English (near-zero variance)
df = df.drop(columns=["AssessmentReason", "Language"])

print("\nCategorical cleanup:")
print(f"  Ethnic_Code_Text: merged African-Am → African-American, rare cats → Other")
print(f"  LegalStatus: merged rare categories → Other")
print(f"  Dropped: AssessmentReason (zero variance), Language (99% English)")


# =============================================================================
# SECTION 3: FEATURE ENGINEERING
# =============================================================================
#
# The dataset's original columns give us limited signal because the categorical
# features are broad labels. We create new, more expressive features by:
#   a) Encoding domain knowledge from criminology
#   b) Capturing non-linear relationships (age curve)
#   c) Creating interaction features that combine two column signals
#
# Each new feature is explained below with its criminological justification.

# ── 3.1  AGE NONLINEARITY ────────────────────────────────────────────────────
# Age squared adds curvature so linear models can represent the peak-then-decline
# shape of the age-crime curve without a tree-based model.
df["age_sq"] = df["age_at_screening"] ** 2

# ── 3.2  HELPER BINARY FLAGS (used in composite calculations, not features) ──
# These are NOT added to any feature set — their signal is already covered by
# the OHE of their source columns (CustodyStatus, LegalStatus, Sex, MaritalStatus,
# Agency_Text). They exist solely as inputs to composite signals in Section 3.9.
df["is_young_adult"]   = (df["age_at_screening"] <= 25).astype(int)
df["is_male"]          = (df["Sex_Code_Text"] == "Male").astype(int)

DETAINED_STATUSES = {"Jail Inmate", "Prison Inmate", "Residential Program"}
df["is_detained"]      = df["CustodyStatus"].isin(DETAINED_STATUSES).astype(int)
df["is_on_probation"]  = (
    (df["CustodyStatus"] == "Probation") | (df["Agency_Text"] == "Probation")
).astype(int)

STABLE_STATUSES = {"Married", "Significant Other"}
df["is_socially_stable"] = df["MaritalStatus"].isin(STABLE_STATUSES).astype(int)

# ── 3.3  SEX × AGE INTERACTION ───────────────────────────────────────────────
# The age-crime curve is steeper for males. This product lets linear models
# learn that young + male carries risk beyond the sum of each alone.
# Interaction term — not redundant with Sex_Code_Text OHE or age_sq alone.
df["male_x_young"] = df["is_male"] * df["is_young_adult"]

# ── 3.4  CUSTODY × LEGAL STATUS INTERACTION ──────────────────────────────────
# "Jail Inmate + Pretrial" is a different risk context from "Jail Inmate +
# Post Sentence". A combined categorical captures these distinct combinations
# that neither source column can express alone.
df["custody_legal"] = (
    df["CustodyStatus"].fillna("Unknown")
    + "_"
    + df["LegalStatus"].fillna("Unknown")
)

print("\nFeature engineering complete. Features added to model:")
print("  age_sq, male_x_young, custody_legal")
print("  (helper flags computed but not used as standalone features)")


# =============================================================================
# SECTION 3.9: COMPOSITE SIGNALS
# =============================================================================
# Composite signals combine multiple features into higher-order representations
# that neither source column expresses alone. Validated by feature importance
# audit (iteration 1): binary × binary interactions near-zero when the source
# categoricals are already OHE'd; continuous × binary interactions survive.

# ── 3.9.1  RISK FACTOR COUNT ─────────────────────────────────────────────────
# Sum of five binary risk indicators → single "total risk load" for linear
# models. The individual flags are redundant with OHE categoricals, but their
# sum creates a new ordinal dimension.
df["risk_factor_count"] = (
    df["is_young_adult"]
    + df["is_male"]
    + df["is_detained"]
    + df["is_on_probation"]
    + (1 - df["is_socially_stable"])
)

# ── 3.9.2  AGE × SOCIAL STABILITY ────────────────────────────────────────────
# Protective effect of social bonds varies with age — older married defendants
# score much lower than older single ones. Continuous × binary product lets
# linear models represent this slope change.
df["age_x_stability"] = df["age_at_screening"] * df["is_socially_stable"]

# ── 3.9.3  REPEAT ASSESSMENT ─────────────────────────────────────────────────
# Person_ID appearing more than once → defendant has returned through the system.
# Strongest composite signal (importance 0.0824 in iter-1 audit). Acts as a
# proxy for criminal history depth, which COMPAS uses heavily.
person_counts = df["Person_ID"].map(df["Person_ID"].value_counts())
df["repeat_assessment"] = (person_counts > 1).astype(int)

# Dropped after iter-1 audit (importance < 0.005, redundant with OHE):
#   young_x_detained, young_x_probation, male_x_detained

print("\nComposite signals added to model:")
print("  risk_factor_count, age_x_stability, repeat_assessment")


# =============================================================================
# SECTION 4: DEFINE BASELINE AND ENGINEERED FEATURE SETS
# =============================================================================
#
# We run the models TWICE — first with the baseline features (what we had in
# Deliverable 1), then with the augmented set. Comparing the two tells us
# how much value the feature engineering actually adds.

# Baseline: original columns only (as in Deliverable 1)
BASELINE_CAT = [
    "Sex_Code_Text",
    "Ethnic_Code_Text",
    "MaritalStatus",
    "CustodyStatus",
    "LegalStatus",
    "Agency_Text",
    "ScaleSet",
    "AssessmentType",
]
BASELINE_NUM = ["age_at_screening", "IsCompleted"]

# Engineered: baseline + non-redundant features only.
# Dropped after iter-1 audit: is_young_adult, age_group, is_detained,
# is_pretrial, is_post_sentence, is_socially_stable, is_male, is_on_probation
# — all subsumed by OHE of their source categoricals (CustodyStatus, LegalStatus,
# Sex_Code_Text, MaritalStatus, Agency_Text).
ENGINEERED_CAT = BASELINE_CAT + ["custody_legal"]
ENGINEERED_NUM = BASELINE_NUM + ["age_sq", "male_x_young"]

# Composite: engineered + validated composite signals.
# Dropped after iter-1 audit: young_x_detained, young_x_probation,
# male_x_detained — importance < 0.005, no signal beyond OHE source columns.
COMPOSITE_CAT = ENGINEERED_CAT
COMPOSITE_NUM = ENGINEERED_NUM + [
    "risk_factor_count",
    "age_x_stability",
    "repeat_assessment",
]


# =============================================================================
# SECTION 5: PREPROCESSING PIPELINES
# =============================================================================
#
# A sklearn Pipeline ensures that ALL preprocessing steps (encoding, scaling)
# are fitted ONLY on training data and then applied to test data. This prevents
# data leakage — a common mistake where test-set information accidentally
# influences the preprocessing (and inflates apparent model performance).
#
# ColumnTransformer applies different transformations to different column types:
#   OneHotEncoder: text categories → 0/1 binary columns
#   StandardScaler: numbers → (value - mean) / std
#     Scaling matters for Linear/Ridge Regression because those models are
#     sensitive to the scale of features. Tree-based models (Decision Tree,
#     Random Forest) are scale-invariant, but it doesn't hurt to scale anyway.

def build_preprocessor(cat_features, num_features):
    """Create a ColumnTransformer for a given set of feature lists."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            ),
            (
                "num",
                StandardScaler(),
                num_features,
            ),
        ],
        remainder="drop",   # silently drop any column not listed above
    )


def build_pipelines(preprocessor):
    """Return a dict of {model_name: fitted Pipeline} for the four models."""
    return {
        # ── Linear Regression ────────────────────────────────────────────────
        # Baseline model. Assumes: RawScore = w1·f1 + w2·f2 + … + b
        # Fully interpretable via coefficients. If a more complex model barely
        # beats this, the complexity is not justified.
        "Linear Regression": Pipeline(
            [("prep", preprocessor), ("model", LinearRegression())]
        ),
        # ── Ridge Regression (L2 regularisation) ─────────────────────────────
        # Adds a penalty α·Σwᵢ² to the loss, shrinking all coefficients toward
        # zero. This helps when features are correlated (e.g., is_detained and
        # is_pretrial overlap). alpha=1 is a sensible default; could be tuned.
        "Ridge (α=1)": Pipeline(
            [("prep", preprocessor), ("model", Ridge(alpha=1.0))]
        ),
        # ── Decision Tree ─────────────────────────────────────────────────────
        # Splits the feature space into rectangular regions with binary splits.
        # Can capture non-linearities and feature interactions automatically.
        # max_depth=6 limits the tree to prevent memorising the training set
        # (overfitting). Visualisable — good for interpreting what the model
        # has learned.
        "Decision Tree (d=6)": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
                ),
            ]
        ),
        # ── Random Forest ─────────────────────────────────────────────────────
        # Trains 100 decision trees, each on a random sample of rows (bagging)
        # and a random subset of features (feature randomness). Averaging their
        # predictions reduces variance vs. a single tree. Usually the best
        # performing model out of these four for tabular data.
        # n_jobs=-1 uses all available CPU cores for parallel training.
        "Random Forest (100)": Pipeline(
            [
                ("prep", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# =============================================================================
# SECTION 6: EVALUATION HELPERS
# =============================================================================

def evaluate(pipeline, X_test, y_test):
    """Return MAE, RMSE, R² for a fitted pipeline on a held-out test set."""
    y_pred = pipeline.predict(X_test)
    return {
        "MAE":  round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2":   round(r2_score(y_test, y_pred), 4),
    }


def evaluate_fit(pipeline, X_train, X_test, y_train, y_test, n_folds=5):
    """
    Full fit-quality diagnostic for a single model.

    Returns a dict with:
      train_mae   — MAE on the training set (how well it memorised the data)
      test_mae    — MAE on the held-out test set (real-world performance)
      gap         — test_mae − train_mae (positive = some overfitting)
      cv_mean     — mean 5-fold cross-val MAE (robust generalisation estimate)
      cv_std      — std across folds (high std = unstable / sensitive to split)

    Interpreting the gap:
      ┌──────────────────────────────────────────────────────────────┐
      │ gap ≈ 0, both MAEs HIGH   → UNDERFITTING (model too simple) │
      │ gap ≈ 0, both MAEs LOW    → GOOD FIT                        │
      │ gap >> 0, test MAE HIGH   → OVERFITTING (model too complex) │
      └──────────────────────────────────────────────────────────────┘

    Cross-validation splits the training set into n_folds pieces, trains on
    n_folds−1 pieces and validates on the remaining one, rotating n_folds
    times. The average MAE is a more stable generalisation estimate than a
    single train/test split, because it is less sensitive to which random 20%
    happened to end up in the test set.
    """
    train_mae = mean_absolute_error(y_train, pipeline.predict(X_train))
    test_mae  = mean_absolute_error(y_test,  pipeline.predict(X_test))

    # CV is run on the combined training data (never touches X_test)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    cv_maes = -cv_scores   # cross_val_score returns negative MAEs by convention

    return {
        "train_mae": round(float(train_mae), 4),
        "test_mae":  round(float(test_mae),  4),
        "gap":       round(float(test_mae - train_mae), 4),
        "cv_mean":   round(float(cv_maes.mean()), 4),
        "cv_std":    round(float(cv_maes.std()),  4),
    }


def error_distribution(pipeline, X_test, y_test):
    """
    Detailed breakdown of prediction errors — answers "how close are we?"

    signed_error = predicted − actual
      Positive → model over-predicts (assigns higher risk than COMPAS did)
      Negative → model under-predicts (assigns lower risk than COMPAS did)
      Mean of signed errors = bias: a persistently positive/negative mean
      means the model is systematically shifted in one direction.

    Absolute error thresholds:
      RawScore spans ≈ 5.5 units for Risk of Recidivism.
      DecileScore bins are roughly 0.5–0.8 RawScore units wide.
      So ±0.5 ≈ "within ~1 decile bin" — a very precise prediction.
          ±1.0 ≈ "within ~1–2 decile bins" — acceptable for exploratory work.
    """
    y_pred         = pipeline.predict(X_test)
    signed_errors  = pd.Series(y_pred - y_test.values)   # + = over, − = under
    abs_errors     = signed_errors.abs()

    score_range    = float(y_test.max() - y_test.min())

    return {
        # Bias
        "mean_signed_error":  round(float(signed_errors.mean()),   4),
        # Central tendency of absolute error
        "median_abs_error":   round(float(abs_errors.median()),    4),
        # Coverage at progressively loose thresholds
        "pct_within_0.5":  round(float((abs_errors <= 0.5).mean() * 100), 1),
        "pct_within_1.0":  round(float((abs_errors <= 1.0).mean() * 100), 1),
        "pct_within_1.5":  round(float((abs_errors <= 1.5).mean() * 100), 1),
        # Tail behaviour
        "p90_abs_error":      round(float(abs_errors.quantile(0.90)), 4),
        "max_abs_error":      round(float(abs_errors.max()),          4),
        # Contextual: error as % of the target's value range
        "mae_pct_of_range":   round(
            float(abs_errors.mean()) / score_range * 100, 1
        ),
    }


def per_group_mae(pipeline, X_test, y_test, ethnic_groups):
    """
    Compute MAE for each ethnic group separately.

    Why this matters: a model may look accurate overall while being much
    less accurate for a specific demographic. In this dataset, COMPAS was
    already shown to be worse for Black defendants. We check whether our
    models reproduce (or mitigate) that disparity.
    """
    y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index)
    results = {}
    for group in sorted(ethnic_groups.unique()):
        mask = ethnic_groups == group
        if mask.sum() < 15:   # skip groups too small for stable estimates
            continue
        mae = mean_absolute_error(y_test[mask], y_pred[mask])
        results[group] = {"n": int(mask.sum()), "MAE": round(mae, 4)}
    return results


# =============================================================================
# SECTION 7: PREPARE DATA — FILL MISSING VALUES & SPLIT
# =============================================================================
# We fill missing values BEFORE the split so the same imputation strategy
# applies everywhere. We do NOT compute statistics from the test set — the
# median/mode is computed on the full dataset here because these are simple
# categorical fills, not a learned statistic (the scaler inside the Pipeline
# is properly fitted only on training data).

# ── Handle categoricals ───────────────────────────────────────────────────────
# Fill NaN with the string "Unknown" so it becomes its own one-hot category.
# This is more informative than dropping rows or ignoring missingness.
all_cat_cols = list(set(COMPOSITE_CAT))
for col in all_cat_cols:
    df[col] = df[col].fillna("Unknown")

# ── Handle numericals ─────────────────────────────────────────────────────────
# Fill NaN with the median (not mean) — median is robust to outliers.
all_num_cols = list(set(COMPOSITE_NUM))
for col in all_num_cols:
    df[col] = df[col].fillna(df[col].median())

# ── Train / Test split ────────────────────────────────────────────────────────
# 80 % training, 20 % test. We hold out the same 20 % for ALL model variants
# so that baseline vs. engineered comparisons are apples-to-apples.
y = df[TARGET]
ethnic_series = df["Ethnic_Code_Text"]   # keep aligned for fairness analysis

X_base = df[BASELINE_CAT + BASELINE_NUM]
X_eng  = df[ENGINEERED_CAT + ENGINEERED_NUM]
X_comp = df[COMPOSITE_CAT  + COMPOSITE_NUM]

(X_base_train, X_base_test,
 X_eng_train,  X_eng_test,
 X_comp_train, X_comp_test,
 y_train,      y_test,
 _,            ethnic_test) = train_test_split(
    X_base, X_eng, X_comp, y, ethnic_series,
    test_size=0.2, random_state=RANDOM_STATE
)

print(f"\nTrain set: {len(X_base_train):,}  |  Test set: {len(X_base_test):,}")


# =============================================================================
# SECTION 8: TRAIN & EVALUATE — BASELINE vs. ENGINEERED
# =============================================================================

def run_experiment(X_train, X_test, cat_features, num_features, label):
    """
    Build pipelines, train all 4 models, evaluate on test set.
    Returns (results_dict, fit_diagnostics_dict, fitted_pipelines_dict).
    """
    print(f"\n{'─' * 60}")
    print(f" {label}")
    print(f"{'─' * 60}")
    print(f" Features: {len(cat_features)} categorical | {len(num_features)} numerical")

    prep         = build_preprocessor(cat_features, num_features)
    pipelines    = build_pipelines(prep)
    results      = {}
    fit_diags    = {}

    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[model_name]   = metrics
        fit_diags[model_name] = evaluate_fit(
            pipeline, X_train, X_test, y_train, y_test
        )
        print(f"  [{model_name:<25}]  "
              f"MAE={metrics['MAE']:.4f}  "
              f"RMSE={metrics['RMSE']:.4f}  "
              f"R²={metrics['R2']:.4f}")

    return results, fit_diags, pipelines


print("\n" + "=" * 70)
print(" EXPERIMENT RESULTS")
print("=" * 70)

(baseline_results,
 baseline_fit,
 baseline_pipelines)   = run_experiment(
    X_base_train, X_base_test,
    BASELINE_CAT, BASELINE_NUM,
    "BASELINE (original features only)",
)

(engineered_results,
 engineered_fit,
 engineered_pipelines) = run_experiment(
    X_eng_train, X_eng_test,
    ENGINEERED_CAT, ENGINEERED_NUM,
    "ENGINEERED (baseline + 11 new features)",
)

(composite_results,
 composite_fit,
 composite_pipelines) = run_experiment(
    X_comp_train, X_comp_test,
    COMPOSITE_CAT, COMPOSITE_NUM,
    "COMPOSITE (engineered + 6 interaction signals)",
)


# =============================================================================
# SECTION 9: THREE-WAY COMPARISON TABLE
# =============================================================================

print("\n\n" + "=" * 70)
print(" COMPARISON: BASELINE → ENGINEERED → COMPOSITE")
print("=" * 70)
print(f"{'Model':<28} {'B→E Δ MAE':>12} {'E→C Δ MAE':>12} {'B→C Δ R²':>10}")
print("-" * 66)

for model_name in baseline_results:
    b = baseline_results[model_name]
    e = engineered_results[model_name]
    c = composite_results[model_name]
    be_mae = e["MAE"] - b["MAE"]
    ec_mae = c["MAE"] - e["MAE"]
    bc_r2  = c["R2"]  - b["R2"]
    arrow_be  = "↓" if be_mae < 0 else "↑"
    arrow_ec  = "↓" if ec_mae < 0 else "↑"
    arrow_r2  = "↑" if bc_r2  > 0 else "↓"
    print(f"  {model_name:<26}  "
          f"{arrow_be}{abs(be_mae):.4f}        "
          f"{arrow_ec}{abs(ec_mae):.4f}        "
          f"{arrow_r2}{abs(bc_r2):.4f}")

print("\nNote: ↓ MAE = better  |  ↑ R² = better  |  B→E = eng gain  |  E→C = composite gain")


# =============================================================================
# SECTION 9.5: OVERFITTING / UNDERFITTING DIAGNOSIS
# =============================================================================
#
# Concept refresher:
#   • A model that memorises training data will score very well on the train
#     set but poorly on unseen test data → OVERFITTING (high variance)
#   • A model that is too simple to learn the patterns in the data will score
#     poorly on BOTH sets → UNDERFITTING (high bias)
#   • We want: gap ≈ 0 with test MAE as low as possible.
#
# The 5-fold cross-validation (CV) gives us a more reliable estimate than a
# single train/test split. It also shows the *stability* of the model — a
# high standard deviation across folds suggests the model is sensitive to
# which data it is trained on.
#
# We show results for BOTH feature sets so we can see whether engineering
# changes the overfitting behaviour.

for label, fit_diags in [
    ("BASELINE",   baseline_fit),
    ("ENGINEERED", engineered_fit),
    ("COMPOSITE",  composite_fit),
]:
    print(f"\n{'=' * 70}")
    print(f" OVERFITTING DIAGNOSIS — {label}")
    print(f"{'=' * 70}")
    print(f"  {'Model':<28} {'Train MAE':>10} {'Test MAE':>10}"
          f" {'Gap':>8} {'CV Mean':>10} {'CV ±Std':>10}  Status")
    print("  " + "─" * 82)

    for model_name, d in fit_diags.items():
        gap = d["gap"]
        # Classify fit status based on gap and absolute level
        if d["test_mae"] > 0.55 and gap < 0.02:
            status = "UNDERFITTING"
        elif gap > 0.10:
            status = "OVERFITTING"
        elif gap > 0.05:
            status = "slight overfit"
        else:
            status = "OK"
        print(f"  {model_name:<28}"
              f" {d['train_mae']:>10.4f}"
              f" {d['test_mae']:>10.4f}"
              f" {gap:>+8.4f}"
              f" {d['cv_mean']:>10.4f}"
              f" ±{d['cv_std']:.4f}     {status}")

    print()
    print("  Gap = Test MAE − Train MAE")
    print("  A large positive gap = overfitting. Both values high = underfitting.")
    print("  CV std > 0.02 = model is sensitive to which data it trains on.")


# =============================================================================
# SECTION 9.6: DETAILED ERROR DISTRIBUTION (Engineered — all models)
# =============================================================================
#
# This answers the practical question: "How close are the predictions?"
# We report several complementary perspectives:
#
#   Bias (mean signed error)
#     → Is the model systematically high or low? A model that always over-
#       predicts risk is more harmful than one that is symmetric in its errors.
#
#   Median absolute error
#     → Robust to outliers. The "typical" prediction is this far off.
#
#   Coverage percentages
#     → What fraction of predictions fall within ±0.5 / ±1.0 / ±1.5 units?
#       Remember: the RawScore range for Risk of Recidivism is ≈ 5.5 units,
#       and DecileScore bins are ≈ 0.5–0.8 units wide. So:
#         ±0.5  ≈  within ~1 decile bin   (very good)
#         ±1.0  ≈  within ~1–2 decile bins (acceptable)
#         ±1.5  ≈  within ~2–3 decile bins (rough estimate)
#
#   90th-percentile and max absolute error
#     → Characterise the *worst cases*. High-stakes decisions care more about
#       tail errors (the cases that are very wrong) than average errors.
#
#   MAE as % of score range
#     → Normalises the error to be interpretable regardless of scale.

print(f"\n\n{'=' * 70}")
print(" ERROR DISTRIBUTION — COMPOSITE FEATURE SET (all models)")
print("=" * 70)
print(" RawScore target range for Risk of Recidivism : "
      f"{y_test.min():.2f}  to  {y_test.max():.2f} "
      f"(span ≈ {y_test.max() - y_test.min():.2f} units)")
print(" DecileScore bins are ≈ 0.5–0.8 RawScore units wide")
print()

for model_name, pipeline in composite_pipelines.items():
    ed = error_distribution(pipeline, X_comp_test, y_test)
    bias_dir = "over-predicted" if ed["mean_signed_error"] > 0 else "under-predicted"

    print(f"  ── {model_name} ──")
    print(f"     Bias (mean signed error)      : "
          f"{ed['mean_signed_error']:+.4f}   model {bias_dir} on average")
    print(f"     Median absolute error         : {ed['median_abs_error']:.4f}")
    print(f"     MAE as % of score range       : {ed['mae_pct_of_range']:.1f} %")
    print(f"     Within ±0.5 score pts (~1 bin): {ed['pct_within_0.5']:>5.1f} %")
    print(f"     Within ±1.0 score pts (~2 bin): {ed['pct_within_1.0']:>5.1f} %")
    print(f"     Within ±1.5 score pts (~3 bin): {ed['pct_within_1.5']:>5.1f} %")
    print(f"     90th-percentile abs error     : {ed['p90_abs_error']:.4f}")
    print(f"     Max abs error (worst case)    : {ed['max_abs_error']:.4f}")
    print()


# =============================================================================
# =============================================================================

# SECTION 10: FAIRNESS / BIAS ANALYSIS
# =============================================================================
#
# We use the BEST performing model (Random Forest, engineered features) for the
# fairness analysis. Key question: does the model predict some ethnic groups'
# scores more accurately than others?
#
# A model with equal MAE across groups would be "well-calibrated" from a
# fairness perspective. Disparity in MAE means the model's errors are
# unevenly distributed — some groups get systematically worse predictions.
#
# IMPORTANT NUANCE: Even if the model tries to minimise average MAE, the data
# itself may contain more informative features for some groups than others,
# producing unavoidable disparities without any intentional discrimination.

print("\n\n" + "=" * 70)
print(" FAIRNESS ANALYSIS — Per-Ethnic-Group MAE (Composite, Random Forest)")
print("=" * 70)
print(" Lower MAE = model predicts that group's scores more accurately")
print(" Large differences across groups signal potential model bias\n")

best_pipeline = composite_pipelines["Random Forest (100)"]
group_errors  = per_group_mae(best_pipeline, X_comp_test, y_test, ethnic_test)

# Sort by MAE for readability
group_df = (
    pd.DataFrame(group_errors)
    .T
    .rename_axis("Ethnic Group")
    .sort_values("MAE")
)
print(group_df.to_string())

maes    = group_df["MAE"].astype(float)
best_g  = maes.idxmin()
worst_g = maes.idxmax()
disparity = round(float(maes.max() - maes.min()), 4)

print(f"\n  ✔ Best  calibrated: {best_g}  (MAE = {maes.min():.4f})")
print(f"  ✘ Worst calibrated: {worst_g}  (MAE = {maes.max():.4f})")
print(f"  → Raw MAE disparity: {disparity:.4f} score points")
print(f"    This means the model's average error is {disparity:.4f} RawScore")
print(f"    points larger for {worst_g} defendants than for {best_g}.")

# Show how disparity changed across all three feature sets
print("\n── Disparity across feature iterations ──")
for label, pipelines, X_test_variant in [
    ("Baseline  ", baseline_pipelines,   X_base_test),
    ("Engineered", engineered_pipelines, X_eng_test),
    ("Composite ", composite_pipelines,  X_comp_test),
]:
    ge = per_group_mae(pipelines["Random Forest (100)"],
                       X_test_variant, y_test, ethnic_test)
    m  = {g: v["MAE"] for g, v in ge.items()}
    disp = round(max(m.values()) - min(m.values()), 4)
    print(f"  {label} → MAE disparity across ethnic groups: {disp:.4f}")


# =============================================================================
# SECTION 11: FEATURE IMPORTANCE (Composite Random Forest)
# =============================================================================
#
# Impurity-based feature importance tells us which features the forest used
# most for its splits. High importance ≠ causal relationship. But it tells us:
#   - What information drives the model's predictions
#   - Whether race is being used implicitly (via Ethnic_Code_Text encoding)
#   - Whether composite signals add signal beyond engineered features
#   - Which composites are dead weight and can be dropped next iteration

print("\n\n" + "=" * 70)
print(" FEATURE IMPORTANCE — Random Forest (Composite Features)")
print("=" * 70)

# Retrieve the fitted preprocessor from the composite pipeline
prep_step = best_pipeline.named_steps["prep"]
ohe       = prep_step.named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(COMPOSITE_CAT).tolist()
all_names = cat_names + COMPOSITE_NUM

importances = best_pipeline.named_steps["model"].feature_importances_

imp_df = (
    pd.DataFrame({"Feature": all_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .reset_index(drop=True)
)

engineered_feature_keywords = [
    "age_sq", "male_x_young", "custody_legal",
]
composite_feature_keywords = [
    "risk_factor_count", "age_x_stability", "repeat_assessment",
]

def feature_origin(name):
    if any(kw in name for kw in composite_feature_keywords):
        return "◆ composite"
    if any(kw in name for kw in engineered_feature_keywords):
        return "★ engineered"
    return "  baseline"

# Show top 30 features and group by origin
print("\nTop 30 features (sorted by importance):\n")
print(f"  {'#':<4} {'Feature':<45} {'Importance':>10}  Origin")
print("  " + "-" * 70)
for i, row in imp_df.head(30).iterrows():
    print(f"  {i+1:<4} {row['Feature']:<45} {row['Importance']:>10.4f}  {feature_origin(row['Feature'])}")

# Summarise total importance by tier
comp_total = imp_df[imp_df["Feature"].apply(
    lambda f: any(kw in f for kw in composite_feature_keywords)
)]["Importance"].sum()
eng_total = imp_df[imp_df["Feature"].apply(
    lambda f: any(kw in f for kw in engineered_feature_keywords)
)]["Importance"].sum()
base_total = 1.0 - comp_total - eng_total

print(f"\n  Total importance — baseline features  : {base_total:.4f} ({base_total*100:.1f}%)")
print(f"  Total importance — engineered features : {eng_total:.4f}  ({eng_total*100:.1f}%)")
print(f"  Total importance — composite signals   : {comp_total:.4f}  ({comp_total*100:.1f}%)")

# Per-composite importance — tells us which to keep for iteration 2
print("\n  Composite signal breakdown (iteration 1 audit):")
print(f"  {'Signal':<25} {'Importance':>10}  Keep?")
print("  " + "-" * 45)
for kw in ["risk_factor_count", "age_x_stability", "repeat_assessment"]:
    kw_imp = imp_df[imp_df["Feature"] == kw]["Importance"]
    imp_val = float(kw_imp.iloc[0]) if len(kw_imp) > 0 else 0.0
    keep = "YES" if imp_val > 0.005 else "drop (< 0.005)"
    print(f"  {kw:<25} {imp_val:>10.4f}  {keep}")


# =============================================================================
# SECTION 12: ETHICAL CONSIDERATIONS
# =============================================================================

print("""

=============================================================================
 ETHICAL CONSIDERATIONS SUMMARY
=============================================================================

1. PREDICTING A BIASED GROUND TRUTH
   RawScore is the output of the proprietary COMPAS algorithm, which already
   embeds racial disparities (documented by ProPublica). Our models learn to
   predict COMPAS scores — not actual re-offending. Achieving a high R² means
   we are good at replicating COMPAS's biased outputs, not at predicting truth.

2. RACE AS AN EXPLICIT INPUT FEATURE
   We include Ethnic_Code_Text deliberately to (a) measure bias and (b) allow
   the model to encode race explicitly. In real deployment, using race in risk
   scoring is unconstitutional in many US jurisdictions. However, excluding it
   does not guarantee fairness — see point 3.

3. PROXY DISCRIMINATION VIA ENGINEERED FEATURES
   Several newly engineered features may act as proxies for race:
   - is_detained, is_pretrial: pretrial detention rates are higher for Black
     defendants due to unequal bail practices.
   - is_on_probation: probation rates reflect racially unequal sentencing.
   - custody_legal interaction: same structural issue.
   Engineering better predictive features can inadvertently amplify bias by
   giving the model more powerful proxy channels to race.

4. PER-GROUP ERROR DISPARITY (measured in Section 10)
   The MAE remains unequal across ethnic groups even after feature engineering.
   This means some groups get systematically better or worse predictions —
   which translates to unequal access to fair risk assessment in practice.

5. FEEDBACK LOOPS
   Any model deployed in a real justice system will generate new "ground truth"
   labels for future defendants. If those labels reflect the model's bias, the
   next generation of models trained on this data will inherit and amplify it.

6. LIMITED FEATURE SET
   COMPAS uses 137 features; we have ~20 effective signals. Our R² (~0.45) is
   expected to be lower. This gap means our model is less reliable — extra
   caution is warranted if it were to inform real decisions.

7. ADJUSTMENTS SINCE DELIVERABLE 1
   - Narrowed scope to Risk of Recidivism only (cleaner problem definition)
   - Added 11 engineered features with criminological justification
   - Introduced a direct before/after comparison to quantify value of engineering
   - Added per-group disparity tracking before vs. after engineering
   - Cleaned implausible age values (< 10 or > 100 years)
=============================================================================
""")

print("Script completed successfully.")
