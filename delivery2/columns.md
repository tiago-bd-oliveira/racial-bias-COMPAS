# COMPAS Dataset Column Descriptions

> [!IMPORTANT]
> This file was generated using AI for better understanding of the dataset.

This document provides a detailed reference for the variables found in the ProPublica COMPAS dataset. This dataset is a cornerstone for research into algorithmic fairness, specifically exploring the intersection of predictive modeling and criminal justice.

## 1. Identifiers & Demographics
These columns describe the individual's basic background.
- **id**: A unique record identifier.
- **name / first / last**: The full name of the defendant.
- **sex**: The recorded gender of the defendant.
- **dob**: Date of Birth.
- **age**: The defendant's age at the time of the COMPAS assessment (Continuous variable).
- **age_cat**: Age category (e.g., <25, 25-45, >45).
- **race**: The self-reported or recorded race of the defendant. Used primarily for fairness auditing.

## 2. Criminal History (Priors)
These features track history prior to the current arrest.
- **priors_count**: Total number of prior adult criminal charges/convictions.
- **juv_fel_count**: Count of prior juvenile felony offenses.
- **juv_misd_count**: Count of prior juvenile misdemeanor offenses.
- **juv_other_count**: Count of prior juvenile offenses that were not felonies or misdemeanors.

## 3. Current Offense (Prefix `c_`)
Describes the specific crime that triggered the current COMPAS assessment.
- **c_case_number**: Case identifier for the current charge.
- **c_charge_desc**: Text description of the current offense (e.g., "Grand Theft").
- **c_charge_degree**: Severity level of the charge (F = Felony, M = Misdemeanor).
- **c_offense_date**: The date the current crime was committed.
- **c_arrest_date**: The date the defendant was arrested for this crime.
- **c_days_from_compas**: Number of days between the arrest/offense and the assessment date.
- **c_jail_in / c_jail_out**: Timestamps for when the defendant entered and exited jail for this arrest.

## 4. Algorithm Scores (The "Proxy Labels")
Outputs from the COMPAS algorithm.
- **decile_score**: A 1-10 risk ranking (1 = Low Risk, 10 = High Risk). Used as the regression target.
- **score_text**: Qualitative translation of the decile score (Low, Medium, High).
- **type_of_assessment**: The specific assessment model used (General Recidivism).
- **v_decile_score / v_score_text**: Risk scores specifically for *Violent* recidivism.
- **v_type_of_assessment**: Indicates the assessment type for the violent risk score.

## 5. Recidivism Outcomes (Ground Truth)
These flags track if the person actually re-offended after the initial assessment.
- **is_recid**: Binary flag indicating any subsequent arrest.
- **two_year_recid**: The primary target for fairness analysis; indicates if a new crime was committed within a 2-year window.
- **violent_recid / is_violent_recid**: Indicators if the new crime was a violent offense.
- **r_case_number / r_charge_desc / r_charge_degree**: Details regarding the recidivism (new) offense.
- **r_offense_date / r_jail_in / r_jail_out**: Timestamps related to the recidivism event.

## 6. Temporal & Survival Analysis
Used to account for time spent in the community vs. time incarcerated.
- **in_custody / out_custody**: General entry/exit dates for custody.
- **start / end**: The start and end days of the "at-risk" observation window.
- **event**: A binary flag for survival analysis (1 = recidivism occurred, 0 = observation ended).

---
*Note: Some columns like `priors_count` or `decile_score` may appear twice in the raw data due to database join operations during dataset construction.*


