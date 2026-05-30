# Ethical Considerations

Ethics is a big part of this project. Given the fact that our study aims to bring racial bias to light, it's important that we follow very strict ethical practices.

## Ethics in Feature Selection

In terms of features for the training of our first model, we focused purely on the defendants personal features and criminal record. We did not take into account any information that is not a personal characteristic of the defendant. 

We also had to take into account that the COMPAS decile score is calculated before a sentence is given. Because of this, we did not take into account any information linked to events that happen after the COMPAS screening where the score was attributed to the subject. 

Here are some examples of those decisions:

- **Exclude jail time information**: As mentioned before, the score is calculated before the sentence is given, to help the judge make a better decision. So it is not a cause for the score, instead it is a consequence of it.

- **Not measuring arrest time**: Since we had both the offense date and the arrest date, we could have included the time it took for the subject to get arrested. However, we decided **not to** because that would have more to do with police efficiency than the actual danger of the person.

- **Derive more criminal record information**: We decided to derive a lot of metrics from the original criminal statistics, to test some bias hypothesis. It is easier to prove the bias if correlation to the final score is bigger for the race variable than for the criminal flags like `velocity_repeat_offender`.

## Scope of our Ethics

This project aims to expose the racial bias of the COMPAS score. However the score is only calculated when the subject is arrested. There is already a level of bias on the decision to arrest someone, by part of the police. What we are trying to say is that it is hard to mitigate the bias when the existence of the records in itself could happen due to bias.

However this is something we cannot control, and is totally out of our scope. We fully accept the risk that bias may persist simply through the fact that some criminals don't get arrested, and some people do unjustly.

## Recidivism Model: False Positive Rates by Race

Our main analysis treats the COMPAS `decile_score` as the prediction target. To gauge how
that score relates to **real** outcomes, we trained a second model (a Random Forest classifier,
see `recid/recidivism_model.ipynb`) that predicts the actual recidivism flag `is_recid` instead.
Because the score itself is what we are auditing, `decile_score` is excluded from this model's
features.

The relevant fairness metric here is the **false positive rate (FPR)**: among defendants who did
**not** reoffend, the share the model wrongly flagged as future recidivists. A group with a higher
FPR is unfairly burdened — its innocent members are labelled "high risk" more often.

| Race | Base recid. rate | FPR (with race) | FPR (no race) |
| --- | --- | --- | --- |
| African-American | 0.558 | **0.358** | 0.328 |
| Hispanic | 0.410 | 0.288 | 0.305 |
| Caucasian | 0.428 | **0.157** | 0.180 |
| Other | 0.337 | 0.073 | 0.091 |

Two observations stand out:

- **The disparity mirrors ProPublica's original COMPAS finding.** African-American defendants who
  did not reoffend are wrongly flagged roughly **2.3× more often** than Caucasian defendants
  (0.358 vs 0.157). The error is also directional: the false *negative* rate inverts the pattern
  (Caucasians who *did* reoffend are missed far more often, ~0.43 vs ~0.23). A model fit on actual
  outcomes reproduces exactly the asymmetry COMPAS was criticised for — over-predicting risk for
  Black defendants and under-predicting it for white ones.

- **Removing race barely changes the gap** (FPR ratio drops only from ~2.3× to ~1.8×). Since the
  classifier never sees race in the "no race" version, the remaining disparity must flow through
  correlated features such as `priors_count` and `age_at_event`, which act as **proxies** for race
  due to upstream differences in policing and arrest rates. This is direct evidence that simply
  hiding the protected attribute does **not** make the model fair.

**Implication for COMPAS.** Because COMPAS's own decile score correlates strongly with the same
features, the score inherits this skew relative to real outcomes: it systematically rates
non-reoffending Black defendants as higher-risk than non-reoffending white defendants. The bias is
therefore not a quirk of one modelling choice — it is baked into the relationship between the
available features and the recorded outcomes, which is precisely why it persists even when race is
withheld.