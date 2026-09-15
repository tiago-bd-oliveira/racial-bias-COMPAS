# Racial Bias in the COMPAS Score

University project (LEI — IAA) auditing the **COMPAS** recidivism risk score for racial bias,
using the [ProPublica dataset](https://github.com/propublica/compas-analysis) of ~7,000 criminal
defendants from Broward County, Florida (2013–2014).

## What we do

Two modelling tracks, each trained twice — once **with** `race` as a feature and once **without** it
(race-blind) — so we can check whether hiding the protected attribute actually removes the disparity:

| Track | Target | Type |
| --- | --- | --- |
| `delivery2/score/` | `decile_score` (COMPAS 1–10 risk score) | regression |
| `delivery2/recid/` | `is_recid` (actual reoffence) | classification |

Main finding: the models reproduce ProPublica's result — African-American defendants who did not
reoffend are flagged as high risk far more often than Caucasian ones — and **dropping `race` barely
closes the gap**, because features like `priors_count` and age act as proxies for it.

## Repository layout

```
data/
  compas-scores-two-years.csv   raw ProPublica dataset
  compas_preprocessed.csv       output of the pre-processing notebook
delivery1/
  project_idea.ipynb            problem definition + initial exploratory analysis
delivery2/
  columns.md                    dataset column reference
  ethic-considerations.md       ethical reasoning behind feature selection + bias results
  pre_processing.ipynb          cleaning, feature engineering -> compas_preprocessed.csv
  score/                        regression on the COMPAS decile score
    model_testing.ipynb
    model_testing_no_race.ipynb
  recid/                        classification on actual recidivism
    recidivism_model.ipynb
    recidivism_model_no_race.ipynb
docs/
  report.pdf                    final report
  slides.pdf                    presentation slides
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run `delivery2/pre_processing.ipynb` first — it generates `data/compas_preprocessed.csv`, which every
model notebook reads. Trained models are written to `models/` (git-ignored).
