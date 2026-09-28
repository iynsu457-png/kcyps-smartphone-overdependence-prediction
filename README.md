# Predicting next-wave high-risk smartphone overdependence in Korean adolescents

Analysis code for a longitudinal prediction study using the Korean Children and
Youth Panel Survey (KCYPS) 2018. Predictors measured at grade *t* estimate
high-risk Smartphone Addiction Proneness Scale (SAPS) classification at grade
*t+1*, across the Grade 7 to Grade 10 transitions.

Analytic sample: 10,972 adolescent-grade transitions from 4,634 adolescents, 455
features. Primary model XGBoost, held-out AUROC 0.819.

## Data

KCYPS 2018 is public-use data obtained by application from the National Youth
Policy Institute (<https://www.nypi.re.kr/archive>). Participant-level files are
not redistributed here. Place the m1 and e4 panels as

```
pipeline/01_data/raw/KCYPS2018m1[SPSS]/*.sav
pipeline/01_data/raw/KCYPS2018e4[SPSS]/*.sav
```

and run `pipeline/02_code/build_transitions.py` to construct the analytic file.

## Layout

```
pipeline/01_data     variable list, SAPS banding reference
pipeline/02_code     model pipeline, performance metrics, SHAP
pipeline/03_results  reported aggregate results
revision_R1/code     additional analyses requested in review
revision_R1/results  their aggregate outputs
```

`pipeline` is the analysis as first submitted: it builds the analytic file,
develops and evaluates the models, and produces the values in Tables 1 to 3.
`revision_R1` holds the analyses added in response to peer review, one script per
comment, covering calibration, precision-recall, outcome-spectrum sensitivity,
subgroup performance, incremental value over the SAPS score and the supplementary
tables. The two are kept apart so that the originally reported results stay
distinguishable from what was added later; nothing in `pipeline` was refitted.
Plotting code is not included; the figures are drawn from the result files here.

## Running

Python 3.11, `pip install -r requirements.txt`.

| Script | Output |
|---|---|
| `pipeline/02_code/build_transitions.py` | analytic file |
| `pipeline/02_code/ml_pipeline_final.py` | six classifiers, hyperparameter search, thresholds, held-out predictions |
| `pipeline/02_code/extract_table2.py` | Table 1 |
| `pipeline/02_code/recompute_metrics_cluster_bootstrap.py` | Tables 2 and 3 |
| `pipeline/02_code/recompute_shap_xgboost.py` | SHAP values |

`revision_R1/code` holds one script per review analysis: outcome-spectrum
sensitivity, model and threshold selection, calibration and decision curves,
precision-recall, incremental value over the SAPS score, data provenance,
encoding, dimensionality reduction, sample size, learning curves, grouped
importance, subgroup performance and participant flow.

Seeds are fixed at 42. Results computed from stored predictions are
deterministic; analyses that refit a model can differ slightly across library
versions.

## Ethics

Reviewed by the Institutional Review Board of Daejeon University and determined
exempt as secondary analysis of publicly available de-identified data
(IRB No. 1040647-202604-HR-001-03).

## Licence

MIT, see `LICENSE`. KCYPS 2018 data remains subject to the terms of the National
Youth Policy Institute.
