# Predicting next-wave high-risk smartphone overdependence in Korean adolescents

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23131942.svg)](https://doi.org/10.5281/zenodo.23131942)

Analysis code for a longitudinal prediction study using the Korean Children and
Youth Panel Survey (KCYPS) 2018. Predictors measured at grade *t* estimate
high-risk Smartphone Addiction Proneness Scale (SAPS) classification at grade
*t+1*, across the Grade 7 to Grade 10 transitions.

Analytic sample: 10,972 adolescent-grade transitions from 4,634 adolescents, 455
features. Primary model XGBoost, held-out AUROC 0.819.

## Checking the reported values

The aggregate results behind every number in the paper are in the repository, so
the reported values can be read off without obtaining the data or running
anything:

| File | Contains |
|---|---|
| `pipeline/03_results/cluster_bootstrap_results.pkl` | Tables 2 and 3: discrimination and threshold metrics with cluster-bootstrap confidence intervals |
| `pipeline/03_results/table2_demographics.json` | Table 1 |
| `pipeline/03_results/shap_top25_xgboost_aggregated_filtered.csv` | the SHAP hierarchy reported in the Results |
| `pipeline/03_results/model_performance.csv`, `transition_performance.csv` | per-classifier and per-transition metrics |
| `revision_R1/results/*.csv` | the analyses added in review, one file per analysis |

Re-deriving these from the raw data requires the KCYPS panels; see below.

## Data

KCYPS 2018 is public-use data obtained by application from the National Youth
Policy Institute (<https://www.nypi.re.kr/archive>).

**No individual-level records are distributed here.** The raw panels, the analytic
file (`transitions.pkl`) and the held-out predictions (`test_predictions.csv`) are
all individual-level records derived from KCYPS 2018. The National Youth Policy
Institute holds the copyright in the data and prohibits transferring it or copying
any part of it into separate files, so none of them are redistributed.

To run the pipeline, obtain the m1 and e4 panels and place them as

```
pipeline/01_data/raw/KCYPS2018m1[SPSS]/*.sav
pipeline/01_data/raw/KCYPS2018e4[SPSS]/*.sav
```

then run, in order:

1. `pipeline/02_code/build_transitions.py` — builds the analytic file and verifies
   the participant flow.
2. `pipeline/02_code/ml_pipeline_final.py` — fits the six classifiers, runs the
   hyperparameter search, fixes the thresholds and writes the held-out
   predictions.

Every other script reads one of those two outputs.

## Layout

```
pipeline/01_data     feature list, SAPS banding counts by grade
pipeline/02_code     model pipeline, performance metrics, SHAP, figures
pipeline/03_results  reported aggregate results
revision_R1/code     additional analyses requested in review
revision_R1/results  their aggregate outputs
```

`pipeline` is the analysis as first submitted: it builds the analytic file,
develops and evaluates the models, and produces the values in Tables 1 to 3 and
Figures 1 to 5. `revision_R1` holds the analyses added in response to peer review,
one script per comment, and produces Figures 6 to 8, Supplementary Figure S1 and
the supplementary tables. The two are kept apart so that the originally reported
results stay distinguishable from what was added later; nothing in `pipeline` was
refitted.

Scripts that assemble the manuscript documents are not included.

## Running

Python 3.11.4, `pip install -r requirements.txt`. The pinned versions are those
used for the reported results.

### Primary analysis

| Script | Output |
|---|---|
| `build_transitions.py` | analytic file |
| `ml_pipeline_final.py` | six classifiers, hyperparameter search, thresholds, held-out predictions |
| `extract_table2.py` | Table 1 |
| `recompute_metrics_cluster_bootstrap.py` | Tables 2 and 3 |
| `recompute_shap_xgboost.py` | SHAP values, Figures 4 and 5 |
| `generate_figure1.py` | Figure 1 |
| `make_figure2_roc.py` | Figure 2 |
| `generate_figure4_pertransition.py` | Figure 3 |

### Review analyses

One script per comment, covering outcome-spectrum sensitivity, model and threshold
selection, calibration and decision curves, precision-recall, incremental value
over the SAPS score, data provenance, encoding, dimensionality reduction, sample
size, learning curves, grouped importance, subgroup performance and participant
flow. Figures are drawn by `make_figure1_v2.py` (Figure 1), `make_figure3_v2.py`
(Figure 3), `make_figures45_v2.py` (Figures 4 and 5), `performance_calibration.py`
(Figure 6), `make_figure7.py` (Figure 7), `pr_analysis.py` (Figure 8) and
`make_figure_s1_lc.py` (Supplementary Figure S1).

Three scripts write intermediates that others read, so run them first:

| Run first | Needed by |
|---|---|
| `build_fullspectrum.py` | `sensitivity_fullspectrum.py`, `three_class_analysis.py`, `pr_analysis.py` |
| `performance_calibration.py` | `subgroup_fairness.py`, `transition_uncertainty.py`, `dim_reduction.py`, `sample_size_folds.py` |
| `sensitivity_fullspectrum.py` | `pr_analysis.py`, `three_class_analysis.py` |

The remaining scripts are independent.

### Reproducibility

Seeds are fixed at 42. Results computed from the stored predictions are
deterministic. Analyses that refit a model can differ slightly across library
versions, because SMOTE+ENN nearest-neighbour ties and gradient-boosting
implementations are not stable between releases, which is why the versions are
pinned.

## Ethics

Reviewed by the Institutional Review Board of Daejeon University and determined
exempt as secondary analysis of publicly available de-identified data
(IRB No. 1040647-202604-HR-001-03).

## Citation

Song D, Lee Y-S, Youn B-Y. Analysis code for predicting next-wave high-risk
smartphone overdependence in Korean adolescents (KCYPS 2018). Zenodo.
<https://doi.org/10.5281/zenodo.23131942>

The DOI above always resolves to the latest version. Machine-readable metadata is
in `CITATION.cff`.

## Licence

MIT, see `LICENSE`. KCYPS 2018 data remains subject to the terms of the National
Youth Policy Institute.
