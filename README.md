# CrossModel BenchLab

### An Auditable Framework for Fair, Reproducible and Cost-Aware Machine Learning Benchmarking

**Team:** Coder Girls

**Hackathon:** AI Innovation Hackathon 2026

**Track:** 5 — Research & Publications

**Challenge:** 06 — CrossModel

## Team Members

- Nusrat Bhuiyan Nowrin
- Tania Tun Jannat

## Project Overview

CrossModel BenchLab is an experimental framework
for comparing classification models under documented
and reproducible benchmarking conditions.

It combines nested cross-validation,
protocol metadata checks, intentional violation
experiments and ranking sensitivity analysis.

## Research Questions

**RQ1:** Can automated audit rules detect
intentional protocol inconsistencies?

**RQ2:** Do the inconsistencies change
measured model performance or rankings?

## Dataset

Breast Cancer Wisconsin (Diagnostic)

- 569 samples
- 30 features
- Two diagnostic classes

Source:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Models

1. Logistic Regression
2. Random Forest
3. Support Vector Machine

## Evaluation Protocol

- 80/20 stratified development/test split
- Five outer folds repeated three times
- Three inner folds
- Ten hyperparameter candidates per model
- Primary metric: F1-score
- Fixed random seeds and shared outer partitions

## Controlled Benchmark Results

| Model | Mean F1 | Standard Deviation | Evaluations |
|---|---:|---:|---:|
| Logistic Regression | 0.9815 | 0.0078 | 15 |
| SVM | 0.9749 | 0.0087 | 15 |
| Random Forest | 0.9685 | 0.0111 | 15 |

## Fairness Audit

The metadata-based audit examines documented
partition consistency, preprocessing records,
tuning budgets, random seeds, metrics
and environment information.

Audit statuses:

- PASS
- FLAGGED
- NOT VERIFIABLE

## Intentional Violation Experiments

Two SVM diagnostic scenarios:

1. Unequal tuning budget: 10 to 2 configurations
2. Preprocessing leakage using full development data

The independent holdout partition was excluded
from both diagnostic conditions.

## RQ1: Audit Detection Results

- True Positives: 10
- False Negatives: 0
- False Positives: 0
- True Negatives: 65
- Detection Rate: 100.00%

These measurements describe the injected test cases,
not general-purpose leakage detection capability.

## RQ2: Ranking Sensitivity

- **Controlled:** Logistic Regression > SVM > Random Forest
- **Preprocessing Leakage:** Logistic Regression > SVM > Random Forest
- **Unequal Tuning:** Logistic Regression > Random Forest > SVM

Full numerical outputs are available in results/.

## Independent Holdout Results

- Selected Model: Logistic Regression
- Test Samples: 114
- F1: 0.9793
- Accuracy: 0.9737
- ROC-AUC: 0.9957

The positive class represents benign observations.
The benchmark is not intended for medical diagnosis.

## Technology Stack

Python, scikit-learn, pandas, NumPy,
Matplotlib, Plotly, Streamlit and Google Colab.

## Run the Dashboard

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app.py
```

The application loads saved data from results/.

## Repository Contents

- app.py — Streamlit application
- requirements.txt — Dashboard dependencies
- README.md — Project documentation
- results/ — Experiment CSVs, JSON and charts
- docs/methodology.md — Research method
- docs/research_findings.md — Actual findings
- CrossModel_BenchLab_Coder_Girls.ipynb —
  Experiment notebook (upload separately)

## Reproducibility

Open the supplied Google Colab notebook and
run the cells sequentially.

The recorded environment information and
experimental outputs are provided in results/.

## Limitations

- One public dataset and three classifier families
- Two diagnostic SVM protocol modifications
- Five folds for diagnostic comparisons
- Metadata audits cannot prove the absence of
  hidden code-level implementation errors
- Equal parameter counts do not imply equal compute
- Repeated folds are statistically correlated

## Future Work

- Additional public datasets
- More classifier families
- Code-level provenance verification
- Wider violation-injection experiments
- Hardware-aware computational budgets

## Development Attribution

ChatGPT was used to assist with code drafting,
debugging guidance, documentation and dashboard
development. Experimental results were generated
by executing the supplied research workflow.

## References

- UCI Machine Learning Repository
- scikit-learn documentation
- Streamlit documentation
- Related research described in the Stage 1 proposal
