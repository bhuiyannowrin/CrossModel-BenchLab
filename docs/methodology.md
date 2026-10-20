# Research Methodology

## Project Information

Project: CrossModel BenchLab

Team: Coder Girls

Track: Research & Publications
Challenge: CrossModel — Benchmarking Multiple Models Fairly

## Dataset

Breast Cancer Wisconsin (Diagnostic)

- 569 samples
- 30 numerical input features
- Binary classification
- Class 0: Malignant
- Class 1: Benign

Dataset source:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## Data Partitioning

An 80/20 stratified split divides the data into
development and independent holdout partitions.

The independent test data is reserved for evaluation
after model-family selection and final tuning.

## Machine Learning Models

1. Logistic Regression
2. Random Forest
3. Support Vector Machine

## Nested Cross Validation

- Five outer stratified folds
- Three outer repeats
- Three inner stratified folds
- Ten candidate configurations per model
- Primary scoring metric: F1-score
- Fixed random seeds
- Identical outer partitions for all models

Logistic Regression and SVM use StandardScaler
inside model pipelines.

## Fairness Audit

The baseline audit examines recorded information
about:

- Partition signatures
- Preprocessing pipeline metadata
- Hyperparameter tuning counts
- Random seed records
- Scoring metric
- Environment metadata availability

Audit decisions include PASS, FLAGGED
and NOT VERIFIABLE.

A metadata PASS does not independently prove
the absence of hidden implementation problems.

## Diagnostic Protocol Violations

Two deliberate SVM experiments were conducted
using the first five outer folds.

### Experiment A: Unequal Tuning

Reduce the SVM tuning budget from ten candidate
configurations to two.

### Experiment B: Preprocessing Leakage

Fit the StandardScaler on the entire development
partition before cross-validation.

This deliberately introduces validation information
into preprocessing.

The independent holdout partition is excluded.

## Research Questions

RQ1: Can specified metadata audit checks detect
the injected protocol inconsistencies?

RQ2: How do the inconsistencies affect F1-score
and model rankings?

## Independent Holdout

The selected model family is chosen using
controlled nested cross-validation results.

A final three-fold GridSearchCV is fitted on
development data only.

The selected estimator is evaluated once on
the independent holdout partition.

## Limitations

- One public dataset
- Three classifier families
- Two altered SVM scenarios
- Five diagnostic outer folds
- Metadata-based checks
- Repeated folds are correlated
- Equal candidate counts do not guarantee
  equal computational effort

This project is a benchmarking research prototype,
not a clinical diagnostic system.
