# Research Findings

## CrossModel BenchLab

Team Coder Girls

## Controlled Benchmark

| Model | Mean F1 | Standard Deviation | Evaluations |
|---|---:|---:|---:|
| Logistic Regression | 0.9815 | 0.0078 | 15 |
| SVM | 0.9749 | 0.0087 | 15 |
| Random Forest | 0.9685 | 0.0111 | 15 |

These values are computed from the controlled
nested cross-validation results.

## RQ1: Audit Detection

- True Positives: 10
- False Negatives: 0
- False Positives: 0
- True Negatives: 65
- Detection Rate: 100.00%

This rate applies only to the documented
injected protocol deviations.

## RQ2: Ranking Sensitivity

- **Controlled:** Logistic Regression > SVM > Random Forest
- **Preprocessing Leakage:** Logistic Regression > SVM > Random Forest
- **Unequal Tuning:** Logistic Regression > Random Forest > SVM

See these output files for more information:

- ranking_comparison.csv
- paired_f1_differences.csv
- fold_ranking_changes.csv

## Independent Holdout Evaluation

- Selected Model: Logistic Regression
- Test Samples: 114
- Accuracy: 0.9737
- Precision: 0.9726
- Recall: 0.9861
- F1-score: 0.9793
- ROC-AUC: 0.9957

The positive class (1) represents benign cases.

## Interpretation

The results illustrate the performance and
audit outcomes observed under the tested
benchmark protocols.

An injected protocol violation need not
produce a measurable ranking change.

## Limitations

These results are limited to one public dataset,
three classifier families and two deliberately
altered SVM scenarios.

The metadata audit does not detect every possible
form of data leakage or implementation error.
