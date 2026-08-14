# KINESSO Data Science Analysis

Academic project analysing customer propensity modelling in a marketing context.

## Datasets
- UCI Online Shoppers Purchasing Intention Dataset
- UCI Bank Marketing Dataset

## Models
- Decision Tree
- XGBoost

## Method
- Stratified 80:20 train-test split
- Class weighting for imbalanced targets
- One-hot encoding of categorical variables
- Duplicate removal for Online Shoppers
- `duration` excluded from the primary Bank Marketing model because it is only known after a completed contact
- Evaluation with precision, recall, F1-score and ROC-AUC

## Final results

| Dataset | Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Online Shoppers | Decision Tree | 0.532 | 0.817 | 0.644 | 0.908 |
| Online Shoppers | XGBoost | 0.558 | 0.843 | 0.672 | 0.935 |
| Bank Marketing | Decision Tree | 0.360 | 0.552 | 0.435 | 0.769 |
| Bank Marketing | XGBoost | 0.348 | 0.646 | 0.452 | 0.804 |

## Key findings
`PageValues` was the strongest XGBoost predictor for Online Shoppers.
`poutcome_success` was the strongest XGBoost predictor for Bank Marketing.

## Structure
```text
data/       Dataset instructions
figures/    Model figures
notebooks/  Reproducible notebook
src/        Python analysis script
```

## Data sources
- https://archive.ics.uci.edu/dataset/468/online
- https://archive.ics.uci.edu/dataset/222/bank+marketing

Raw datasets are not included. Download them from the original sources and place
them in `data/` using the filenames documented in `data/README.md`.
