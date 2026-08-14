# kinesso-data-science-analysis
# Data

This project uses two publicly available datasets:

1. UCI Online Shoppers Purchasing Intention Dataset
   https://archive.ics.uci.edu/dataset/468/online

2. UCI Bank Marketing Dataset
   https://archive.ics.uci.edu/dataset/222/bank+marketing

The datasets are not included in this repository. Please obtain them from
their original sources before running the analysis.
# KINESSO Data Science Analysis

## Overview

This project investigates machine learning approaches to customer
propensity modelling in marketing, based on the Data Scientist role
advertised by KINESSO in Melbourne.

Two publicly available datasets are analysed:

- UCI Online Shoppers Purchasing Intention Dataset
- UCI Bank Marketing Dataset

## Objectives

The analysis aims to:

- predict customer purchase or response propensity;
- compare Decision Tree and XGBoost classifiers;
- identify important predictive factors;
- evaluate model performance using precision, recall, F1-score and ROC-AUC;
- compare insights across two marketing contexts.

## Methodology

The analysis uses:

- stratified 80:20 train-test splitting;
- categorical feature encoding;
- class weighting to address class imbalance;
- Decision Tree classification;
- XGBoost classification.

For the Bank Marketing analysis, `duration` was excluded from the primary
model because it represents the duration of a completed customer contact
and would not be available for pre-contact targeting.

## Key Findings

XGBoost achieved the strongest overall performance across both datasets.

For Online Shoppers, XGBoost achieved:

- F1-score: 0.672
- ROC-AUC: 0.935
- Recall: 0.843

For Bank Marketing, XGBoost achieved:

- F1-score: 0.452
- ROC-AUC: 0.804
- Recall: 0.646

The most influential predictor in the Online Shoppers model was
`PageValues`, while `poutcome_success` was the strongest predictor in the
Bank Marketing model.

## Repository Structure

```text
data/       Dataset information
figures/    Generated visualisations
notebooks/  Analysis notebook
src/        Python source code
