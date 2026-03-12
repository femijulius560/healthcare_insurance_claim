# Insurance Claim Prediction & Interactive Dashboard

## Overview

A machine learning pipeline that predicts healthcare insurance claims with an interactive Streamlit dashboard for exploring predictions and analyzing claim patterns.

## Features

- **Data Pipeline**: Cleaning, EDA, and feature engineering
- **Predictive Model**: Random Forest with bias correction and tail calibration
- **Interactive Dashboard**: Single prediction, Batch predictions, filtering, KPIs, and visualizations
- **Risk Detection**: Identifies high-risk patient profiles

## Dataset

This project uses healthcare insurance data from Kaggle. The dataset contains 1,340 records with demographic and health information to predict insurance claim costs.

**Features:**
- Age, BMI, Blood Pressure, Children, Gender, Smoking Status, Diabetes Status, Region

**Target Variable:** Insurance claim cost (continuous)

## Quick Start

### Installation

```bash
git clone <repo-url>
cd project
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Dashboard

```bash
streamlit run app.py
```

## Project Structure

```
project/
├── README.md
├── app.py
├── claim_analysis.ipynb
├── insurance_claim_raw.csv
├── insurance_claim_cleaned.csv
├── predictions_full_report.csv
├── rf_pipeline.pkl
├── requirements.txt
└── .gitignore
```

## Workflow

1. **Data Cleaning**: Remove outliers, handle missing values, feature engineering
2. **EDA**: Analyze claim patterns across demographics and health features
3. **Modeling**: Random Forest with log transformation and hyperparameter tuning
4. **Calibration**: Bias correction and tail calibration for extreme claims
5. **Dashboard**: Batch/single predictions, filtering, KPIs, and visualizations

## Technologies

Python, Pandas, NumPy, Scikit-Learn, Plotly, Streamlit, Joblib

## Key Insights

- Smokers have significantly higher claims
- High BMI correlates with increased claims
- Older age groups show rising costs

