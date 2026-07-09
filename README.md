# Predictive Maintenance: Machine Failure Prediction

Exploratory data analysis and modeling groundwork for predicting industrial machine failures, built on three well-known predictive maintenance datasets. The project's primary focus is the **AI4I 2020 dataset**, supplemented by exploration of the **Microsoft Azure Predictive Maintenance** dataset and the **NASA Bearing Run-to-Failure (C-MAPS)** dataset.

The end goal is to develop machine learning models that can predict machine failures from operating conditions and sensor readings, enabling proactive maintenance instead of reactive repairs.

## Overview

- **Type:** Data science / exploratory data analysis project
- **Language:** Python (Jupyter Notebooks)
- **Runtime:** Jupyter Notebooks with the standard scientific Python stack
- **Key libraries:** `pandas`, `scikit-learn`, `seaborn`, `matplotlib`, `kagglehub`

## Datasets

| Dataset | Description | Status |
|---|---|---|
| **AI4I 2020** | 10,000 records, binary failure classification with 5 failure modes (TWF, HDF, PWF, OSF, RNF) | Primary — fully explored |
| **Microsoft Azure Predictive Maintenance** | Telemetry-based failure prediction dataset | Setup / exploration phase |
| **NASA Bearing (C-MAPS)** | Run-to-failure bearing sensor data | Exploratory |

## Repository Structure

```
notebooks/
  AI4I-eda-reorganized.ipynb         Primary EDA: 10,000 records, binary classification
  microsoft-azure-...eda.ipynb       Azure dataset exploration (setup phase)
  nasa-cmaps-eda.ipynb               NASA bearing C-MAPS dataset exploration

data/
  ai4i-final-csv.csv                 Processed AI4I dataset (875 KB)

dashboards/
  ai4i_dashboard.pbix                Power BI visualization dashboard

db/
  predictive_maintenance              SQLite database file

images/
  dashboard.png                      Screenshot of Power BI dashboard
  scatter.png                        Visualization output

.gitignore                           Ignores Python virtual environment
```

## Getting Started

### Install dependencies

```bash
pip install pandas seaborn matplotlib numpy scikit-learn kagglehub
```

### Run the primary notebook

```bash
jupyter notebook notebooks/AI4I-eda-reorganized.ipynb
```

> **Note:** The notebook automatically downloads the AI4I 2020 dataset from Kaggle on first run. This requires `kagglehub` authentication to be configured beforehand.

## What the AI4I Notebook Covers

1. **Data loading and inspection** — initial look at the raw dataset structure
2. **Statistical summary and class imbalance analysis** — the target variable is highly imbalanced (~3.39% positive class)
3. **Correlation matrix heatmap** — relationships between sensor and process variables
4. **Failure type distribution analysis** — breakdown across the five failure modes: TWF, HDF, PWF, OSF, RNF
5. **Failure patterns by product quality type** — comparison across quality tiers H (high), M (medium), and L (low)

## Dashboards

`dashboards/ai4i_dashboard.pbix` is a Power BI dashboard visualizing key insights derived from the processed AI4I dataset. A screenshot is available at `images/dashboard.png`.
