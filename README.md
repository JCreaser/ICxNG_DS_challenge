# Forecasting task

**Input file: `data/Forecasting_dataset.csv`**

**Software: Python + Jupyter Notebook**

**Libraries: `requirements.txt`**

Forecast a week of electricity demand and import from about three months of historic power-flow data.. The work is split into two guided notebooks. Run them in order, from the project folder.

## Task 1: Preparing the data (`task_1_preprocess.ipynb`)

Turn the raw data into a clean hourly time series ready for forecasting.

## Task 2: Forecasting (`task_2_analysis.ipynb`)

Train on demand and import data except the final week in the dataset, then forecast that week and compare it with the ground truth.

## How the notebooks work

- `# TODO` cells need a little code
- **Check** cells test your work
- **Question** cells ask for a short written answer

Helper functions for Task 2 are in `src/task_helpers.py`.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```
