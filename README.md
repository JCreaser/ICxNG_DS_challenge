# Forecasting task

**Input file required: "Forecasting dataset"**

**Suggested software: Python or Jupyter Notebook**

**Libraries required: Up to you**

Your task is to forecast a week of demand and import data based on a 3 month dataset of historic power flows.

Split the dataset so that you train on all of the data except for the final week, and hold that last week out as a test set. Produce a forecast for the test week, compare it against the actual recorded values, and report metrics (for example MAE, RMSE, and MAPE) that confirm how well the model performs.

Alongside your forecast data, create a methodology documenting the process. This can be in the form of an additional document, but well documented code in Python or a Jupyter notebook will suffice.

Things to consider:

- Is the data ready to be used for forecasting?
- If not, what factors might you want to consider checking in the data?
- Given the nature of the data, what additional datasets/external factors might you want to consider?

Hints:

- the Python toolbox called statsmodels has some good time series analysis tools
- but feel free to try something more out of the box like a recurrent neural network (RNN). 
