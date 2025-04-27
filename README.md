# Google-Stock-Price-with-LSTM-using-PyTorch


## Overview

This project demonstrates how to build, train, and evaluate LSTM-based neural networks in PyTorch to predict Google’s daily closing stock price. Using historical data from 2018 to 2025, we explore three different LSTM architectures—ranging from a simple single‐layer model to a more complex multi‐layer, dropout‐regularized network—and compare their performance. Finally, we generate a 10‐day forecast beyond the available dataset to illustrate the model’s ability to anticipate near‐term price movements.

## Project Flow

1. **Data Loading & Cleaning**  
   - Read historical Google stock prices (`.csv` with Date, Open, High, Low, Close, Volume).  
   - Parse dates, handle missing values (if any), and index by date.

2. **Exploratory Data Analysis & Preprocessing**  
   - Display dataset info and summary statistics.  
   - Plot time series of the closing price.  
   - Compute and visualize feature correlations (e.g. Close vs. Volume).  
   - Normalize the closing‐price series (MinMax scaling).  
   - Generate sliding windows of length *T* (e.g. 60 days) to form `(X, y)` training sequences.

3. **Model Definition**  
   - **Model 1 (Simple):** One LSTM layer, small hidden size, no dropout.  
   - **Model 2 (Medium):** Two LSTM layers, moderate hidden size, dropout between layers.  
   - **Model 3 (Complex):** Two LSTM layers, larger hidden size, higher dropout for regularization.

4. **Training Setup**  
   - Use MSE loss and Adam optimizer.  
   - Train each model for a fixed number of epochs, logging epoch‐wise training loss.  
   - Leverage GPU if available for faster training.

5. **Making Predictions**  
   - Feed test‐set sequences into each trained model.  
   - Invert normalization to obtain predicted prices in original scale.

6. **Results Visualization**  
   - Plot actual vs. predicted closing prices for each of the three models on the test set.  
   - Compare prediction curves side‐by‐side to assess which architecture best captures the price dynamics.

7. **10-Day Forecasting**  
   - Using the best‐performing model, iteratively forecast the next 10 days beyond the available data.  
   - Plot and tabulate these forecasted prices.

## Key Features

- **Flexible LSTM Architectures**  
  Easily swap between single‐layer and multi‐layer LSTMs, adjust hidden sizes and dropout rates to control model capacity and overfitting.

- **End-to-End PyTorch Implementation**  
  Leverages PyTorch’s `nn.LSTM`, custom `Dataset`/`DataLoader`, GPU acceleration, and training loop with clear logging.

- **Data Normalization & Windowing**  
  Automates MinMax scaling of the target series and creation of rolling windows—critical for stable RNN training on time‐series data.

- **Comprehensive EDA**  
  Uses `pandas`, `matplotlib`, and `seaborn` to explore data distributions, trends, and feature correlations before modeling.

- **Visualization of Results**  
  Side‐by‐side comparison plots make it easy to see which model most closely follows the true closing price and where it deviates.

- **Future Price Forecasting**  
  Demonstrates how to extend the model beyond test data for practical forecasting applications.

## Results

- **Training Loss:**  
  All three models converge, with the more complex architectures achieving lower MSE on the training set.

- **Test‐Set Performance:**  
  - Model 1 (Simple) captures the broad trend but lags on sharp movements.  
  - Model 2 (Medium) better tracks mid‐term fluctuations.  
  - Model 3 (Complex) yields the tightest fit to the actual closing price, with the lowest test‐set MSE.

- **10-Day Forecast:**  
  The chosen best model projects the next 10 days of closing prices, revealing a moderate upward/downward trend (see “Forecast” plot and table in the notebook for exact values).

*(For detailed loss curves, numeric metrics, and interactive plots, please refer to the Jupyter notebook: `google-stock-price-lstm-using-pytorch.ipynb`.)*  
