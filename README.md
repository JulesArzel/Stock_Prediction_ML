# Financial Time Series Prediction

## Overview
This project focuses on predicting financial time series returns using deep learning techniques.  
The goal is to build a model that forecasts the **future daily return** \( R_{t+1} \) of a financial index based on historical prices and engineered features.

The dataset used is the **CAC40 historical closing prices**, which can be downloaded [here](https://turinici.com/wp-content/uploads/cours/common/close_cac40_historical.csv).

---

## Problem Description
Given a time series of stock prices \( S_t \), the task is to predict the logarithmic return of the next day:

\[
R_t = \log\left(\frac{S_t}{S_{t-1}}\right), \quad \text{Target: } R_{t+1}
\]

The dataset is divided into:
- **Training set**: 80% of the data
- **Validation set**: 20% of the data

---

## Feature Engineering
For each time step \( t \), the following features are computed:

1. **Logarithmic returns**  
2. **Rolling averages of returns** for horizons:  
   - 5 days (1 week)  
   - 25 days (1 month)  
   - 75 days (3 months)  
   - 150 days (6 months)  
   - 255 days (1 year)  
   - 765 days (3 years)  
3. **Annualized volatility** for the same horizons  
4. **Exponential Moving Averages (EMA)** of prices normalized by the current price  
5. **Sliding window of past prices** over a 3-month horizon

The final dataset contains:
- 15 statistical features (returns and volatilities)  
- 1 current price \( S_t \)  
- 7 EMA ratios  
- A sliding window of \( n \) past prices  
- **Target:** next-day return \( R_{t+1} \)

---

## Methodology
1. **Data preprocessing**  
   - Load and clean dataset (remove missing values, convert dates)  
   - Compute returns, moving averages, volatilities, EMAs  
   - Construct sliding window features  
   - Normalize all features using z-score standardization  

2. **Visualization**  
   - Price evolution with EMA overlays  
   - Rolling volatilities  
   - Distribution of log returns  
   - Sliding window trends  

3. **Modeling**  
   - Build and train deep learning models (MLP, LSTM, etc.)  
   - Train on 80% of the dataset, validate on 20%  
   - Evaluate prediction performance  

---

## Requirements
The project is implemented in **Python** with the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow` / `keras`

---

## How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
