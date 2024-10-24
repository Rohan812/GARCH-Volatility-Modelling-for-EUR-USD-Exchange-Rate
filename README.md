# 🌐 GARCH-Volatility Modelling for EUR/USD Exchange Rate 📊

Welcome to the **GARCH Volatility Modelling** project, where we explore the dynamics of the **EUR/USD exchange rate** using the **GARCH(p,q)** model! 🌍📈

In this project, we:
- 📥 Download historical data for the EUR/USD exchange rate.
- 🧮 Compute **log returns** for the exchange rate.
- ⚙️ Fit a **GARCH(p,q)** model to log returns.
- 🎯 Use **Akaike Information Criterion (AIC)** and **Bayesian Information Criterion (BIC)** to determine the optimal parameters `(p, q)`.
- 🔮 Predict future log returns using the optimal model and compare them with actual log returns for accuracy.

---

## 1️⃣ Data Collection - EUR/USD Exchange Rate Data

We use the **Yahoo Finance** library to retrieve data for the EUR/USD exchange rate from **2019-01-01** to **2023-12-31**.

```python
# Downloading the historical exchange rate data
data = yf.download('EURUSD=X', start='2019-01-01', end='2023-12-31')
data.head(10)
```
![Data Head](https://github.com/user-attachments/assets/1552198a-fb36-49ae-b26e-ae7e97c55113)

# Daily Log Returns Calculation

This section describes how to calculate the daily log returns for a stock's adjusted closing prices.

```python
df['Daily Log Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
df.head(10)
log_returns = df['Daily Log Returns'].dropna()
log_returns
```

![Log Returns](https://github.com/user-attachments/assets/5a0696e1-cb57-4117-85d0-b1158d4db4ea)

# Interactive Log Returns vs. Date

This section presents an interactive graph of daily log returns plotted against the date.

![Log Returns EUR-USD Exch rate](https://github.com/user-attachments/assets/7d87d6d1-8673-4ceb-93bc-91d4ebb54c15)

# Distribution of Log Returns

This section includes histograms to visualize the distribution of daily log returns, checking the assumptions of the GARCH model.

## Histogram with Normal Distribution Line

This histogram displays the log returns with a normal distribution line traced on the histogram.

![Histogram of Log Returns (Normal Dist Fitted)](https://github.com/user-attachments/assets/b371c480-a36e-40dd-8bae-cb008515e534)

## Histogram with T-Distribution Line

This histogram shows the log returns with a t-distribution line traced on the histogram.

![Histogram of Log Returns (t-Dist Fitted)](https://github.com/user-attachments/assets/218f9d2f-e321-40aa-9b50-0f39d1352f92)

# Q-Q Plots of Log Returns

This section includes Q-Q plots to assess the distribution of daily log returns. The first plot compares the log returns to a normal distribution, while the second plot compares the log returns to a t-distribution.

## Q-Q Plot with Normal Distribution

This Q-Q plot displays the log returns against the theoretical quantiles of the normal distribution.

![Q-Q Plot for checking Normal Distribution](https://github.com/user-attachments/assets/04d95d39-bb0d-403e-9400-3659b0377853)

## Q-Q Plot with T Distribution

This Q-Q plot displays the log returns against the theoretical quantiles of the t-distribution.

![Q-Q Plot for checking t-Distribution](https://github.com/user-attachments/assets/ea20f84c-6356-4d37-8af5-a9c246306b9b)

# GARCH Model

The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model is a statistical model used to analyze and forecast the volatility of financial time series data. It is particularly useful for modeling situations where the variance of the error terms varies over time, a phenomenon known as conditional heteroskedasticity. GARCH models allow for the estimation of time-varying volatility, capturing the clustering effect often observed in financial returns, where high-volatility periods tend to be followed by high-volatility periods and low-volatility periods by low-volatility periods.

## GARCH(p, q) Model Formula

The GARCH(p, q) model is defined by the following equations:

1. **Return Equation:**
   r_t = μ + ε_t

2. **Conditional Variance Equation:**
   σ_t² = α₀ + Σ (from i=1 to p) α_i ε_{t-i}² + Σ (from j=1 to q) β_j σ_{t-j}²

Where:
- r_t is the return at time t,
- μ is the mean of the returns,
- ε_t is the error term (residual) at time t,
- σ_t² is the conditional variance at time t,
- α₀ is a constant term,
- α_i are the coefficients for the lagged squared residuals (error terms),
- β_j are the coefficients for the lagged conditional variances,
- p is the order of the GARCH term (number of lagged variances),
- q is the order of the ARCH term (number of lagged errors).

This model provides a framework for capturing the dynamics of volatility in financial markets, making it a fundamental tool in econometrics and finance.

