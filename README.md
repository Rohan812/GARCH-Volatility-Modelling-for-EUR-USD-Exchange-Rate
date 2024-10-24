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
import yfinance as yf

# Downloading the historical exchange rate data
data = yf.download('EURUSD=X', start='2019-01-01', end='2023-12-31')
data.head(10)

