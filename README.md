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

![Data Head](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/Data%20Head.png)

# Daily Log Returns Calculation

This section describes how to calculate the daily log returns for a stock's adjusted closing prices.

```python
df['Daily Log Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
df.head(10)
log_returns = df['Daily Log Returns'].dropna()
log_returns
```

![Log Returns](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/Log%20Returns.png)

# Interactive Log Returns vs. Date

This section presents an interactive graph of daily log returns plotted against the date.

![Log Returns EUR-USD Exch rate](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/Log%20Returns%20EUR-USD%20Exch%20rate.png)

# Distribution of Log Returns

This section includes histograms to visualize the distribution of daily log returns, checking the assumptions of the GARCH model.

## Histogram with Normal Distribution Line

This histogram displays the log returns with a normal distribution line traced on the histogram.

![Histogram of Log Returns (Normal Dist Fitted)](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/Histogram%20of%20Log%20Returns%20(Normal%20Dist%20Fitted).png)

## Histogram with T-Distribution Line

This histogram shows the log returns with a t-distribution line traced on the histogram.

![Histogram of Log Returns (t-Dist Fitted)](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/Histogram%20of%20Log%20Returns%20(t-Dist%20Fitted).png)

# Q-Q Plots of Log Returns

This section includes Q-Q plots to assess the distribution of daily log returns. The first plot compares the log returns to a normal distribution, while the second plot compares the log returns to a t-distribution.

## Q-Q Plot with Normal Distribution

This Q-Q plot displays the log returns against the theoretical quantiles of the normal distribution.

![Q-Q Plot for checking Normal Distribution](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/Q-Q%20Plot%20for%20checking%20Normal%20Distribution.png)

## Q-Q Plot with T Distribution

This Q-Q plot displays the log returns against the theoretical quantiles of the t-distribution.

![Q-Q Plot for checking t-Distribution](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/Q-Q%20Plot%20for%20checking%20t-Distribution.png)

# GARCH Model

The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model is a statistical model used to analyze and forecast the volatility of financial time series data. It is particularly useful for modeling situations where the variance of the error terms varies over time, a phenomenon known as conditional heteroskedasticity. GARCH models allow for the estimation of time-varying volatility, capturing the clustering effect often observed in financial returns, where high-volatility periods tend to be followed by high-volatility periods and low-volatility periods by low-volatility periods.

## GARCH(p, q) Model Formula

The GARCH(p, q) model is defined by the following equations:

![GARCH(p,q) Equation](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/GARCH(p%2Cq)%20Equation.png)


### Model Fitting and Evaluation (AIC & BIC)

To determine the optimal GARCH model for our data, we tested various combinations of \( p \) (GARCH terms) and \( q \) (ARCH terms). Each model was evaluated using the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC), which assess model quality while penalizing for complexity. Lower AIC and BIC values indicate a more suitable model fit, balancing accuracy with simplicity.

### Model Selection Using AIC

To identify the optimal GARCH model, we evaluated various combinations of \( p \) (GARCH terms) and \( q \) (ARCH terms) using the Akaike Information Criterion (AIC). The AIC penalizes model complexity to prevent overfitting, with lower values indicating a better fit. A line plot of AIC values across different combinations of \( p \) and \( q \) provides insight into the model that best balances accuracy with simplicity.

![AIC values](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/AIC%20values%20values.png)

From the AIC plot, it is evident that the GARCH(1,1) model yields the minimum AIC value, suggesting that it is the optimal model among the configurations tested.

### Model Selection Using BIC

Similarly, we calculated the Bayesian Information Criterion (BIC) for each \( p \) and \( q \) combination to further guide model selection. The BIC imposes a stronger penalty for model complexity than AIC, and thus may favor simpler models. By analyzing the line plot of BIC values, we can determine the combination of \( p \) and \( q \) that yields the most parsimonious model.

![BIC values](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/BIC%20values%20values.png)

The BIC plot shows that the GARCH(1,1) model also achieves the lowest BIC value, confirming it as the best fit for our data.

## GARCH(1,1) Model Equation

![GARCH(1,1) Equation](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/GARCH(1%2C1)%20Equation.png)

## Likelihood Ratio Test for Model Comparison

Since the AIC and BIC values for the GARCH(1,1) and GARCH(1,3) models are very close, we will perform a Likelihood Ratio (LR) test to statistically assess whether the difference in model fit is significant. This will help determine if the GARCH(1,3) model is statistically better than the GARCH(1,1) model.

### Hypotheses
- **Null Hypothesis (H0)**: The GARCH(1,1) model is adequate; the GARCH(1,3) model does not provide a significantly better fit.
- **Alternative Hypothesis (H1)**: The GARCH(1,3) model provides a significantly better fit than the GARCH(1,1) model.

### LR Test Formula
The Likelihood Ratio test statistic is calculated as follows:

LR = -2 * (log L(GARCH(1,1)) - log L(GARCH(1,3)))

where log L represents the log-likelihood of the respective models.

### Likelihood Ratio Test Results

After performing the Likelihood Ratio test, we found that the LR statistic is not significant, as it is greater than the critical value from the chi-squared distribution. Therefore, we fail to reject the null hypothesis.

### Conclusion
Since the GARCH(1,1) model is deemed adequate, it suggests that the GARCH(1,3) model does not provide a statistically significant improvement in fit over the GARCH(1,1) model. This result supports our earlier analysis that the GARCH(1,1) model is the optimal choice for modeling the volatility of the Eurodollar exchange rate in this context.

## Testing Data

For this project, we downloaded EUR/USD exchange rate data spanning from January 1, 2024, to October 20, 2024. This dataset will be utilized for testing the performance of our selected GARCH(1,1) model.

![Testing Data](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/Testing%20Data.png)


## Volatility Prediction

We have predicted the volatility for the testing data and compared it with the actual volatility values. The results are visualized in the line plot below, which illustrates the predicted and actual volatility over the testing period.

### Comparison of Predicted and Actual Volatility

![Predicted vs Actual Volatility](https://github.com/Rohan812/GARCH-Volatility-Modelling-for-EUR-USD-Exchange-Rate/blob/main/Predicted%20vs%20Actual%20Volatility.png)

This plot allows us to assess the accuracy of the GARCH(1,1) model's volatility predictions against the actual market volatility observed during the testing period.

