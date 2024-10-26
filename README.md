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

![Untitled design](https://github.com/user-attachments/assets/b309bfc7-ed51-41d1-8295-f1b8d641a4b6)


### Model Fitting and Evaluation (AIC & BIC)

To determine the optimal GARCH model for our data, we tested various combinations of \( p \) (GARCH terms) and \( q \) (ARCH terms). Each model was evaluated using the Akaike Information Criterion (AIC) and the Bayesian Information Criterion (BIC), which assess model quality while penalizing for complexity. Lower AIC and BIC values indicate a more suitable model fit, balancing accuracy with simplicity.

### Model Selection Using AIC

To identify the optimal GARCH model, we evaluated various combinations of \( p \) (GARCH terms) and \( q \) (ARCH terms) using the Akaike Information Criterion (AIC). The AIC penalizes model complexity to prevent overfitting, with lower values indicating a better fit. A line plot of AIC values across different combinations of \( p \) and \( q \) provides insight into the model that best balances accuracy with simplicity.

![newplot (3)](https://github.com/user-attachments/assets/2fd0dbb1-0576-4945-8219-875d3a35f9d7)

From the AIC plot, it is evident that the GARCH(1,1) model yields the minimum AIC value, suggesting that it is the optimal model among the configurations tested.

### Model Selection Using BIC

Similarly, we calculated the Bayesian Information Criterion (BIC) for each \( p \) and \( q \) combination to further guide model selection. The BIC imposes a stronger penalty for model complexity than AIC, and thus may favor simpler models. By analyzing the line plot of BIC values, we can determine the combination of \( p \) and \( q \) that yields the most parsimonious model.

![newplot (4)](https://github.com/user-attachments/assets/9ac3f589-cfdf-4c9c-9c81-c87773794a2d)

The BIC plot shows that the GARCH(1,1) model also achieves the lowest BIC value, confirming it as the best fit for our data.

## GARCH(1,1) Model Equation

![Untitled design (1)](https://github.com/user-attachments/assets/c16f87d1-48a6-43cb-bc02-7a922685b3cd)

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

![Screenshot (247)](https://github.com/user-attachments/assets/f5e16e79-f784-4061-973b-3bc324c6ffc2)

## Volatility Prediction

We have predicted the volatility for the testing data and compared it with the actual volatility values. The results are visualized in the line plot below, which illustrates the predicted and actual volatility over the testing period.

### Comparison of Predicted and Actual Volatility

![newplot (5)](https://github.com/user-attachments/assets/5bfba38f-3a67-4c4f-bfdc-0abc18ba6d0c)

This plot allows us to assess the accuracy of the GARCH(1,1) model's volatility predictions against the actual market volatility observed during the testing period.

