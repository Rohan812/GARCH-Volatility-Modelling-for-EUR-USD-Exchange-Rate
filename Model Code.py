#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from scipy.stats import chi2
import datetime as dt
import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[3]:


data = yf.download('EURUSD=X', start = '2019-01-01', end = '2023-12-31')


# In[4]:


data.head(10)


# In[30]:


df = data.copy(deep=True)
df.head(10)


# In[31]:


df['Daily Volatility'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
df.head(10)


# In[32]:


log_returns = df['Daily Volatility'].dropna()
log_returns


# In[34]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=log_returns.index, y=log_returns, mode='lines', name='Log Returns'))
fig.add_trace(go.Scatter(x=log_returns.index, y=[0]*len(log_returns), mode='lines', line=dict(color='black', width=2), name='Zero Line'))
fig.update_layout(
    title='Interactive Log Returns of EUR/USD',
    xaxis_title='Date',
    yaxis_title='Log Returns',
    yaxis=dict(range=[-0.05, 0.05]),
    template='plotly_dark',  # You can also try 'ggplot2', 'seaborn', 'plotly', etc.
    hovermode='x',           
    showlegend=True,
)
fig.show()


# In[35]:


max_p = int(input("Enter the highest value of p (GARCH order for past variances): "))
max_q = int(input("Enter the highest value of q (GARCH order for past shocks): "))


# In[36]:


hist_data = np.histogram(log_returns, bins=30, density=True)
mu, std = norm.fit(log_returns)
xmin, xmax = min(log_returns), max(log_returns)
x = np.linspace(-0.015, 0.015, 100)
p = norm.pdf(x, mu, std)
hist_trace = go.Histogram(
    x=log_returns,
    histnorm='probability density',
    nbinsx=30,
    marker=dict(color='dodgerblue', line=dict(color='black', width=1)),
    opacity=0.7,
    name='Log Returns'
)

line_trace = go.Scatter(
    x=x,
    y=p,
    mode='lines',
    line=dict(color='red', width=2),
    name=f'Normal Distribution<br>μ = {mu:.4f}, σ = {std:.4f}'
)


layout = go.Layout(
    title='Interactive Histogram of Log Returns with Fitted Normal Distribution',
    xaxis=dict(title='Log Returns', range=[-0.015, 0.015]),
    yaxis=dict(title='Density'),
    hovermode='closest',
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    paper_bgcolor='rgba(255,255,255,1)',  # White background
    showlegend=True,
)


fig = go.Figure(data=[hist_trace, line_trace], layout=layout)


fig.show()


# In[37]:


hist_data = np.histogram(log_returns, bins=30, density=True)
dof, loc, scale = stats.t.fit(log_returns)
xmin, xmax = min(log_returns), max(log_returns)
x = np.linspace(xmin, xmax, 100)
p = stats.t.pdf(x, dof, loc, scale)
hist_trace = go.Histogram(
    x=log_returns,
    histnorm='probability density',
    nbinsx=30,
    marker=dict(color='dodgerblue', line=dict(color='black', width=1)),
    opacity=0.7,
    name='Log Returns'
)
t_dist_trace = go.Scatter(
    x=x,
    y=p,
    mode='lines',
    line=dict(color='red', width=2),
    name=f't-Distribution (dof={dof:.2f}, loc={loc:.4f}, scale={scale:.4f})'
)
layout = go.Layout(
    title="Interactive Histogram of Log Returns with Fitted t-Distribution",
    xaxis=dict(title="Log Returns", range=[-0.015, 0.015]),
    yaxis=dict(title="Density"),
    showlegend=True
)
fig = go.Figure(data=[hist_trace, t_dist_trace], layout=layout)
fig.show()


# # Checking whether Log Returns are Normally distributed

# In[38]:


qq = sm.qqplot(log_returns.dropna(), line ='s', alpha=0.5)
qq_data = qq.gca().lines[0].get_xydata()
theoretical_quantiles = qq_data[:, 0]
sample_quantiles = qq_data[:, 1]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=theoretical_quantiles,
    y=sample_quantiles,
    mode='markers',
    marker=dict(color='dodgerblue', size=6),
    name='Q-Q Points'
))
fig.add_trace(go.Scatter(
    x=theoretical_quantiles,
    y=theoretical_quantiles,  # Identity line (y = x)
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Reference Line'
))
fig.update_layout(
    title="Interactive Q-Q Plot for Normal Distribution",
    xaxis_title="Theoretical Quantiles",
    yaxis_title="Sample Quantiles",
    width=800,
    height=600,
    template='plotly'
)
fig.show()


# ### As we can observe from the qq-plot that the Log returns are not Normally Distriuted as has fatter tails. So, we will check if Log Returns are t-student distributed

# In[39]:


n = len(log_returns)  # Number of observations
dof = n - 1  # Degrees of freedom
loc = log_returns.mean()  # Location parameter (mean)
scale = log_returns.std(ddof=1)
sm.qqplot(log_returns.dropna(), dist=stats.t, line='45', distargs=(dof,), loc=0, scale=scale)
plt.title("Q-Q Plot for t-Distribution")
plt.show()


# In[40]:


def optimal_garch(returns, max_p, max_q):
    aic_values=[]
    bic_values=[]
    loglik_values = []
    optimal_aic = np.inf
    optimal_bic = np.inf
    optimal_model = None
    pq_combinations = []
    optimal_pq = (0,0)
    
    for p in range(1, max_p+1):
        for q in range(1, max_q+1):
            try:
                model = arch_model(returns, vol='Garch', p=p, q=q, rescale = False)
                result = model.fit()
                print(f"AIC = {result.aic}")
                
                aic_values.append(result.aic)
                bic_values.append(result.bic)
                loglik_values.append(result.loglikelihood)
                pq_combinations.append((p,q))
                
                if result.aic < optimal_aic:
                    optimal_aic = result.aic
                    optimal_model = result
                    optimal_pq = (p,q)
                    
                    #if result.bic < optimal_bic:
                        #optimal_bic = result.bic
            except Exception as e:
                print(f"Error fitting model with p={p}, q={q}: {e}")
    
    print(f"Optimal Model: GARCH({optimal_pq[0]},{optimal_pq[1]})")
    print(f"Optimal AIC: {optimal_aic}")
    print(f"Optimal BIC: {optimal_bic}")
    
    return optimal_model , aic_values , bic_values, pq_combinations, loglik_values


# In[41]:


optimal_model, aic_values, bic_values, pq_combinations, loglik_values = optimal_garch(log_returns, max_p, max_q)


# In[64]:


pq_labels = [f"GARCH({p},{q})" for p, q in pq_combinations]


# In[66]:


aic_trace = go.Scatter(
    x=pq_labels,
    y=aic_values,
    mode='lines+markers',
    name='AIC',
    marker=dict(symbol='circle', size=10, color='deepskyblue', line=dict(color='darkblue', width=2)),
    line=dict(color='royalblue', width=2, dash='dash'),
    hoverinfo='x+y'
)

min_aic = min(aic_values)
min_idx = aic_values.index(min_aic)
min_label = pq_labels[min_idx]

layout = go.Layout(
    title='AIC for Different GARCH(p, q) Models',
    xaxis=dict(title='GARCH(p, q)', tickangle=45, tickfont=dict(size=12)),
    yaxis=dict(title='Criterion Value', zeroline=True, gridcolor='gray'),
    plot_bgcolor='rgba(245, 245, 245, 0.9)',  
    paper_bgcolor='rgba(0, 0, 0, 0)',  
    hovermode='x unified',  
    margin=dict(l=50, r=50, t=50, b=50),
    annotations=[
        dict(
            x=min_label,
            y=min_aic,
            xref="x",
            yref="y",
            text=f"Min AIC: {min_aic:.2f}",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40,
            font=dict(color='black', size=12)
        )
    ]
)


fig = go.Figure(data=[aic_trace], layout=layout)

fig.show()


# In[67]:


bic_trace = go.Scatter(
    x=pq_labels,
    y=bic_values,
    mode='lines+markers',
    name='BIC',
    marker=dict(symbol='circle', size=10, color='limegreen', line=dict(color='darkgreen', width=2)),
    line=dict(color='green', width=2, dash='dashdot'),
    hoverinfo='x+y'
)

min_bic = min(bic_values)
min_bic_idx = bic_values.index(min_bic)
min_bic_label = pq_labels[min_bic_idx]

layout = go.Layout(
    title='BIC for Different GARCH(p, q) Models',
    xaxis=dict(title='GARCH(p, q)', tickangle=45, tickfont=dict(size=12)),
    yaxis=dict(title='Criterion Value', zeroline=True, gridcolor='gray'),
    plot_bgcolor='rgba(245, 245, 245, 0.9)',  
    paper_bgcolor='rgba(0, 0, 0, 0)',  
    hovermode='x unified',
    margin=dict(l=50, r=50, t=50, b=50),
    annotations=[
        dict(
            x=min_bic_label,
            y=min_bic,
            xref="x",
            yref="y",
            text=f"Min BIC: {min_bic:.2f}",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40,
            font=dict(color='black', size=12)
        )
    ]
)

fig = go.Figure(data=[bic_trace], layout=layout)

fig.show()


# In[68]:


print(pq_labels, aic_values , bic_values)
criterion_values = pd.DataFrame({'pq_combinations':pq_labels, 'AIC Values':aic_values , 'BIC Values':bic_values, 'LogLikelihood Values':loglik_values})
criterion_values


# In[69]:


loglik_garch_11 = criterion_values['LogLikelihood Values'][0]
loglik_garch_13 = criterion_values['LogLikelihood Values'][2]
lr_stat = 2 * (loglik_garch_13 - loglik_garch_11)
critical_value = chi2.ppf(0.95, df=1)
if lr_stat > critical_value:
    print("GARCH(1,3) is statistically significantly better than GARCH(1,1)")
else:
    print("No significant difference between GARCH(1,1) and GARCH(1,3)")


# In[70]:


testing_data = yf.download('EURUSD=X', start = '2023-01-01', end = '2024-10-20')
testing_data.head(30)


# In[71]:


testing_df = testing_data.copy(deep=True)
testing_df = pd.DataFrame(testing_df)
testing_df['Actual Daily Volatility'] = np.log(testing_df['Adj Close']/testing_df['Adj Close'].shift(1))
testing_df = testing_df.iloc[1:]
testing_df.head(10)


# In[55]:


df.tail(10)


# In[72]:


n_test = len(testing_df)
predicted_volatility = optimal_model.forecast(horizon=n_test)
volatility_forecast = np.sqrt(predicted_volatility.variance.values[-1])
last_observed_return = df['Daily Volatility'][-1]
predicted_log_returns = np.zeros(n_test)
for i in range(n_test):
    predicted_log_returns[i] = last_observed_return + volatility_forecast[i] * np.random.standard_t(len(testing_df)-1)
testing_df['Predicted Volatility'] = predicted_log_returns
testing_df.head(20)


# In[73]:


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=testing_df.index,
    y=testing_df['Actual Daily Volatility'],
    mode='lines',
    name='Actual Daily Volatility',
    line=dict(color='blue'),
    hovertemplate='Actual Daily Volatility: %{y:.2f}<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=testing_df.index,
    y=testing_df['Predicted Volatility'],
    mode='lines',
    name='Predicted Volatility',
    line=dict(color='red'),
    hovertemplate='Predicted Volatility: %{y:.2f}<extra></extra>'
))
fig.update_layout(
    title='Actual vs Predicted Predicted Volatility',
    xaxis_title='Date',
    yaxis_title='Predicted Volatility',
    legend_title='Legend',
    template='plotly_white',  # Optional: choose a template for the style
    hovermode='x unified',  # Optional: hover over x-axis shows both values
)
fig.show()


# In[74]:


actual_volatility = testing_df['Actual Daily Volatility']
predicted_volatility = testing_df['Predicted Volatility']

# Calculate MAE, MSE, and RMSE
mae = mean_absolute_error(actual_volatility, predicted_volatility)
mse = mean_squared_error(actual_volatility, predicted_volatility)
rmse = np.sqrt(mse)

# Print the results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


# In[ ]:




