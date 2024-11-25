import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

#%%
############ VAR Model + Plots for MONTHLY data ############
# Import the CSV as a dataframe
df = pd.read_csv('csv_output/d_ifix_sentiment.csv', parse_dates=["date"])

# Group by the year and month, then calculate the average for numeric columns
df = (
    df.groupby(df['date'].dt.to_period('M'))
      .mean(numeric_only=True)
      .reset_index()
)
df['date'] = df['date'].dt.to_timestamp()

#%%
# Create a single plot for IFIX value
plt.figure(figsize=(12, 4), dpi=120)

# Set starting and ending dates based on your data's minimum and maximum dates
start_date = df['date'].min()
end_date = df['date'].max()

# Plot IFIX value
plt.plot(df["date"], df["ifix_value"], label="Valor do IFIX", color="indigo")
plt.xlim(start_date, end_date)  # Set axis limits based on data range
plt.xlabel("Data", fontsize=14)
plt.ylabel("Valor do IFIX", fontsize=14)
plt.title("Valor do IFIX ao longo do tempo", fontsize=16)

# Set font size for tick labels
plt.tick_params(axis='both', labelsize=12) 

# Add vertical lines for every 3 months, starting from the first date's year and first quarter
for year in range(start_date.year, end_date.year + 1):
    for quarter_start_month in [1, 4, 7, 10]:  # 1st month of each quarter
        quarter_start = pd.to_datetime(f"{year}-{quarter_start_month}-01")
        if start_date <= quarter_start <= end_date:  # Ensure it's within the date range
            plt.axvline(quarter_start, linestyle="--", color="k", alpha=0.5)

# Display the plot
plt.tight_layout()
plt.show()

#%%
# Check and handle nonstationarity

# Function to check stationarity with p-value of 5%
def check_stationarity(series, cutoff=0.05):
    adf_test = adfuller(series)
    print('ADF Statistic:', adf_test[0])
    print('p-value:', round(adf_test[1], 4))
    print('Critical Values:')
    
    for key, value in adf_test[4].items():  
        print('\t%s: %.3f' % (key, value))
    
    is_stationary = adf_test[1] < cutoff
    if is_stationary:
        print('Series is stationary\n')
    else:
        print('Series is non-stationary\n')
    
    return is_stationary

# Check for stationarity
ifix_stationary = check_stationarity(df['ifix_value'])
sentiment_stationary = check_stationarity(df['average_sentiment'])

#%%
# Difference if not stationary
if not ifix_stationary:
    df['ifix_value'] = df['ifix_value'].diff()
    df.dropna(subset=['ifix_value'], inplace=True)

# Verify if new differenciated series are stationary
new_ifix_stationary = check_stationarity(df['ifix_value'])

#%%
# Create two subplots:
# Average Sentiment Over Time and IFIX Value Over Time
fig, axs = plt.subplots(2, 1, figsize=(12, 8), dpi=120)
plt.subplots_adjust(hspace=0.8)  # Adjust the vertical spacing

# Set starting and ending dates based on your data's minimum and maximum dates
start_date = df['date'].min()
end_date = df['date'].max()

# Plot average_sentiment
axs[0].plot(df["date"], df["average_sentiment"], label="Média do sentimento", color="tomato")
axs[0].set_xlim(start_date, end_date)  # Set axis limits based on data range
axs[0].set_xlabel("Data", fontsize=14)
axs[0].set_ylabel("Média do sentimento", fontsize=14)
axs[0].set_title("Valor médio do sentimento ao longo do tempo", fontsize=16)

# Plot ifix_value
axs[1].plot(df["date"], df["ifix_value"], label="Valor do IFIX", color="indigo")
axs[1].set_xlim(start_date, end_date)  # Set axis limits based on data range
axs[1].set_xlabel("Data", fontsize=14)
axs[1].set_ylabel("Valor do IFIX", fontsize=14)
axs[1].set_title("Valor do IFIX ao longo do tempo", fontsize=16)

# Increase tick label font size
for ax in axs:
    ax.tick_params(axis='both', labelsize=12)  # Set font size for tick labels

# Add vertical lines for every 3 months, starting from the first date's year and first quarter
for year in range(start_date.year, end_date.year + 1):
    for quarter_start_month in [1, 4, 7, 10]:  # 1st month of each quarter
        quarter_start = pd.to_datetime(f"{year}-{quarter_start_month}-01")
        if start_date <= quarter_start <= end_date:  # Ensure it's within the date range
            axs[0].axvline(quarter_start, linestyle="--", color="k", alpha=0.5)
            axs[1].axvline(quarter_start, linestyle="--", color="k", alpha=0.5)

# Adjust spacing between subplots
plt.tight_layout()

plt.show()

#%%
############ MODEL 3 - IFIX x Average Sentiment (monthly average) ############

# Select only the necessary columns for the VAR model
df_var = df[['ifix_value', 'average_sentiment']]

# Print descriptive statistics
df_var.describe()

#%%
# Fit the VAR model
model = VAR(df_var)

#%%
# Fit the model and select the optimal lag based on BIC
# The maxlags parameter is the maximum number of lags to check
lag_order = model.select_order(maxlags=6)
print(lag_order.summary())  # This will show BIC and other criteria for different lag orders

#%%
# Get the lag corresponding to the lowest BIC
optimal_lag_bic = lag_order.bic

# Fit the VAR model using the optimal lag based on BIC
var_model_bic = model.fit(optimal_lag_bic)

print(var_model_bic.summary())  # Check the model summary with the chosen lag

#%%
# Get the lag corresponding to the lowest AIC
optimal_lag_aic = lag_order.aic

# Fit the VAR model using the optimal lag based on AIC
var_model_aic = model.fit(optimal_lag_aic)

print(var_model_aic.summary())  # Check the model summary with the chosen lag

#%%
# Compare the log-likelihoods of the two models and choose the one with the higher value
log_likelihood_bic = var_model_bic.llf
log_likelihood_aic = var_model_aic.llf

if log_likelihood_bic > log_likelihood_aic:
    var_model = var_model_bic
    print("Selected model: BIC with higher log-likelihood")
else:
    var_model = var_model_aic
    print("Selected model: AIC with higher log-likelihood")

#%%
# Test Granger causality
print("\nCheck if 'average_sentiment' Granger-causes 'ifix_value'")
granger_test = var_model.test_causality('ifix_value', ['average_sentiment'], kind='f')
print(granger_test.summary())

print("\nCheck if 'ifix_value' Granger-causes 'average_sentiment'")
granger_test_reverse = var_model.test_causality('average_sentiment', ['ifix_value'], kind='f')
print(granger_test_reverse.summary())

#%%
# Compute the impulse response function (IRF)
irf = var_model.irf(10)  # The number of periods ahead to compute the response

# Plot the IRF to visualize the response of 'ifix_value' to a shock in 'average_sentiment' and vice versa
fig = irf.plot(orth=False)

axes = fig.axes
axes[0].set_title("Valor do IFIX → Valor do IFIX")
axes[1].set_title("Média do Sentimento → Valor do IFIX")
axes[2].set_title("Valor do IFIX → Média do Sentimento")
axes[3].set_title("Média do Sentimento → Média do Sentimento")

# # Show the cumulative IRF (optional, to see the long-term effect)
irf.plot_cum_effects(orth=False)
plt.show()

