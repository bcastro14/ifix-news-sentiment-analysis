# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:32:55 2024

@author: elias
"""

print(f"Start date: {start_date.date()}\n" + 
      f"End date:{end_date.date()}\n" +
      f"Number of news:{len(news)}")

#%%
end_date = datetime(2020, 9, 15)
day_delta = timedelta(days=15)
start_date = end_date - day_delta

#%%
df1 = combined_df
df2 = filtered_df

combined_filtered_df = pd.concat([df1, df2])
# Identify and keep only the rows that are unique (i.e., not duplicated)
# `keep=False` marks both the original and duplicate as duplicates
unique_rows = combined_filtered_df[~combined_filtered_df.duplicated(keep=False)]
unique_rows.to_csv('unique_rows.csv', index = False)

#%%
# Function to normalize non-ASCII characters:
def normalize_text(text):
    # Dictionary to map non-ASCII characters to ASCII equivalents
    normalization_map = {
        '‘': "'", '’': "'",             # Curly single quotes to straight quotes
        '“': '"', '”': '"',             # Curly double quotes to straight quotes
        '—': '-', '─': '-', '–': '-',    # Em dash to hyphen
        'á': 'a', 'ã': 'a', 'à': 'a', 'â': 'a',
        'Á': 'A', 'Ã': 'A', 'À': 'A', 'Â': 'A',
        'é': 'e', 'è': 'e', 'ê': 'e',
        'É': 'E', 'È': 'E', 'Ê': 'E',
        'í': 'i', 'ì': 'i',
        'Í': 'I', 'Ì': 'I',
        'ó': 'o', 'ô': 'o', 'õ': 'o',
        'Ó': 'o', 'Ô': 'o', 'Õ': 'o',
        'ú': 'u', 'ü': 'u',
        'Ú': 'u', 'Ü': 'u',
        'ç': 'c'
    }
    
    # Replace each non-ASCII character using the map
    for non_ascii_char, ascii_char in normalization_map.items():
        text = text.replace(non_ascii_char, ascii_char)
    
    return text

#%%
# Normalizing non-ascii characters in "all_news":
for item in all_news:
    for key, value in item.items():
        if isinstance(value, str):  # Check if the value is a string
            item[key] = normalize_text(value)  # Apply normalization to the string

#%%
import pandas as pd

# Load the CSV files
df1 = pd.read_csv('csv_output/b_news_result_reuters1.csv')
df2 = pd.read_csv('csv_output/b_news_result_reuters2.csv')

# Concatenate the two DataFrames
df_combined = pd.concat([df1, df2], ignore_index=True)

# Drop the "position" column
df_combined = df_combined.drop(columns=['position'])

# Check for duplicate rows (considering all columns)
duplicate_rows = df_combined[df_combined.duplicated()]
duplicate_row_count = duplicate_rows.shape[0]

# Check for duplicates based on the "title" column
duplicate_titles = df_combined[df_combined.duplicated(subset=['title'])]
duplicate_title_count = duplicate_titles.shape[0]

# Check for duplicates based on both "title" and "snippet" columns
duplicate_title_snippet = df_combined[df_combined.duplicated(subset=['title', 'snippet'])]
duplicate_title_snippet_count = duplicate_title_snippet.shape[0]


# Print results
print(f"Total number of duplicate rows (all columns): {duplicate_row_count}")
print(f"Total number of duplicate titles: {duplicate_title_count}")
print(f"Total number of duplicate title-snippet pairs: {duplicate_title_snippet_count}")

#%%
import pandas as pd

# Load the CSV into a DataFrame
df = pd.read_csv('csv_output/b_news_result_reuters2.csv')

# Replace double spaces in the 'snippet' column
df['snippet'] = df['snippet'].str.replace("\n", "").str.replace("  ", " ")

# (Optional) Do the same for other columns like 'title'
df['title'] = df['title'].str.replace("\n", "").str.replace("  ", " ")

# Save the modified DataFrame back to a CSV if needed
df.to_csv('csv_output/your_modified_file.csv', index=False)

#%%
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Function to scrape date from Reuters
def get_reuters_date(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    date_element = soup.find('span', class_='date-line__date___kNbY')
    if date_element:
        return date_element.text.strip()
    return None

# Function to scrape date from Bloomberg
def get_bloomberg_date(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    time_element = soup.find('time', datetime=True)
    if time_element:
        return time_element['datetime'][:10]  # Extract only the date part (YYYY-MM-DD)
    return None

# Function to fetch the correct date based on the source
def fetch_actual_date(row):
    if 'week ago' in row['date'] or 'month ago' in row['date']:
        url = row['link']
        if 'Reuters' in row['source']:
            return get_reuters_date(url)
        elif 'Bloomberg' in row['source']:
            return get_bloomberg_date(url)
    return row['absolute_date']  # If already has an absolute date

# Load CSV file
df = pd.read_csv('csv_output/d_news_with_sentiment.csv')

# Apply the function to rows with relative dates
df['corrected_date'] = df.apply(fetch_actual_date, axis=1)

# Save the updated DataFrame
df.to_csv('csv_output/updated_date.csv', index=False)

#%%
import pandas as pd

df = pd.read_csv('csv_output/d_news_with_sentiment.csv')
filtered_df = df[~df['date'].str.contains('week ago|month ago|weeks ago', na=False)]
filtered_df.to_csv('csv_output/d_news_with_sentiment_valid_dates.csv', index=False)

#%%
import pandas as pd
import matplotlib.pyplot as plt

# df1 = pd.read_csv('csv_output/d_news_with_sentiment.csv')
df1 = pd.read_csv('csv_output/d_news_with_sentiment_valid_dates.csv')

#%%
# Create a date range covering from the lowest to the greatest absolute_date
min_date = df1['absolute_date'].min()
max_date = df1['absolute_date'].max()
all_dates = pd.date_range(start=min_date, end=max_date)

# Count the number of occurrences for each date
count_by_date = df1.groupby('absolute_date').size().reindex(all_dates, fill_value=0).reset_index()
count_by_date.columns = ['date', 'count']

#%%
# Plot the data for each year
years = count_by_date['date'].dt.year.unique()
for year in years:
    yearly_data = count_by_date[count_by_date['date'].dt.year == year]
    plt.figure(figsize=(10, 5))
    plt.figure(dpi=300)
    plt.plot(yearly_data['date'], yearly_data['count'], marker='o')
    plt.title(f'News Count per Day in {year}')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
#%%
# Group by absolute_date and calculate the arithmetic mean for the sentiment columns
df_sent = df1.groupby('absolute_date').agg({
    'title_sentiment': 'mean',
    'snippet_sentiment': 'mean',
    'average_sentiment': 'mean'
}).reset_index()

#%%

# Load IFIX data
df_ifix = pd.read_csv('combined_output_ifix.csv', parse_dates=['Date'])

# Convert absolute_date to datetime format in df_sent to avoid comparison errors
df_sent['absolute_date'] = pd.to_datetime(df_sent['absolute_date'])

# Initialize empty lists to hold the final data
final_dates = []
final_values = []
final_title_sentiment = []
final_snippet_sentiment = []
final_average_sentiment = []

# Initialize temporary storage for sentiment accumulation
temp_title = []
temp_snippet = []
temp_average = []

#%%
# Iterate through df_ifix to process dates
for i, row in df_ifix.iterrows():
    current_date = row['Date']
    value = row['Value']
    
    # Filter df_sent for dates before or equal to the current date
    eligible_rows = df_sent[df_sent['absolute_date'] <= current_date]
    
    # If there are any eligible rows, calculate the average sentiment for these rows
    if not eligible_rows.empty:
        temp_title.append(eligible_rows['title_sentiment'].mean())
        temp_snippet.append(eligible_rows['snippet_sentiment'].mean())
        temp_average.append(eligible_rows['average_sentiment'].mean())

    # Compute the overall average of the accumulated values
    avg_title = sum(temp_title) / len(temp_title) if temp_title else None
    avg_snippet = sum(temp_snippet) / len(temp_snippet) if temp_snippet else None
    avg_average = sum(temp_average) / len(temp_average) if temp_average else None
    
    # Append to final lists
    final_dates.append(current_date)
    final_values.append(value)
    final_title_sentiment.append(avg_title)
    final_snippet_sentiment.append(avg_snippet)
    final_average_sentiment.append(avg_average)

    # Clear temp accumulators for the next round
    temp_title.clear()
    temp_snippet.clear()
    temp_average.clear()

# Create the final df_ifix_sent DataFrame
df_ifix_sent = pd.DataFrame({
    'Date': final_dates,
    'Value': final_values,
    'title_sentiment': final_title_sentiment,
    'snippet_sentiment': final_snippet_sentiment,
    'average_sentiment': final_average_sentiment
})

#%%
# Display or save the resulting dataframe
df_ifix_sent.to_csv("csv_output/df_ifix_sent.csv", index=False)

#%%
# Load the CSV files
df = pd.read_csv('csv_output/a_news_result_bloomberg.csv')
df2 = pd.read_csv('csv_output/a_news_result_reuters.csv')

# Convert 'absolute_date' to datetime format
df['absolute_date'] = pd.to_datetime(df['absolute_date'])
df2['absolute_date'] = pd.to_datetime(df2['absolute_date'])

# Define the cutoff date
cutoff_date = pd.Timestamp('2018-03-01')

# Filter both dataframes to keep rows where 'absolute_date' is on or after March 2018
df_filtered = df[df['absolute_date'] >= cutoff_date]
df2_filtered = df2[df2['absolute_date'] >= cutoff_date]

# Optionally, you can save the filtered dataframes back to CSVs
df_filtered.to_csv('csv_output/a_news_result_bloomberg.csv', index=False)
df2_filtered.to_csv('csv_output/a_news_result_reuters.csv', index=False)

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

#%%
############ Preparing the data for the VAR Model + Plots ############
# Import the CSV as a dataframe
df = pd.read_csv('csv_output/d_ifix_sentiment.csv', parse_dates=["date"])

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
title_stationary = check_stationarity(df['title_sentiment'])
snippet_stationary = check_stationarity(df['snippet_sentiment'])

#%%
# Difference if not stationary
if not ifix_stationary:
    df['ifix_value'] = df['ifix_value'].diff()
    

# Verify if new differenciated series are stationary
# new_ifix_stationary = check_stationarity(df['ifix_value'])

#%%
df['average_sentiment'] = df['average_sentiment'].diff()
#%%
df.dropna(subset=['average_sentiment'], inplace=True)

df.dropna(subset=['ifix_value'], inplace=True)

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
############ MODEL 1 - IFIX x Average Sentiment (entire dataset) ############

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
lag_order = model.select_order(maxlags=30)
print(lag_order.summary())  # This will show BIC and other criteria for different lag orders

#%%
# Get the lag corresponding to the lowest BIC
optimal_lag_bic = lag_order.bic

# Fit the VAR model using the optimal lag based on BIC
var_model = model.fit(optimal_lag_bic)

print(var_model.summary())  # Check the model summary with the chosen lag

#%%
# Get the lag corresponding to the lowest AIC
optimal_lag_aic = lag_order.aic

# Fit the VAR model using the optimal lag based on AIC
var_model = model.fit(optimal_lag_aic)

print(var_model.summary())  # Check the model summary with the chosen lag

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


#%%
########## Testing the result of the model with title_sentiment ##########
df_var = df[['ifix_value', 'title_sentiment']]
model = VAR(df_var)

lag_order = model.select_order(maxlags=30)
print(lag_order.summary())

#%%
# Get the lag corresponding to the lowest BIC
optimal_lag_bic = lag_order.bic
var_model = model.fit(optimal_lag_bic)
print(var_model.summary())

#%%
# Get the lag corresponding to the lowest AIC
optimal_lag_aic = lag_order.aic
var_model = model.fit(optimal_lag_aic)
print(var_model.summary())

#%%
########## Testing the result of the model with snippet_sentiment ##########
df_var = df[['ifix_value', 'snippet_sentiment']]
model = VAR(df_var)

lag_order = model.select_order(maxlags=30)
print(lag_order.summary())

#%%
# Get the lag corresponding to the lowest BIC
optimal_lag_bic = lag_order.bic
var_model = model.fit(optimal_lag_bic)
print(var_model.summary())

#%%
# Get the lag corresponding to the lowest AIC
optimal_lag_aic = lag_order.aic
var_model = model.fit(optimal_lag_aic)
print(var_model.summary())

#%%
##############################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

#%%
############ Preparing the data for the VAR Model + Plots ############
# Import the CSV as a dataframe
df = pd.read_csv('csv_output/d_ifix_sentiment.csv', parse_dates=["date"])

# Filter the dataframe based on the date range
start_date = '2023-08-21'
end_date = '2024-08-21'
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

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
title_stationary = check_stationarity(df['title_sentiment'])
snippet_stationary = check_stationarity(df['snippet_sentiment'])

#%%
# Difference if not stationary
df['ifix_value'] = df['ifix_value'].diff()
df['average_sentiment'] = df['average_sentiment'].diff()
    

# Verify if new differenciated series are stationary
# new_ifix_stationary = check_stationarity(df['ifix_value'])

df.dropna(subset=['ifix_value'], inplace=True)
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
############ MODEL 2 - IFIX x Average Sentiment (dataset limited to 1 year) ############

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
lag_order = model.select_order(maxlags=30)
print(lag_order.summary())  # This will show BIC and other criteria for different lag orders

#%%
# Get the lag corresponding to the lowest BIC
optimal_lag_bic = lag_order.bic

# Fit the VAR model using the optimal lag based on BIC
var_model = model.fit(optimal_lag_bic)

print(var_model.summary())  # Check the model summary with the chosen lag

#%%
# Get the lag corresponding to the lowest AIC
optimal_lag_aic = lag_order.aic

# Fit the VAR model using the optimal lag based on AIC
var_model = model.fit(optimal_lag_aic)

print(var_model.summary())  # Check the model summary with the chosen lag

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

