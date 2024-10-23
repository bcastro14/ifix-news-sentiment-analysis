import pandas as pd
import numpy as np

#%%
##### CREATING DF WITH BOTH SENTIMENT AND IFIX VALUES, 
##### RANGING THROUGH MIN AND MAX DATE OF THE SENTIMENT SERIES.
df1 = pd.read_csv('csv_output/c_news_with_sentiment.csv')
df2 = pd.read_csv('csv_output/i_combined_output_ifix.csv')

# Create df_sent by selecting relevant columns and renaming 'absolute_date' to 'date'
df_sent = df1[['absolute_date', 'title_sentiment', 'snippet_sentiment', 'average_sentiment']].copy()
df_sent.rename(columns={'absolute_date': 'date'}, inplace=True)

# Convert 'date' to datetime for both df_sent and df2
df_sent['date'] = pd.to_datetime(df_sent['date'])
df2['Date'] = pd.to_datetime(df2['Date'])

# Calculate arithmetic average for sentiments per date in df_sent
df_sent = df_sent.groupby('date').mean().reset_index()

# Get the maximum and minimum date from df_sent
min_date = df_sent['date'].min()
max_date = df_sent['date'].max()

# Create df_ifix_sent with all dates between min_date and max_date
df_ifix_sent = pd.DataFrame(pd.date_range(start=min_date, end=max_date, freq='D'), columns=['Date'])

# Sort df_ifix_sent in ascending order by Date
df_ifix_sent = df_ifix_sent.sort_values(by='Date', ascending=True)

# Merge df_sent and df2 into df_ifix_sent based on 'Date'
df_ifix_sent = df_ifix_sent.merge(df2, on='Date', how='left').merge(df_sent, left_on='Date', right_on='date', how='left')

# Drop the extra 'date' column from df_sent and fill missing values with NA
df_ifix_sent = df_ifix_sent.drop(columns=['date']).fillna(pd.NA)

# Renaming Date and Value columns
df_ifix_sent.rename(columns={'Date': 'date', 'Value':'ifix_value'}, inplace=True)

# Round sentiment columns to 5 decimal places
df_ifix_sent['title_sentiment'] = df_ifix_sent['title_sentiment'].round(5)
df_ifix_sent['snippet_sentiment'] = df_ifix_sent['snippet_sentiment'].round(5)
df_ifix_sent['average_sentiment'] = df_ifix_sent['average_sentiment'].round(5)

# Saving a CSV with all dates, and full of NAs
df_ifix_sent.to_csv("csv_output/d_ifix_sentiment_alldates.csv", index=False)

#%%
##### PROCESSING AVERAGE ACCUMULATED SENTIMENT VALUE FOR DAYS WITHOUT IFIX VALUE.

# Create df_ifix_sent_filtered considering the average accumulated sentiment value.
def apply_sentiment_buffer(df):
    buffer_title_sentiment = []
    buffer_snippet_sentiment = []
    buffer_average_sentiment = []

    filtered_rows = []
    
    for idx, row in df.iterrows():
        if pd.isna(row['ifix_value']):
            if not pd.isna(row['title_sentiment']):
                # Accumulate sentiments for dates with no ifix_value
                buffer_title_sentiment.append(row['title_sentiment'])
                buffer_snippet_sentiment.append(row['snippet_sentiment'])
                buffer_average_sentiment.append(row['average_sentiment'])
            # Skip rows that have no ifix_value (and no sentiment)
            continue
        
        # Handle the base date (ifix_value exists)
        if buffer_title_sentiment:
            # Calculate the averages with both buffer and current row (if current has sentiments)
            if not pd.isna(row['title_sentiment']):
                buffer_title_sentiment.append(row['title_sentiment'])
                buffer_snippet_sentiment.append(row['snippet_sentiment'])
                buffer_average_sentiment.append(row['average_sentiment'])
            
            # Update the row with averaged sentiments
            row['title_sentiment'] = np.mean(buffer_title_sentiment)
            row['snippet_sentiment'] = np.mean(buffer_snippet_sentiment)
            row['average_sentiment'] = np.mean(buffer_average_sentiment)

            # Clear buffer after applying it
            buffer_title_sentiment.clear()
            buffer_snippet_sentiment.clear()
            buffer_average_sentiment.clear()
        
        # Add the updated row to filtered_rows
        filtered_rows.append(row)
    
    # Convert back to DataFrame
    return pd.DataFrame(filtered_rows)

# Apply the sentiment buffer logic to create df_ifix_sent_filtered
df_ifix_sent_filtered = apply_sentiment_buffer(df_ifix_sent)

#%%
# Remove all rows in which there aren't both sentiment and ifix values, 
# so that the data is alligned.

# Before removing NAs, print the number of NaN values under sentiment columns:
na_count = df_ifix_sent_filtered['average_sentiment'].isna().sum()
print(f"Number of NAs in 'average_sentiment': {na_count}")

df_ifix_sent_filtered.dropna(subset=['ifix_value', 'average_sentiment'], inplace=True)

#%%
# Round sentiment columns to 5 decimal places
df_ifix_sent_filtered['title_sentiment'] = df_ifix_sent_filtered['title_sentiment'].round(5)
df_ifix_sent_filtered['snippet_sentiment'] = df_ifix_sent_filtered['snippet_sentiment'].round(5)
df_ifix_sent_filtered['average_sentiment'] = df_ifix_sent_filtered['average_sentiment'].round(5)

#%%
# Saving as CSV file
df_ifix_sent_filtered.to_csv("csv_output/d_ifix_sentiment.csv", index=False)