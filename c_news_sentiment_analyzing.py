import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the Vader sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to calculate sentiment score using Vader
def get_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

#%%
# Load the CSV into a pandas DataFrame
df = pd.read_csv("csv_output/b_combined_news.csv")

#%%
# Apply sentiment analysis to 'title' and 'snippet' columns
df['title_sentiment'] = df['title'].apply(get_sentiment)
df['snippet_sentiment'] = df['snippet'].apply(get_sentiment)

# Create a new column 'average_sentiment' as the mean of 'title_sentiment' and 'snippet_sentiment'
df['average_sentiment'] = df[['title_sentiment', 'snippet_sentiment']].mean(axis=1)

#%%
# Print summary statistics for each sentiment column
df.describe()

#%%
# Save the updated DataFrame to a new CSV file
df.to_csv("csv_output/c_news_with_sentiment.csv", index=False)