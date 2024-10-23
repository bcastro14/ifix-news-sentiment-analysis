### Initial setup:
import pandas as pd
import matplotlib.pyplot as plt

# Define a function to check if "brazil" or "brasil" is present in the text (case insensitive)
def contains_brazil_or_brasil(text):
    text = text.lower()
    return 'brazil' in text or 'brasil' in text

# Define a function to check for for some names (case insensitive)
def contains_string(text):
    strings = ['petrobras', 'sao paulo', 'são paulo', 'rio de janeiro', 'brasilia', 'brasília']
    return any(item in text.lower() for item in strings)

#%%
#################### Bloomberg processing ####################

# Load the CSV file
df = pd.read_csv('csv_output/a_news_result_bloomberg.csv')

# Drop the 'position' column
df = df.drop(columns=['position'])

#%%
# Step 1: Filter rows by 'source' field
valid_sources = ['Bloomberg.com', 'Bloomberg']
filtered_df = df[df['source'].isin(valid_sources)]
excluded_df_source = df[~df['source'].isin(valid_sources)]

print(f"Number of rows excluded by source: {len(excluded_df_source)}")

#%%
# Step 2: Filter rows where 'brazil' or 'brasil' appears in the title, snippet, or link fields
# and also checks for other specific words.
excluded_df_brazil = filtered_df[
    ~(filtered_df['title'].apply(contains_brazil_or_brasil) | 
      filtered_df['snippet'].apply(contains_brazil_or_brasil) | 
      filtered_df['link'].apply(contains_brazil_or_brasil) |
      filtered_df['title'].apply(contains_string) | 
      filtered_df['snippet'].apply(contains_string))
]

filtered_df = filtered_df[
    filtered_df['title'].apply(contains_brazil_or_brasil) | 
    filtered_df['snippet'].apply(contains_brazil_or_brasil) | 
    filtered_df['link'].apply(contains_brazil_or_brasil) |
    filtered_df['title'].apply(contains_string) | 
    filtered_df['snippet'].apply(contains_string)
]

print(f"Number of rows excluded considering the specified strings: {len(excluded_df_brazil)}")

#%%
# Step 3: Filter rows by 'link' field and move matching rows to excluded_df_link
invalid_links = [
    "https://www.bloomberg.com/company/",
    "https://www.bloomberg.com/en/",
    "https://www.bloomberg.com/view/"
]
excluded_df_link = filtered_df[filtered_df['link'].str.contains('|'.join(invalid_links), na=False)]
filtered_df = filtered_df[~filtered_df['link'].str.contains('|'.join(invalid_links), na=False)]

print(f"Number of rows excluded by invalid link: {len(excluded_df_link)}")

#%%
# Step 4: Check for duplicates in 'title' and 'snippet' (case-insensitive) and remove them
filtered_df['title_lower'] = filtered_df['title'].str.lower()
filtered_df['snippet_lower'] = filtered_df['snippet'].str.lower()

# Identify duplicates based on title and snippet
duplicated_rows = filtered_df.duplicated(subset=['title_lower', 'snippet_lower'], keep='first')
excluded_df_duplicates = filtered_df[duplicated_rows]

# Remove duplicates from the main dataframe
filtered_df = filtered_df[~duplicated_rows]

# Drop the temporary 'title_lower' and 'snippet_lower' columns
filtered_df.drop(['title_lower', 'snippet_lower'], axis=1, inplace=True)

print(f"Number of rows excluded by being duplicates: {len(excluded_df_duplicates)}")

#%%
# Save the filtered CSV:
filtered_df.to_csv('csv_output/b_bloomberg.csv', index=False)



#%%
#################### Reuters processing ####################
# Load the CSV file
df2 = pd.read_csv('csv_output/a_news_result_reuters.csv')

# Drop the 'position' column
df2 = df2.drop(columns=['position'])

#%%
# Step 1: Filter rows by 'source' field
valid_sources2 = ['Reuters']
filtered_df2 = df2[df2['source'].isin(valid_sources2)]
excluded_df_source2 = df2[~df2['source'].isin(valid_sources2)]

print(f"Number of rows excluded by source: {len(excluded_df_source2)}")

#%%
# Step 2: Filter rows where 'brazil' or 'brasil' appears in the title, snippet, or link fields
# and also checks for other specific words.
excluded_df_brazil2 = filtered_df2[
    ~(filtered_df2['title'].apply(contains_brazil_or_brasil) | 
      filtered_df2['snippet'].apply(contains_brazil_or_brasil) | 
      filtered_df2['link'].apply(contains_brazil_or_brasil) |
      filtered_df2['title'].apply(contains_string) | 
      filtered_df2['snippet'].apply(contains_string))
]
filtered_df2 = filtered_df2[
    filtered_df2['title'].apply(contains_brazil_or_brasil) | 
    filtered_df2['snippet'].apply(contains_brazil_or_brasil) | 
    filtered_df2['link'].apply(contains_brazil_or_brasil) |
    filtered_df2['title'].apply(contains_string) | 
    filtered_df2['snippet'].apply(contains_string)
]

print(f"Number of rows excluded considering the specified strings: {len(excluded_df_brazil2)}")

#%%
# Step 3: Filter rows by 'link' field and move matching rows to excluded_df_link
invalid_links2 = [
    "https://www.reuters.com/sports/",
    "https://www.reuters.com/science/",
    "https://www.reuters.com/pictures/",
    "https://www.reuters.com/news/picture",
    "https://www.reuters.com/lifestyle/",
    "https://www.reuters.com/investigates/",
    "https://www.reuters.com/graphics/",
    "https://www.reuters.com/fact-check/",
    "https://www.reuters.com/article/sports",
    "https://www.reuters.com/article/lifestyle",
    "https://widerimage.reuters.com/",
    "http://www.reuters.com/investigates/",
    "https://www.reuters.com/article/id"
]
excluded_df_link2 = filtered_df2[filtered_df2['link'].str.contains('|'.join(invalid_links2), na=False)]
filtered_df2 = filtered_df2[~filtered_df2['link'].str.contains('|'.join(invalid_links2), na=False)]

print(f"Number of rows excluded by invalid link: {len(excluded_df_link2)}")

#%%
# Step 4: Check for duplicates in 'title' and 'snippet' (case-insensitive) and remove them
filtered_df2['title_lower'] = filtered_df2['title'].str.lower()
filtered_df2['snippet_lower'] = filtered_df2['snippet'].str.lower()

# Identify duplicates based on title and snippet
duplicated_rows2 = filtered_df2.duplicated(subset=['title_lower', 'snippet_lower'], keep='first')
excluded_df_duplicates2 = filtered_df2[duplicated_rows2]

# Remove duplicates from the main dataframe
filtered_df2 = filtered_df2[~duplicated_rows2]

# Drop the temporary 'title_lower' and 'snippet_lower' columns
filtered_df2.drop(['title_lower', 'snippet_lower'], axis=1, inplace=True)

print(f"Number of rows excluded by being duplicates: {len(excluded_df_duplicates2)}")

#%%
# Save the filtered CSV:
filtered_df2.to_csv('csv_output/b_reuters.csv', index=False)



#%%
#################### Combined Processing ####################

# Load both CSVs into dataframes if not done yet
filtered_df = pd.read_csv('csv_output/b_bloomberg.csv')
filtered_df2 = pd.read_csv('csv_output/b_reuters.csv')

#%%
# Concatenate the two dataframes
combined_df = pd.concat([filtered_df, filtered_df2])

# Sort the combined dataframe by 'absolute_date' in descending order
combined_df = combined_df.sort_values(by='absolute_date', ascending=False)

#%%
###### Plot graph News x Month:
# Ensure 'absolute_date' is in datetime format
combined_df['absolute_date'] = pd.to_datetime(combined_df['absolute_date'])

# Group by month and count the number of news items per month
monthly_count = combined_df.groupby(combined_df['absolute_date'].dt.to_period('M')).size()

# Create the plot
plt.figure(figsize=(12, 6), dpi=120)
ax = monthly_count.plot(kind='bar', color='skyblue', width=0.8)

# Add labels to each bar
for i, count in enumerate(monthly_count):
    if i % 2 == 0:  # Show label for every other bar
        ax.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=8)

# Set the tick frequency to show labels every 2 months
ax.set_xticks(range(0, len(monthly_count), 2))
ax.set_xticklabels(monthly_count.index[::2].strftime('%Y-%m'), rotation=45, 
                   ha='right', fontsize=12)

# Add horizontal grid lines
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Title and labels
plt.title('Notícias por mês', fontsize=16)
plt.xlabel('Mês', fontsize=14)
plt.ylabel('Quantidade de notícias', fontsize=14)

# Ensure labels are readable by adding some padding to the layout
plt.tight_layout()

plt.show()

#%%
##### Remove all news from dates before the specified cutoff_date:
# Define the cutoff date
cutoff_date = pd.Timestamp('2021-12-01')

# Filter the dataframe to remove rows where 'absolute_date' is earlier than December 1, 2021
excluded_df_time = combined_df[combined_df['absolute_date'] < cutoff_date]
print(f"Number of rows excluded because of the defined cutoff date: {len(excluded_df_time)}")


combined_df = combined_df[combined_df['absolute_date'] >= cutoff_date]
#%%
# Export the sorted dataframe to a new CSV file
combined_df.to_csv('csv_output/b_combined_news.csv', index=False)


