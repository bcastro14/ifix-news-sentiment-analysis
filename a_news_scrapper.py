# Scrapping foreign financial news about Brazil using HasData

#%%
# Importing libraries

import requests
import pandas as pd
import dateparser
from datetime import datetime, timedelta
import time  # For adding delay

#%%
### Defining API details
keyword = 'brazil bloomberg.com'
api_url = 'https://api.hasdata.com/scrape/google/serp'
headers = {'x-api-key': 'API_KEY_HERE'} # Replace your API key here

#%%
### RUNNING THE SCRAPER MULTIPLE TIMES IN DIFFERENT TIME RANGES

# 1. Define the end date (August 21, 2024)
end_date = datetime(2024, 8, 21) #YYYY, MM, DD
day_delta = timedelta(days=30)

all_news = []

#%%
# 2. Loop through the specified day_delta number of days going backwards
# DEFINE HERE HOW MANY LOOPS TO RUN:
loops = 1

for i in range(loops): 
    start_date = end_date - day_delta

    params = {
        'q': keyword,
        'domain': 'google.com',
        'tbm': 'nws',
        'gl': 'us',
        'hl': 'en',
        'tbs': f'cdr:1,cd_min:{start_date.strftime("%m/%d/%Y")},cd_max:{end_date.strftime("%m/%d/%Y")}',
        'num': 100
    }

    try:
        response = requests.get(api_url, params=params, headers=headers)
        if response.status_code == 200:
            multi_data = response.json()
            multi_news = multi_data['newsResults']
            
            # Printing time range and quantity of news scraped
            print(f"Start date: {start_date.date()}\n" + 
                  f"End date:{end_date.date()}\n" +
                  f"Number of news:{len(multi_news)}\n")
            
            # Parse relative dates to absolute ones
            for item in multi_news:
                item['absolute_date'] = dateparser.parse(item['date'], settings={'RELATIVE_BASE': datetime.now()}).strftime('%Y-%m-%d')
                item['title'] = item['title'].replace("\n", "").replace("  ", " ")
                item['snippet'] = item['snippet'].replace("\n", "")  .replace("  ", " ")      

            all_news.extend(multi_news)  # Collect all news
        else:
            print(f'Error: {response.status_code}')

    except Exception as e:
        print('Error:', e)

    # Delay between requests to avoid overwhelming the server
    time.sleep(2)  # 2-second delay between API calls
    
    # Update the end date for the next 15-day chunk
    end_date = start_date - timedelta(days=1)

#%%
# Save all the news to CSV
df = pd.DataFrame(all_news)
# df.to_csv("csv_output/a_news_result_reuters.csv", index=False)
df.to_csv("csv_output/a_news_result_bloomberg.csv", index=False)
