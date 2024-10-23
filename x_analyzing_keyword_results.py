import pandas as pd
import os

# Define the folder containing the CSV files
folder_path = 'analyzing_keywords_folder/'

# Get a list of all CSV files in the folder, sorted alphabetically
file_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')])

# Initialize an empty list to hold the dataframes
dfs = []

# Loop over each CSV, load it, and add 'in_dfX' columns dynamically
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    
    # Create columns 'in_df1' to 'in_df7' (or more depending on the number of CSVs)
    df[f'in_df{i+1}'] = True  # Mark the current dataframe with True
    for j in range(1, len(file_paths) + 1):
        if j != i + 1:
            df[f'in_df{j}'] = False  # Mark other columns as False
    
    dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs)

# Drop the 'position' column if it exists
if 'position' in combined_df.columns:
    combined_df = combined_df.drop(columns=['position'])

# Define how to aggregate columns, for in_df1 to in_dfX, use 'max'
agg_dict = {f'in_df{i+1}': 'max' for i in range(len(file_paths))}

# For other columns, use 'first'
agg_dict.update({
    col: 'first' for col in combined_df.columns if col not in agg_dict and col != 'title'
})

# Group by 'title' and apply the aggregation
combined_df = combined_df.groupby('title', as_index=False).agg(agg_dict)

# Save the result to a CSV (optional)
combined_df.to_csv('combined.csv', index=False)