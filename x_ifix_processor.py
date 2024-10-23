# Processing the exported IFIX data from B3 into a csv with format Date, Value

import csv
from datetime import datetime

# Function to process the value, removing thousands separator and converting decimal comma to dot
def process_value(value):
    if value:
        return value.replace('.', '').replace(',', '.')
    return None

# Read the input CSV
input_file = 'input.csv'
output_file = 'output_ifix.csv'

data = []

with open(input_file, 'r', encoding='ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    
    # Read the year from the first row
    header = next(reader)[0]  # First cell contains "IFIX - YEAR"
    year = header.split('-')[-1].strip()
    
    # Read the month names
    months = next(reader)[1:]  # Skip the first column (Day)
    
    # Iterate through the data rows
    for row in reader:
        if not row or len(row) < 2:  # Skip empty rows or rows with less than 2 elements
            continue
        
        day = row[0]
        if not day.isdigit():  # Skip any footer or non-day rows
            continue
            
        for i, value in enumerate(row[1:], start=1):
            if value:  # Only process non-empty values
                # Format the date as DD/MM/YYYY
                date_str = f'{day.zfill(2)}/{i:02d}/{year}'
                date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                
                # Process the value and append to the data list
                processed_value = process_value(value)
                data.append([date_obj, processed_value])

# Sort the data by the date (first column)
data.sort(key=lambda x: x[0])

# Write the sorted data to the output CSV
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Date', 'Value'])
    
    # Write sorted data
    for date_obj, value in data:
        writer.writerow([date_obj.strftime('%d/%m/%Y'), value])

print(f"Data successfully processed and saved to {output_file}")

