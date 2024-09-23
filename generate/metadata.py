import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta
import os

# CHANGE SO THAT VARIABLES WITH PARTICIPANT ID IN VAR DESC REMAINS UNIQUE ACROSS FILES

# Function to generate a random string
def random_string(length=6):
    return ''.join(random.choices(string.ascii_letters, k=length))

# Function to generate random dates between a given range
def random_date(start, end):
    start_date = datetime.strptime(start, "%d/%m/%Y")
    end_date = datetime.strptime(end, "%d/%m/%Y")
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)

# Function to parse the value range from a string like '1 to 489'
def parse_range(value_range):
    if 'to' in value_range:
        parts = value_range.split('to')
        return float(parts[0].strip()), float(parts[1].strip())
    return None

# Function to generate random data based on metadata for each filename
def generate_metadata(metadata_csv, num_records=100, save_location=None):
    # Load the metadata CSV
    metadata = pd.read_csv(metadata_csv)

    # Get unique filenames
    filenames = metadata['filename'].unique()

    # Dictionary to store dataframes for each filename
    datasets = {}

    for filename in filenames:
        # Filter the metadata for the current filename
        file_metadata = metadata[metadata['filename'] == filename]

        # Create an empty dictionary to store the generated data
        data = {}

        for _, row in file_metadata.iterrows():
            col_name = row['variable name']
            data_type = row['data type']
            value_range = row['values']

            # If value_range is NaN or empty, skip generating data unless it's a string
            if pd.isna(value_range) or value_range.strip() == '':
                if data_type == 'string':
                    # Generate random strings for string fields, even if values are not specified
                    data[col_name] = [random_string() for _ in range(num_records)]
                continue

            # Generate random data based on the data type
            if data_type == 'integer':
                min_val, max_val = parse_range(value_range)
                data[col_name] = np.random.randint(min_val, max_val + 1, num_records)
            
            elif data_type == 'float':
                min_val, max_val = parse_range(value_range)
                data[col_name] = np.random.uniform(min_val, max_val, num_records)
            
            elif data_type == 'category':
                values = value_range.strip('[]').split(', ')
                data[col_name] = [random.choice(values) for _ in range(num_records)]
            
            elif data_type == 'string':
                if 'id' in row['variable description'].lower():
                    # Ensure uniqueness for IDs
                    data[col_name] = [random_string() for _ in range(num_records)]
                else:
                    data[col_name] = [random_string() for _ in range(num_records)]
            
            elif data_type == 'date':
                start_date, end_date = value_range.split('to')
                data[col_name] = [random_date(start_date.strip(), end_date.strip()).strftime("%d/%m/%Y") for _ in range(num_records)]
        
        # Convert the data dictionary into a pandas DataFrame
        datasets[filename] = pd.DataFrame(data)

        output_filename = os.path.basename(filename)

        if save_location != None:
            # Save the dataframe as a CSV file, using the filename (from metadata) to name it
            try:
                datasets[filename].to_csv(f"{save_location}/{output_filename}_synthetic.csv", index=False)
            except Exception:
                print("Could not save files, please check the save location exists")

        print(f"Generated: {output_filename}_synthetic.csv")

    return datasets
