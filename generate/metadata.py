import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta
import os

# Function to generate a random string
def random_string(length=6):
    return ''.join(random.choices(string.ascii_letters, k=length))

def random_integer(length=6):
    # Generate a random integer between 10^(length-1) and 10^length - 1
    return random.randint(10**(length-1), (10**length) - 1)

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
# NEED TO FIX DATE AND TIME
def generate_metadata(metadata_csv, num_records=100, save_location=None):
    try:
        # Load the metadata CSV
        try:
            metadata = pd.read_csv(metadata_csv)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata CSV file '{metadata_csv}' not found. Please check the file path.")
        except pd.errors.EmptyDataError:
            raise ValueError("The provided metadata CSV is empty or corrupted.")
        except Exception as e:
            raise IOError(f"An error occurred while reading the metadata CSV: {str(e)}")

        # Check if the required columns exist in the metadata
        required_columns = ['filename', 'variable name', 'data type', 'values', 'variable description']
        for col in required_columns:
            if col not in metadata.columns:
                raise ValueError(f"Missing required column '{col}' in the metadata CSV.")

        # Get unique filenames
        filenames = metadata['filename'].unique()

        # Generate unique participant IDs
        participant_ids_string = [random_string() for _ in range(num_records)]
        participant_ids_integer = [random_string() for _ in range(num_records)]  # Should this be random_integer instead?

        # Dictionary to store dataframes for each filename
        datasets = {}

        for filename in filenames:
            # Filter the metadata for the current filename
            file_metadata = metadata[metadata['filename'] == filename]

            # Create an empty dictionary to store the generated data
            data = {}

            for _, row in file_metadata.iterrows():
                try:
                    col_name = row['variable name']
                    data_type = row['data type']
                    value_range = row['values']

                    # Handle missing or NaN value_range
                    if pd.isna(value_range) or value_range.strip() == '':
                        if data_type == 'string':
                            data[col_name] = [random_string() for _ in range(num_records)]
                        else:
                            raise ValueError(f"Missing or invalid value range for column '{col_name}' with data type '{data_type}'")
                        continue

                    # Generate random data based on the data type
                    if data_type == 'integer':
                        if 'Participant ID' in row['variable description']:
                            data[col_name] = participant_ids_integer
                        else:
                            min_val, max_val = parse_range(value_range)
                            if min_val is None or max_val is None:
                                raise ValueError(f"Invalid range specified for integer column '{col_name}'")
                            data[col_name] = np.random.randint(min_val, max_val + 1, num_records)

                    elif data_type == 'float':
                        min_val, max_val = parse_range(value_range)
                        if min_val is None or max_val is None:
                            raise ValueError(f"Invalid range specified for float column '{col_name}'")
                        data[col_name] = np.random.uniform(min_val, max_val, num_records)

                    elif data_type == 'category':
                        values = value_range.strip('[]').split(', ')
                        if not values:
                            raise ValueError(f"Category column '{col_name}' has no valid categories listed.")
                        data[col_name] = [random.choice(values) for _ in range(num_records)]

                    elif data_type == 'string':
                        if 'Participant ID' in row['variable description']:
                            data[col_name] = participant_ids_string
                        else:
                            data[col_name] = [random_string() for _ in range(num_records)]

                    elif data_type == 'date':
                        try:
                            start_date, end_date = value_range.split('to')
                            data[col_name] = [random_date(start_date.strip(), end_date.strip()).strftime("%d/%m/%Y") for _ in range(num_records)]
                        except ValueError:
                            raise ValueError(f"Invalid date range specified for date column '{col_name}'")

                except Exception as e:
                    print(f"Error processing column '{col_name}' in file '{filename}': {str(e)}")
                    continue

            # Convert the data dictionary into a pandas DataFrame
            datasets[filename] = pd.DataFrame(data)

            output_filename = os.path.basename(filename)

            # Save the dataframe as a CSV file, using the filename (from metadata) to name it
            if save_location is not None:
                try:
                    os.makedirs(save_location, exist_ok=True)  # Ensure directory exists
                    datasets[filename].to_csv(f"{save_location}/{output_filename}_synthetic.csv", index=False)
                except PermissionError:
                    raise PermissionError(f"Permission denied: Unable to save the file in the specified location: {save_location}")
                except FileNotFoundError:
                    raise FileNotFoundError(f"Save location '{save_location}' not found.")
                except Exception as e:
                    raise IOError(f"Could not save the file '{output_filename}_synthetic.csv'. Error: {str(e)}")

            print(f"Generated: {output_filename}_synthetic.csv")

        return datasets

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
    except KeyError as ke:
        print(f"KeyError: {str(ke)}")
    except IOError as ioe:
        print(f"IOError: {str(ioe)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
