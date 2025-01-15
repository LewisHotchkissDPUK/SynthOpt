import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta
import random
import string
from datetime import datetime
import calendar
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

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

def generate_random_string(avg_char_length, avg_space_length):
  num_chars = int(round(avg_char_length))
  num_spaces = int(round(avg_space_length))
  random_string = ''.join(random.choice(string.ascii_letters) for i in range(num_chars - num_spaces))
  for i in range(num_spaces):
    random_string = random_string[:random.randint(0, len(random_string))] + ' ' + random_string[random.randint(0, len(random_string)):]

  return random_string






def structural_data(metadata, label_mapping=None, num_records=100, identifier_column=None):

    metadata = metadata.copy()
    single = False
    if metadata['table_name'].iloc[0] is None:
        single = True
        metadata['table_name'] = 'single'

    # Initialize a dictionary to hold generated data for each table
    generated_data = {}

    # Create a mapping for each table to handle variable generation
    table_variable_mapping = {}
    
    for index, row in metadata.iterrows():
        table_name = row['table_name']
        variable_name = row['variable_name']
        
        # Initialize the table if it doesn't exist
        if table_name not in table_variable_mapping:
            table_variable_mapping[table_name] = []

        # Append variable row details to the specific table
        table_variable_mapping[table_name].append(row)

    # Function to generate a random value based on the metadata row
    def generate_random_value(row):
        dtype = row['datatype']
        value_range = row['values']

        if 'object' in str(dtype):
            return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10)))
        else:
            try:
                if isinstance(value_range, str):
                    value_range = eval(value_range)  # Evaluate the string representation of a tuple/list

                if isinstance(value_range, (tuple, list)): # and len(value_range) == 2
                    if 'int' in str(dtype).lower():
                        # Special case for binary values
                        if value_range == (0, 1):
                            return random.choice([0, 1])  # For binary values, return either 0 or 1
                        
                        if isinstance(value_range, list) and all(isinstance(v, (int, float)) for v in value_range): # if the value range is a list [1,2,3] instead of a tuple (1,3) pick a random value from list
                            return random.choice(value_range)
                        
                        return random.randint(value_range[0], value_range[1])
                    
                    elif 'float' in str(dtype):
                        return round(random.uniform(value_range[0], value_range[1]), 2)
                    
            except Exception as e:
                return None
                
    # Loop through each table and generate its data
    for table_name, variables in table_variable_mapping.items():
        generated_data[table_name] = {}
        
        #for row in variables:
        for row in tqdm(variables, desc="Generating Synthetic Data"):
            column_name = row['variable_name']
            data = []

            # Generate data for the current variable
            for _ in range(num_records):
                value = generate_random_value(row)
                data.append(value)
            
            generated_data[table_name][column_name] = data

        # Create DataFrame for the current table
        generated_data[table_name] = pd.DataFrame(generated_data[table_name])

        # Handle date combination and avoid duplications
        date_columns = {}
        time_columns = {}
        #for col in generated_data[table_name].columns:
        for col in tqdm(generated_data[table_name].columns, desc="Handling Date and Time Columns"):
            if col.endswith('_synthoptyear'):
                base_name = col[:-13]
                if base_name not in date_columns:
                    date_columns[base_name] = {}
                date_columns[base_name]['year'] = col
            elif col.endswith('_synthoptmonth'):
                base_name = col[:-14]
                if base_name not in date_columns:
                    date_columns[base_name] = {}
                date_columns[base_name]['month'] = col
            elif col.endswith('_synthoptday'):
                base_name = col[:-12]
                if base_name not in date_columns:
                    date_columns[base_name] = {}
                date_columns[base_name]['day'] = col

            elif col.endswith('_synthopthour'):
                base_name = col[:-13]
                if base_name not in time_columns:
                    time_columns[base_name] = {}
                time_columns[base_name]['hour'] = col
            elif col.endswith('_synthoptminute'):
                base_name = col[:-15]
                if base_name not in time_columns:
                    time_columns[base_name] = {}
                time_columns[base_name]['minute'] = col
            elif col.endswith('_synthoptsecond'):
                base_name = col[:-15]
                if base_name not in time_columns:
                    time_columns[base_name] = {}
                time_columns[base_name]['second'] = col




        # Create a list to track the original variable order
        original_order = list(generated_data[table_name].columns)


        base_names = []
        combined_date_cols = {}
        #for base_name, components in date_columns.items():
        for base_name, components in tqdm(date_columns.items(), desc="Combining Date Columns"):
            base_names.append(base_name)
            if all(key in components for key in ['year', 'month', 'day']):
                years = generated_data[table_name][components['year']]
                months = generated_data[table_name][components['month']]
                days = generated_data[table_name][components['day']]

                valid_days = []
                for y, m, d in zip(years, months, days):
                    if pd.notna(y) and pd.notna(m):
                        last_day = calendar.monthrange(y, m)[1]
                        valid_days.append(min(d, last_day))
                    else:
                        valid_days.append(None)

                generated_data[table_name][components['day']] = valid_days

                # Combine the date components into a datetime column
                combined_column_name = base_name  # Use base_name as the new datetime column name
                generated_data[table_name][combined_column_name] = pd.to_datetime(
                    generated_data[table_name][[components['year'], components['month'], components['day']]].rename(
                        columns={
                            components['year']: 'year',
                            components['month']: 'month',
                            components['day']: 'day'
                        }),
                    errors='coerce'
                )

                # Drop the original date columns
                generated_data[table_name].drop(columns=[components['year'], components['month'], components['day']], inplace=True)

                # Track combined columns to update table columns list later
                combined_date_cols.update({components['year']: combined_column_name, components['month']: combined_column_name, components['day']: combined_column_name})


        time_base_names = []
        combined_time_cols = {}
        #for time_base_name, components in time_columns.items():
        for time_base_name, components in tqdm(time_columns.items(), desc="Combining Time Columns"):
            time_base_names.append(time_base_name)
            if all(key in components for key in ['hour', 'minute', 'second']):
                hours = generated_data[table_name][components['hour']]
                minutes = generated_data[table_name][components['minute']]
                seconds = generated_data[table_name][components['second']]

                # Handle invalid time components by coercing them to NaT
                combined_times = []
                for h, m, s in zip(hours, minutes, seconds):
                    try:
                        if pd.notna(h) and pd.notna(m) and pd.notna(s):
                            combined_times.append(f"{int(h):02}:{int(m):02}:{int(s):02}")
                        else:
                            combined_times.append(None)
                    except ValueError:
                        combined_times.append(None)

                # Combine the time components into a single time column
                combined_column_name = time_base_name  # Use time_base_name as the new time column name
                generated_data[table_name][combined_column_name] = pd.to_datetime(
                    combined_times, format='%H:%M:%S', errors='coerce'
                ).time

                # Drop the original time columns
                generated_data[table_name].drop(columns=[components['hour'], components['minute'], components['second']], inplace=True)

                # Track combined columns to update table columns list later
                combined_time_cols.update({components['hour']: combined_column_name, components['minute']: combined_column_name, components['second']: combined_column_name})

        
        new_columns_order = []
        added_base_names = set()  # Track columns from base_names that have been added
        #for col in original_order:
        for col in tqdm(original_order, desc="Reordering Synthetic Data"):
            if col in combined_date_cols:
                # Use the combined datetime column
                new_col = combined_date_cols[col]
            elif col in combined_time_cols:
                # Use the combined datetime column
                new_col = combined_time_cols[col]
            else:
                # Retain the original column
                new_col = col
            # Check if the column is in base_names and has already been added
            if new_col in base_names and new_col in added_base_names:
                continue  # Skip if already added
            if new_col in time_base_names and new_col in added_base_names:
                continue
            # Add the column to the new order
            new_columns_order.append(new_col)
            # Track the column if it's in base_names
            if new_col in base_names:
                added_base_names.add(new_col)
            if new_col in time_base_names:
                added_base_names.add(new_col)

        # Set the DataFrame columns in the new order
        generated_data[table_name] = generated_data[table_name][new_columns_order]


        #### Handles combination for date time columns
        columns = generated_data[table_name].columns

        date_columns = [col for col in columns if col.endswith("_synthoptdate")]
        time_columns = [col for col in columns if col.endswith("_synthopttime")]

        # Find common base names
        base_names = set(col.replace("_synthoptdate", "") for col in date_columns).intersection(
            col.replace("_synthopttime", "") for col in time_columns
        )

        # Combine columns
        #for base in base_names:
        for base in tqdm(base_names, desc="Combining Date and Time Columns"):
            date_col = base + "_synthoptdate"
            time_col = base + "_synthopttime"
            combined_col = base
            # Convert date_col and time_col to strings before concatenation
            combined_datetime = pd.to_datetime(generated_data[table_name][date_col].astype(str) + " " + generated_data[table_name][time_col].astype(str))
            generated_data[table_name].insert(generated_data[table_name].columns.get_loc(date_col), combined_col, combined_datetime)

        # Drop the original _date and _time columns
        generated_data[table_name].drop(columns=[col for base in base_names for col in [base + "_synthoptdate", base + "_synthopttime"]], inplace=True)
        #####



        # Apply label mapping if provided
        if label_mapping:
            #for col in generated_data[table_name].columns:
            for col in tqdm(generated_data[table_name].columns, desc="Applying Label Mapping"):
                full_key = f"{table_name}.{col}"  # THIS IS WHY MAPPING ISNT WORKING _ TABLE NAME HASNT BEEN APPENDED FOR SINGLE TABLES
                if full_key in label_mapping:
                    # Map the values, ensuring NaN values are handled correctly
                    generated_data[table_name][col] = generated_data[table_name][col].map(label_mapping[full_key]).where(
                        generated_data[table_name][col].notna(), np.nan)
                    
                if table_name == 'single': ## added to fix the above full_key issue
                    if col in label_mapping:
                        # Map the values, ensuring NaN values are handled correctly
                        generated_data[table_name][col] = generated_data[table_name][col].map(label_mapping[col]).where(
                            generated_data[table_name][col].notna(), np.nan)
                    
        if identifier_column != None:
            participant_ids_integer = [random_integer() for _ in range(num_records)] 
            for column in generated_data[table_name].columns:
                if identifier_column in column:
                    generated_data[table_name][column] = participant_ids_integer
        
        
        
        # NEW COMPLETENESS HANDLING
        #for _, row in metadata.iterrows():
        for _, row in tqdm(metadata.iterrows(), desc="Applying Completeness", total=(len(metadata))):
            column = row['variable_name']
            completeness = row['completeness'] / 100.0
            
            if column.endswith("_synthoptyear"):
                column = column[:-13]

            if column.endswith("_synthopthour"):
                column = column[:-13]

            if column.endswith("_synthoptdate"):
                column = column[:-13]
            
            if column in generated_data[table_name].columns:
                total_values = len(generated_data[table_name][column])
                target_non_nulls = int(total_values * completeness)
                
                current_non_nulls = generated_data[table_name][column].notnull().sum()
                values_to_remove = current_non_nulls - target_non_nulls
                
                if values_to_remove > 0:
                    drop_indices = np.random.choice(
                        generated_data[table_name][generated_data[table_name][column].notnull()].index,
                        size=values_to_remove,
                        replace=False
                    )
                    generated_data[table_name].loc[drop_indices, column] = np.nan

                    

    if single == True:
        return generated_data['single']
    else:
        return generated_data