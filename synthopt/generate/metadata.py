import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta
import os
from scipy.stats import truncnorm
from scipy.stats import skew
from scipy.stats import skewnorm, multivariate_normal
from numpy.linalg import cholesky
from sklearn.preprocessing import LabelEncoder
import random
import string
from datetime import datetime
import calendar
import warnings
warnings.filterwarnings('ignore')

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

def calculate_average_length(df, columns):
  results = []
  for column in columns:
    char_lengths = []
    space_lengths = []
    for value in df[column]:
      if isinstance(value, str):
        char_lengths.append(len(value))
        space_lengths.append(value.count(" "))
    avg_char_length = sum(char_lengths) / len(char_lengths) if char_lengths else 0
    avg_space_length = sum(space_lengths) / len(space_lengths) if space_lengths else 0

    results.append({
        "column": column,
        "avg_char_length": avg_char_length,
        "avg_space_length": avg_space_length,
    })
  return results


def metadata_process(data, type="correlated"):
    def process_single_dataframe(data, table_name=None):
        metadata = pd.DataFrame(columns=['variable_name', 'datatype', 'completeness', 'values', 'mean', 'standard_deviation', 'skew', 'table_name'])

        # Convert floats that are actually integers
        for column in data.select_dtypes(include='float'):
            if (data[column].dropna() % 1 == 0).all():
                data[column] = data[column].astype("Int64")

        # Identify non-numerical columns
        #non_numerical_columns = list(set(data.columns) - set(data.describe().columns))
        non_numerical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
        date_columns = []
        for column in non_numerical_columns:
            try:
                pd.to_datetime(data[column])
                date_columns.append(column)
            except ValueError:
                pass

        # Identify string/object columns
        all_string_columns = list(set(non_numerical_columns) - set(date_columns))
        categorical_string_columns = []
        for column in data[all_string_columns].columns:
            if data[all_string_columns][column].nunique() < len(data[all_string_columns]) * 0.2:
                categorical_string_columns.append(column)
        non_categorical_string_columns = list(set(all_string_columns) - set(categorical_string_columns))
        
        # Calculate average lengths of non-categorical strings
        average_lengths_df = calculate_average_length(data, non_categorical_string_columns)

        # Encode categorical strings
        orig_data = data.copy()
        le = LabelEncoder()
        for column in categorical_string_columns:
            data[column] = le.fit_transform(data[column])

        # Handle date columns by expanding them
        #for col in date_columns:
        #    if col in data.columns:
        #        data[col] = pd.to_datetime(data[col], errors='coerce')
        for column in date_columns:
            if not pd.to_datetime(data[column], errors='coerce', infer_datetime_format=True).isna().all():
                data[column] = pd.to_datetime(data[column], errors='coerce', infer_datetime_format=True)
                data[column + '_year'] = data[column].dt.year
                data[column + '_month'] = data[column].dt.month
                data[column + '_day'] = data[column].dt.day
                data.insert(data.columns.get_loc(column) + 1, column + '_year', data.pop(column + '_year'))
                data.insert(data.columns.get_loc(column) + 2, column + '_month', data.pop(column + '_month'))
                data.insert(data.columns.get_loc(column) + 3, column + '_day', data.pop(column + '_day'))

                data = data.drop(columns=[column], axis=1)

        #data = data.drop(date_columns, axis=1)


        # Create metadata for each column
        for column in data.columns:
            completeness = (data[column].notna().sum() / len(data)) * 100
            if column in non_categorical_string_columns:
                value_range = None
                mean = next((item['avg_char_length'] for item in average_lengths_df if item['column'] == column), None)
                std_dev = next((item['avg_space_length'] for item in average_lengths_df if item['column'] == column), None)
                skewness_value = None
            else:
                try:
                    value_range = (data[column].min(), data[column].max())
                except Exception:
                    value_range = None
                try:
                    mean = data[column].mean()
                    std_dev = data[column].std()
                except Exception:
                    mean = None
                    std_dev = None
                try:
                    skewness_value = skew(data[column])
                except Exception:
                    skewness_value = None
                
            new_row = pd.DataFrame({
                'variable_name': [column],
                'datatype': [data[column].dtype],
                'completeness': [completeness],
                'values': [value_range],
                'mean': [mean],
                'standard_deviation': [std_dev],
                'skew': [skewness_value],
                'table_name': [table_name] if table_name else [None]
            })
            metadata = pd.concat([metadata, new_row], ignore_index=True)

        # Create label mapping for categorical variables with table name prefix
        label_mapping = {}
        for column in categorical_string_columns:
            prefixed_column = f"{table_name}.{column}" if table_name else column  # Add table name prefix
            label_mapping[prefixed_column] = dict(zip(le.fit_transform(orig_data[column].unique()), orig_data[column].unique()))

        return metadata, label_mapping, data

    # If the input is a dictionary, process each table individually
    if isinstance(data, dict):
        combined_metadata = pd.DataFrame()
        combined_label_mapping = {}
        combined_data = pd.DataFrame()

        for table_name, df in data.items():
            table_metadata, table_label_mapping, processed_data = process_single_dataframe(df, table_name)
            combined_metadata = pd.concat([combined_metadata, table_metadata], ignore_index=True)
            
            # Update with the new label mapping, flattening it
            for key, value in table_label_mapping.items():
                combined_label_mapping[key] = value  # Add the prefixed key directly

            # Prefix columns with table name to prevent conflicts
            processed_data.columns = [f"{table_name}.{col}" for col in processed_data.columns]
            combined_data = pd.concat([combined_data, processed_data], axis=1)

        # Correlation across combined numerical data
        combined_numerical_data = combined_data.select_dtypes(include=['number'])
        correlation_matrix = combined_numerical_data.corr()

        if type == "correlated":
            return combined_metadata, combined_label_mapping, correlation_matrix
        elif type == "structural":
            return combined_metadata[['variable_name', 'datatype', 'completeness', 'values', 'table_name']], combined_label_mapping
        else:
            return combined_metadata, combined_label_mapping
    else:
        # If input is a single DataFrame, process directly
        metadata, label_mapping, processed_data = process_single_dataframe(data)

        # FIX
        numerical_data = processed_data.select_dtypes(include=['number'])
        correlation_matrix = numerical_data.corr()

        #correlation_matrix = processed_data.corr() if type == "correlated" else None

        # if statistical or structural then only return metadata with columns needed (metadata[columns])

        if type == "correlated":
            return metadata, label_mapping, correlation_matrix
        elif type == "structural":
            return metadata[['variable_name', 'datatype', 'completeness', 'values', 'table_name']], label_mapping
        else:
            return metadata, label_mapping



# Function to generate random data based on metadata for each filename
def generate_structural_data(metadata, label_mapping=None, num_records=100, identifier_column=None):
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

        # Check if value_range is valid
        if pd.isna(value_range) or value_range == "None":
            if 'object' in str(dtype):
                return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10)))
            elif 'int' in str(dtype):
                return random.randint(0, 100)
            elif 'float' in str(dtype):
                return round(random.uniform(0.0, 100.0), 2)
        else:
            if 'object' in str(dtype):
                return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10)))
            else:
                try:
                    if isinstance(value_range, str):
                        value_range = eval(value_range)  # Evaluate the string representation of a tuple/list
                    if isinstance(value_range, (tuple, list)) and len(value_range) == 2:
                        if 'int' in str(dtype):
                            # Special case for binary values
                            if value_range == (0, 1):
                                return random.choice([0, 1])  # For binary values, return either 0 or 1
                            return random.randint(value_range[0], value_range[1])
                        elif 'float' in str(dtype):
                            return round(random.uniform(value_range[0], value_range[1]), 2)
                except Exception as e:
                    print(f"Error parsing values: {e}")
                    return None

    # Loop through each table and generate its data
    for table_name, variables in table_variable_mapping.items():
        generated_data[table_name] = {}
        
        for row in variables:
            column_name = row['variable_name']
            data = []

            # Generate data for the current variable
            for _ in range(num_records):
                value = generate_random_value(row)
                data.append(value)

            # Handle completeness
            completeness = row['completeness']
            if completeness < 100.0:
                num_missing = int(num_records * (1 - (completeness / 100.0)))
                missing_indices = random.sample(range(num_records), num_missing)
                for idx in missing_indices:
                    data[idx] = None
            
            generated_data[table_name][column_name] = data

        # Create DataFrame for the current table
        generated_data[table_name] = pd.DataFrame(generated_data[table_name])

        # Handle date combination and avoid duplications
        date_columns = {}
        for col in generated_data[table_name].columns:
            if col.endswith('_year'):
                base_name = col[:-5]
                if base_name not in date_columns:
                    date_columns[base_name] = {}
                date_columns[base_name]['year'] = col
            elif col.endswith('_month'):
                base_name = col[:-6]
                if base_name not in date_columns:
                    date_columns[base_name] = {}
                date_columns[base_name]['month'] = col
            elif col.endswith('_day'):
                base_name = col[:-4]
                if base_name not in date_columns:
                    date_columns[base_name] = {}
                date_columns[base_name]['day'] = col

        # Create a list to track the original variable order
        original_order = list(generated_data[table_name].columns)

        base_names = []

        combined_date_cols = {}
        for base_name, components in date_columns.items():
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

        new_columns_order = []
        added_base_names = set()  # Track columns from base_names that have been added
        for col in original_order:
            if col in combined_date_cols:
                # Use the combined datetime column
                new_col = combined_date_cols[col]
            else:
                # Retain the original column
                new_col = col
            # Check if the column is in base_names and has already been added
            if new_col in base_names and new_col in added_base_names:
                continue  # Skip if already added
            # Add the column to the new order
            new_columns_order.append(new_col)
            # Track the column if it's in base_names
            if new_col in base_names:
                added_base_names.add(new_col)
        # Set the DataFrame columns in the new order
        generated_data[table_name] = generated_data[table_name][new_columns_order]


        # Apply label mapping if provided
        if label_mapping:
            for col in generated_data[table_name].columns:
                full_key = f"{table_name}.{col}"
                if full_key in label_mapping:
                    # Map the values, ensuring NaN values are handled correctly
                    generated_data[table_name][col] = generated_data[table_name][col].map(label_mapping[full_key]).where(
                        generated_data[table_name][col].notna(), np.nan)
                    
        if identifier_column != None:
            participant_ids_integer = [random_integer() for _ in range(num_records)] 
            for column in generated_data[table_name].columns:
                if identifier_column in column:
                    generated_data[table_name][column] = participant_ids_integer

    if single == True:
        return generated_data['single']
    else:
        return generated_data
    



# Function to generate correlated samples with truncated bounds using rejection sampling
def generate_truncated_multivariate_normal(mean, cov, lower, upper, size):
    # Initialize samples array
    samples = []

    lower = np.array(lower)
    upper = np.array(upper)

    # Loop until the required number of samples is obtained
    while len(samples) < size:
        # Draw a batch of multivariate normal samples
        batch_size = size - len(samples)
        candidate_samples = np.atleast_2d(multivariate_normal.rvs(mean=mean, cov=cov, size=batch_size))

        # Apply truncation: Keep only samples within the bounds for all variables
        within_bounds = np.all((candidate_samples >= lower) & (candidate_samples <= upper), axis=1)

        valid_samples = candidate_samples[within_bounds]

        # Append the valid samples to our final sample list
        samples.extend(valid_samples)

    # Convert to a numpy array of the desired size
    return np.array(samples[:size])



def generate_correlated_data(metadata, correlation_matrix, num_records=100, identifier_column=None, label_mapping={}):

    metadata['variable_name'] = metadata.apply(lambda x: f"{x['table_name']}.{x['variable_name']}", axis=1)

    # Number of samples to generate
    num_rows = num_records

    def is_int_or_float(datatype):
        return pd.api.types.is_integer_dtype(datatype) or pd.api.types.is_float_dtype(datatype)

    numerical_metadata = metadata[metadata['datatype'].apply(is_int_or_float)]
    non_numerical_metadata = metadata[~metadata['datatype'].apply(is_int_or_float)]

    # Initialize lists to store means, std_devs, and value ranges
    means = []
    std_devs = []
    variable_names = []
    lower_bounds = []
    upper_bounds = []

    # Collect means, standard deviations, and value ranges for each variable
    for i, (index, row) in enumerate(numerical_metadata.iterrows()):
        means.append(row['mean'])
        std_devs.append(row['standard_deviation'])
        variable_names.append(row['variable_name'])
        lower, upper = row['values']
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    # Create the covariance matrix using the correlation and standard deviations
    covariance_matrix = np.diag(std_devs) @ correlation_matrix @ np.diag(std_devs)

    # Generate truncated multivariate normal data
    synthetic_samples = generate_truncated_multivariate_normal(
        mean=means,
        cov=covariance_matrix,
        lower=lower_bounds,
        upper=upper_bounds,
        size=num_rows
    )

    # Convert samples into a Pandas DataFrame
    synthetic_data = pd.DataFrame(synthetic_samples, columns=variable_names)

    # Introduce missing values (NaNs) according to the completeness percentages
    for i, (index, row) in enumerate(numerical_metadata.iterrows()):
        completeness = row['completeness'] / 100  # Convert to a decimal
        num_valid_rows = int(num_rows * completeness)  # Number of valid rows based on completeness

        # Randomly set some of the data to NaN based on completeness
        if completeness < 1.0:
            nan_indices = np.random.choice(num_rows, size=(num_rows - num_valid_rows), replace=False)
            synthetic_data.iloc[nan_indices, i] = np.nan

    for column in synthetic_data.columns:
        # Find the corresponding datatype in the metadata
        datatype = metadata.loc[metadata['variable_name'] == column, 'datatype'].values
        if len(datatype) > 0 and "int" in str(datatype[0]).lower():   #if len(datatype) > 0 and np.issubdtype(datatype[0], np.integer):
            # Round the values in the column if the datatype is an integer
            synthetic_data[column] = synthetic_data[column].round()# .astype(int)

    if metadata['table_name'].iloc[0] is not None:
        # label mapping
        for column, mapping in label_mapping.items():
            synthetic_data[column] = synthetic_data[column].map(mapping)

    # date combine
    # Identify columns that match the pattern *_year, *_month, *_day
    date_cols = {}
    
    for col in synthetic_data.columns:
        if col.endswith('_year'):
            base_name = col[:-5]
            date_cols.setdefault(base_name, {})['year'] = col
        elif col.endswith('_month'):
            base_name = col[:-6]
            date_cols.setdefault(base_name, {})['month'] = col
        elif col.endswith('_day'):
            base_name = col[:-4]
            date_cols.setdefault(base_name, {})['day'] = col

    # Combine identified columns into a new date column
    for base_name, cols in date_cols.items():
        if 'year' in cols and 'month' in cols and 'day' in cols:
            # Create the new date column with error handling
            synthetic_data[base_name] = pd.to_datetime(
                synthetic_data[[cols['year'], cols['month'], cols['day']]].rename(
                    columns={cols['year']: 'year', cols['month']: 'month', cols['day']: 'day'}
                ),
                errors='coerce'  # Convert invalid dates to NaT
            )
            
            # Drop the original year, month, and day columns
            synthetic_data.drop(columns=[cols['year'], cols['month'], cols['day']], inplace=True)#

    # free text handling!!
    for index, row in non_numerical_metadata.iterrows():
        column_name = row['variable_name']
        mean = row['mean']
        std_dev = row['standard_deviation']
    
        # Check if mean and std_dev are not NaN
        if not pd.isna(mean) and not pd.isna(std_dev):
            # Call the generate_random_string function and assign the result to the data
            synthetic_data[column_name] = [generate_random_string(mean, std_dev) for _ in range(len(synthetic_data))]

    def strip_suffix(variable_name):
        if variable_name.endswith('_year'):
            return variable_name[:-5]  # Remove the '_year' suffix
        elif variable_name.endswith('_month'):
            return variable_name[:-6]  # Remove the '_month' suffix
        elif variable_name.endswith('_day'):
            return variable_name[:-4]  # Remove the '_day' suffix
        else:
            return variable_name
    # Apply the function to create a new column for base names
    metadata_temp = metadata.copy()
    metadata_temp['base_name'] = metadata['variable_name'].apply(strip_suffix)
    # Get unique base names
    unique_variable_names = metadata_temp['base_name'].unique().tolist()
    synthetic_data = synthetic_data[unique_variable_names]

    if identifier_column != None:
        participant_ids_integer = [random_integer() for _ in range(num_records)] 
        #synthetic_data = synthetic_data.drop(columns=[identifier_column])
        #synthetic_data.insert(0,identifier_column,participant_ids_integer)

        for column in synthetic_data.columns:
            if column.endswith('.' + identifier_column):
                synthetic_data[column] = participant_ids_integer


    # Remove the prefixes
    dataframes_dict = {}
    for column in synthetic_data.columns:
        prefix = column.split('.')[0]  
        if prefix not in dataframes_dict:
            prefix_columns = [col for col in synthetic_data.columns if col.startswith(prefix)]            
            new_df = synthetic_data[prefix_columns].copy()            
            new_df.columns = [col[len(prefix) + 1:] for col in new_df.columns]  # Remove prefix           
            if metadata['table_name'].iloc[0] is None:
                # label mapping
                for column, mapping in label_mapping.items():
                    new_df[column] = new_df[column].map(mapping)

            dataframes_dict[prefix] = new_df
    synthetic_data = dataframes_dict

    if metadata['table_name'].iloc[0] is None:
        synthetic_data = synthetic_data['None']

    return synthetic_data
