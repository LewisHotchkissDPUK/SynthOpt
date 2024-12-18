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
from statsmodels.distributions.copula.api import GaussianCopula, CopulaDistribution
from scipy.stats import norm
import scipy.stats as stats


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


def metadata_process(data, identifier_column=None, type="correlated"):
    def process_single_dataframe(data, table_name=None):
        orig_data_completeness = data.copy()
        data = data.copy()
        metadata = pd.DataFrame(columns=['variable_name', 'datatype', 'completeness', 'values', 'mean', 'standard_deviation', 'skew', 'table_name'])

        non_numerical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
        for column in non_numerical_columns:
            # IF THE OBJECT CANT BE CONVERTED TO NUMBER
            try:
                data[column] = pd.to_numeric(data[column], errors='raise')
                non_numerical_columns = non_numerical_columns.remove(column)

                # If all values are integers, convert to Int64 or int
                if data[column].apply(float.is_integer).all():
                    data[column] = data[column].astype('Int64')  # Use Int64 for nullable support
            except:
                None
        if non_numerical_columns == None:
            non_numerical_columns = []
            
        # Convert floats that are actually integers
        for column in data.select_dtypes(include='float'):
            if (data[column].dropna() % 1 == 0).all():
                data[column] = data[column].astype("Int64")
                if data[column].notna().any():
                    data[column] = data[column].fillna(round(data[column].mean())) #(CHANGE, EFFECTS COMPLETENESS BUT NEEDED FOR COVARIANCE)

        # fill na of numerical columns with mean (CHANGE, EFFECTS COMPLETENESS BUT NEEDED FOR COVARIANCE)
        float_columns = data.select_dtypes(include=['float']).columns
        data[float_columns] = data[float_columns].fillna(data[float_columns].mean())
        
        # Identify non-numerical columns
        #non_numerical_columns = list(set(data.columns) - set(data.describe().columns))
        
        
        
        
        
        ###########################################################################################
        non_numerical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
        date_columns = []
        
        date_formats = [
            "%b-%m","%b-%y","%b-%Y",
            "%B-%m","%B-%y","%B-%Y",
            "%d-%b-%y","%d-%b-%Y","%b-%d-%y","%b-%d-%Y",
            "%d-%B-%y","%d-%B-%Y","%B-%d-%y","%B-%d-%Y",
            "%y-%m-%d","%Y-%m-%d","%m-%d-%y","%m-%d-%Y","%d-%m-%y","%d-%m-%Y",
            "%b/%m","%b/%y","%b/%Y",
            "%B/%m","%B/%y","%B/%Y",
            "%d/%b/%y","%d/%b/%Y","%b/%d/%y","%b/%d/%Y",
            "%d/%B/%y","%d/%B/%Y","%B/%d/%y","%B/%d/%Y",
            "%y/%m/%d","%Y/%m/%d","%m/%d/%y","%m/%d/%Y","%d/%m/%y","%d/%m/%Y",
            "%b.%m","%b.%y","%b.%Y",
            "%B.%m","%B.%y","%B.%Y",
            "%d.%b.%y","%d.%b.%Y","%b.%d.%y","%b.%d.%Y",
            "%d.%B.%y","%d.%B.%Y","%B.%d.%y","%B.%d.%Y",
            "%y.%m.%d","%Y.%m.%d","%m.%d.%y","%m.%d.%Y","%d.%m.%y","%d.%m.%Y",
        ]
        
        for column in non_numerical_columns:
            for date_format in date_formats:
                try:
                    converted_column = pd.to_datetime(data[column], format=date_format)
                    if converted_column.notna().any() and converted_column.dt.date.nunique != 1:
                        date_columns.append(column)
                        data[column] = pd.to_datetime(data[column], format=date_format)
                        break
                except ValueError:
                    continue
                                        
        # go through non_numerical_columns which are not in date_columns and check for time formats
        
        time_columns = []
        for column in non_numerical_columns:
            if column not in date_columns:
                try:
                    pd.to_datetime(data[column], format="%H:%M:%S") #, errors='coerce'
                    if pd.to_datetime(data[column], format="%H:%M:%S").notna().sum() != 0:
                        time_columns.append(column)
                        data[column] = pd.to_datetime(data[column], format="%H:%M:%S", errors="coerce")
                except Exception:
                    try:
                        pd.to_datetime(data[column], format="%H:%M") #, errors='coerce'
                        if pd.to_datetime(data[column], format="%H:%M").notna().sum() != 0:
                            time_columns.append(column)
                            data[column] = pd.to_datetime(data[column], format="%H:%M", errors="coerce")
                    except Exception:
                        pass
                            
        ###########################################################################################
        
        
        # add code to try convert strings to numbers

        # Identify string/object columns
        all_string_columns = list(set(non_numerical_columns) - set(date_columns))
        all_string_columns = list(set(all_string_columns) - set(time_columns))
        categorical_string_columns = []
        for column in data[all_string_columns].columns:
            if (data[all_string_columns][column].nunique() < len(data[all_string_columns]) * 0.2) and ((data[all_string_columns][column].value_counts() >= 2).sum() >= (0.2 * len(data[all_string_columns][column].value_counts()))):
                if data[all_string_columns][column].nunique() != len(data[all_string_columns][column]):
                    categorical_string_columns.append(column)
        non_categorical_string_columns = list(set(all_string_columns) - set(categorical_string_columns))
        
        # Calculate average lengths of non-categorical strings
        average_lengths_df = calculate_average_length(data, non_categorical_string_columns)
        # Encode categorical strings
        orig_data = data.copy()
        le = LabelEncoder()
        for column in categorical_string_columns:
            data[column] = data[column].astype(str)
            data[column] = le.fit_transform(data[column])

        
        
        ###########################################################################################
        for column in date_columns:
            data[column + '_year'] = data[column].dt.year
            if data[column + '_year'].notna().any():
                orig_data_completeness[column + '_year'] = data[column + '_year']
                data[column + '_year'] = data[column + '_year'].fillna(round(data[column + '_year'].mean()))
                data[column + '_year'] = data[column + '_year'].astype('Int64')

            data[column + '_month'] = data[column].dt.month
            if data[column + '_month'].notna().any():
                orig_data_completeness[column + '_month'] = data[column + '_month']
                data[column + '_month'] = data[column + '_month'].fillna(round(data[column + '_month'].mean()))
                data[column + '_month'] = data[column + '_month'].astype('Int64')

            data[column + '_day'] = data[column].dt.day
            if data[column + '_day'].notna().any():
                orig_data_completeness[column + '_day'] = data[column + '_day']
                data[column + '_day'] = data[column + '_day'].fillna(round(data[column + '_day'].mean()))
                data[column + '_day'] = data[column + '_day'].astype('Int64')

            data.insert(data.columns.get_loc(column) + 1, column + '_year', data.pop(column + '_year'))
            data.insert(data.columns.get_loc(column) + 2, column + '_month', data.pop(column + '_month'))
            data.insert(data.columns.get_loc(column) + 3, column + '_day', data.pop(column + '_day'))

            data = data.drop(columns=[column], axis=1)
        ###########################################################################################


        ###########################################################################################
        for column in time_columns:
            data[column + '_hour'] = data[column].dt.hour
            if data[column + '_hour'].notna().any():
                orig_data_completeness[column + '_hour'] = data[column + '_hour']
                data[column + '_hour'] = data[column + '_hour'].fillna(round(data[column + '_hour'].mean()))
                data[column + '_hour'] = data[column + '_hour'].astype('Int64')

            data[column + '_minute'] = data[column].dt.minute
            if data[column + '_minute'].notna().any():
                orig_data_completeness[column + '_minute'] = data[column + '_minute']
                data[column + '_minute'] = data[column + '_minute'].fillna(round(data[column + '_minute'].mean()))
                data[column + '_minute'] = data[column + '_minute'].astype('Int64')

            data[column + '_second'] = data[column].dt.second
            if data[column + '_second'].notna().any():
                orig_data_completeness[column + '_second'] = data[column + '_second']
                data[column + '_second'] = data[column + '_second'].fillna(round(data[column + '_second'].mean()))
                data[column + '_second'] = data[column + '_second'].astype('Int64')

            data.insert(data.columns.get_loc(column) + 1, column + '_hour', data.pop(column + '_hour'))
            data.insert(data.columns.get_loc(column) + 2, column + '_minute', data.pop(column + '_minute'))
            data.insert(data.columns.get_loc(column) + 3, column + '_second', data.pop(column + '_second'))

            data = data.drop(columns=[column], axis=1)
        ###########################################################################################
            
            
            
            
            
            
            
        #data = data.drop(date_columns, axis=1)
        
        # Create metadata for each column (DOESNT HANDLE DATE COMPLETENESS)
        for column in data.columns:
            #if column not in orig_data_completeness.columns:
            #    orig_data_completeness[column] = data[column]
                    
            completeness = (orig_data_completeness[column].notna().sum() / len(data)) * 100
            
            if column in non_categorical_string_columns: #or column in non_numerical_columns
                value_range = None
                mean = next((item['avg_char_length'] for item in average_lengths_df if item['column'] == column), None)
                std_dev = next((item['avg_space_length'] for item in average_lengths_df if item['column'] == column), None)
                skewness_value = None
                #datatype = 'object'
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
            orig_data[column] = orig_data[column].astype(str)
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

            processed_data.columns = [identifier_column if identifier_column in col else col for col in processed_data.columns] # remove prefix for ID
            try:
                combined_data = pd.merge(combined_data, processed_data, on=identifier_column, how='outer')
            except:
                combined_data = pd.concat([combined_data, processed_data], axis=1)
                
        # Correlation across combined numerical data
        combined_numerical_data = combined_data.select_dtypes(include=['number'])
        #correlation_matrix = combined_numerical_data.corr()

        combined_numerical_data = combined_numerical_data.dropna(axis=1)
        combined_numerical_data = combined_numerical_data.loc[:, combined_numerical_data.nunique() > 1]
        
        if type == "correlated":
            correlation_matrix = np.corrcoef(combined_numerical_data.astype(float).values, rowvar=False)
            correlation_matrix = pd.DataFrame(correlation_matrix, index=combined_numerical_data.columns, columns=combined_numerical_data.columns)
            
            best_fit_distributions = identify_best_fit_distributions(combined_numerical_data)
            marginals = []
            for column in combined_numerical_data.columns:
                dist, params = best_fit_distributions[column]
                if dist and params:
                    marginals.append(dist(*params))
                else:
                    marginals.append(norm(loc=np.mean(combined_numerical_data[column]), scale=np.std(combined_numerical_data[column])))

        if type == "correlated":
            return combined_metadata, combined_label_mapping, correlation_matrix, marginals
        elif type == "structural":
            return combined_metadata[['variable_name', 'datatype', 'completeness', 'values', 'table_name']], combined_label_mapping
        else:
            return combined_metadata, combined_label_mapping
    else:
        # If input is a single DataFrame, process directly
        metadata, label_mapping, processed_data = process_single_dataframe(data)

        # FIX
        numerical_data = processed_data.select_dtypes(include=['number'])
        #numerical_data_filtered = numerical_data.dropna(axis=1, how='all')  # Drop columns with all NaN values
        #numerical_data_filtered = numerical_data_filtered.loc[:, ~(numerical_data_filtered == 0).all()]  # Drop columns with all zero values
        #columns_to_drop = [col for col in numerical_data_filtered.columns if numerical_data_filtered[col].replace(0, np.nan).isna().all()]
        #numerical_data_filtered = numerical_data_filtered.drop(columns=columns_to_drop)
        #correlation_matrix = numerical_data_filtered.corr()
        #correlation_matrix = numerical_data.corr()
        #correlation_matrix = np.corrcoef(numerical_data.astype(float).values, rowvar=False)

        #print(numerical_data)
        #needed so it matches the removal of empty and single values in generation method
        numerical_data = numerical_data.dropna(axis=1)
        numerical_data = numerical_data.loc[:, numerical_data.nunique() > 1]
        #numerical_data = numerical_data.loc[:, numerical_data.mean() != 0]
        

        if type == "correlated":
            correlation_matrix = np.corrcoef(numerical_data.astype(float).values, rowvar=False)
            correlation_matrix = pd.DataFrame(correlation_matrix, index=numerical_data.columns, columns=numerical_data.columns)
            
            best_fit_distributions = identify_best_fit_distributions(numerical_data)
            marginals = []
            for column in numerical_data.columns:
                dist, params = best_fit_distributions[column]
                if dist and params:
                    marginals.append(dist(*params))
                else:
                    marginals.append(norm(loc=np.mean(numerical_data[column]), scale=np.std(numerical_data[column])))
            #correlation_matrix = processed_data.corr() if type == "correlated" else None

        # if statistical or structural then only return metadata with columns needed (metadata[columns])

        if type == "correlated":

            return metadata, label_mapping, correlation_matrix, marginals
        elif type == "structural":
            return metadata[['variable_name', 'datatype', 'completeness', 'values', 'table_name']], label_mapping
        else:
            return metadata, label_mapping



# Function to generate random data based on metadata for each filename
def generate_structural_data(metadata, label_mapping=None, num_records=100, identifier_column=None):
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

        # Check if value_range is valid
        if pd.isna(value_range) or value_range == "None":
            if 'object' in str(dtype):
                return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 10)))
            elif 'int' in str(dtype).lower():
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
                        if 'int' in str(dtype).lower():
                            # Special case for binary values
                            if value_range == (0, 1):
                                return random.choice([0, 1])  # For binary values, return either 0 or 1
                            return random.randint(value_range[0], value_range[1])
                        elif 'float' in str(dtype):
                            return round(random.uniform(value_range[0], value_range[1]), 2)
                except Exception as e:
                    #print(f"Error parsing values: {e}")
                    #print(row['variable_name'])
                    #print(row['datatype'])
                    #print(row['values'])
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
            #completeness = row['completeness']
            #if completeness < 100.0:
            #    num_missing = int(num_records * (1 - (completeness / 100.0)))
            #    missing_indices = random.sample(range(num_records), num_missing)
            #    for idx in missing_indices:
            #        data[idx] = None
            
            generated_data[table_name][column_name] = data

        # Create DataFrame for the current table
        generated_data[table_name] = pd.DataFrame(generated_data[table_name])

        # Handle date combination and avoid duplications
        date_columns = {}
        time_columns = {}
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

            elif col.endswith('_hour'):
                base_name = col[:-5]
                if base_name not in time_columns:
                    time_columns[base_name] = {}
                time_columns[base_name]['hour'] = col
            elif col.endswith('_minute'):
                base_name = col[:-7]
                if base_name not in time_columns:
                    time_columns[base_name] = {}
                time_columns[base_name]['minute'] = col
            elif col.endswith('_second'):
                base_name = col[:-7]
                if base_name not in time_columns:
                    time_columns[base_name] = {}
                time_columns[base_name]['second'] = col



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


        time_base_names = []
        combined_time_cols = {}
        for time_base_name, components in time_columns.items():
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
        for col in original_order:
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
        
        
        
        # NEW COMPLETENESS HANDLING
        
        for _, row in metadata.iterrows():
            column = row['variable_name']
            completeness = row['completeness'] / 100.0
            
            if column.endswith("_year"):
                column = column[:-5]
            
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
    







def best_fit_distribution(data, bins=200):
    DISTRIBUTIONS = [
        stats.norm, stats.expon, stats.lognorm, stats.gamma,
        stats.beta, stats.uniform, stats.weibull_min, stats.poisson
    ]

    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    best_distribution = None
    best_params = None
    best_sse = np.inf

    for distribution in DISTRIBUTIONS:
        try:
            params = distribution.fit(data)
            pdf = distribution.pdf(bin_centers, *params)
            sse = np.sum(np.power(hist - pdf, 2.0))

            if best_sse > sse > 0:
                best_distribution = distribution
                best_params = params
                best_sse = sse
        except Exception:
            pass

    return best_distribution, best_params

def identify_best_fit_distributions(df, discrete_threshold=10): # change this discrete identification
    #df = df.fillna(df.mean())
    
    result = {}

    for column in df.columns:
        data = df[column].dropna()
        data = data[~data.isin([np.inf, -np.inf])]

        if data.nunique() <= discrete_threshold:
            try:
                mu = data.mean()
                result[column] = (stats.poisson, (mu,))
            except Exception:
                result[column] = (None, None)
        else:
            best_distribution, best_params = best_fit_distribution(data)
            result[column] = (best_distribution, best_params)

    return result

def generate_copula_samples(corr_matrix, marginals, n_samples, variable_names, lower_bounds, upper_bounds):
    gaussian_copula = GaussianCopula(corr=corr_matrix, allow_singular=True)
    copula_dist = CopulaDistribution(gaussian_copula, marginals)
    generated_samples = copula_dist.rvs(nobs=n_samples)

    # Clip the samples to the original data bounds
    #bounds = data.agg([np.min, np.max])  # Correctly get min and max for each feature
    #for i, column in enumerate(data.columns):
    #    min_val = bounds.loc['amin', column] if 'amin' in bounds.index else bounds.loc['min', column]
    #    max_val = bounds.loc['amax', column] if 'amax' in bounds.index else bounds.loc['max', column]
    #    generated_samples[:, i] = np.clip(generated_samples[:, i], min_val, max_val)

    return generated_samples
    








def generate_correlated_data(metadata, correlation_matrix, marginals, num_records=100, identifier_column=None, label_mapping={}):

    metadata = metadata.copy()
    
    metadata['variable_name'] = metadata.apply(lambda x: f"{x['table_name']}.{x['variable_name']}", axis=1)

    # Number of samples to generate
    num_rows = num_records

    def is_int_or_float(datatype):
        return pd.api.types.is_integer_dtype(datatype) or pd.api.types.is_float_dtype(datatype)

    empty_metadata = metadata[metadata["completeness"]==0]
    zero_metadata = metadata[metadata["mean"]==0]
    single_value_metadata = metadata[(metadata["standard_deviation"] == 0) | (pd.isna(metadata["standard_deviation"]))]

    numerical_metadata = metadata[metadata['datatype'].apply(is_int_or_float)]
    non_numerical_metadata = metadata[~metadata['datatype'].apply(is_int_or_float)]

    #orig_numerical_columns = numerical_metadata['variable_name'].tolist()

    numerical_metadata = numerical_metadata[~numerical_metadata['variable_name'].isin(empty_metadata['variable_name'])]
    numerical_metadata = numerical_metadata[~numerical_metadata['variable_name'].isin(zero_metadata['variable_name'])]
    numerical_metadata = numerical_metadata[~numerical_metadata['variable_name'].isin(single_value_metadata['variable_name'])]

    #orig_numerical_columns = numerical_metadata['variable_name'].tolist()
    
    # this should work to remove both nan and zero variables
    #correlation_matrix = pd.DataFrame(correlation_matrix, columns=orig_numerical_columns)    ###### NEW (only columns= bit)
    orig_numerical_columns = correlation_matrix.columns
    
    # SHOULD MAKE SURE IT DOES BOTH ANT NOT JUST ONE!! (REMOVED FOR NOW BUT MAY BREAK)
    #correlation_matrix = correlation_matrix.dropna(axis=1, how='all') # extract these variable names and handle them by sampling from a distribution
    #correlation_matrix = correlation_matrix.dropna(axis=0, how='all')
    correlation_matrix = correlation_matrix.fillna(0)

    remaining_columns = correlation_matrix.columns.tolist()   ###### NEW
    dropped_columns = list(set(orig_numerical_columns) - set(remaining_columns))   ###### NEW
    dropped_metadata = metadata[metadata['variable_name'].isin(dropped_columns)]   ###### NEW

    #correlation_matrix = correlation_matrix.fillna(0)
    correlation_matrix = correlation_matrix.to_numpy()

    numerical_metadata = numerical_metadata[~numerical_metadata['variable_name'].isin(dropped_metadata['variable_name'])]   ###### NEW

    #correlation_matrix = correlation_matrix.loc[numerical_metadata['variable_name'], numerical_metadata['variable_name']]

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

    #lower = np.array(lower_bounds)
    #upper = np.array(upper_bounds)

    synthetic_samples = generate_copula_samples(correlation_matrix, marginals, num_records, variable_names, lower_bounds, upper_bounds)

    # Convert samples into a Pandas DataFrame
    synthetic_data = pd.DataFrame(synthetic_samples, columns=variable_names)

    # REMOVED FOR NOW BUT NEED TO ADD IN LATER ON AFTER THE INT CONVERSION TO DO IT FOR ALL COLUMNS
    # Introduce missing values (NaNs) according to the completeness percentages (ONLY DOES IT FOR NUMERICAL!!! CHANGE!)
    #for i, (index, row) in enumerate(numerical_metadata.iterrows()):
    #    completeness = row['completeness'] / 100  # Convert to a decimal
    #    num_valid_rows = int(num_rows * completeness)  # Number of valid rows based on completeness

    #    # Randomly set some of the data to NaN based on completeness
    #    if completeness < 1.0:
    #        nan_indices = np.random.choice(num_rows, size=(num_rows - num_valid_rows), replace=False)
    #        synthetic_data.iloc[nan_indices, i] = np.nan

    for index, row in zero_metadata.iterrows():
        column_name = row['variable_name']
        synthetic_data[column_name] = 0
    for index, row in empty_metadata.iterrows():
        column_name = row['variable_name']
        synthetic_data[column_name] = None
    for index, row in single_value_metadata.iterrows():
        column_name = row['variable_name']
        synthetic_data[column_name] = single_value_metadata[single_value_metadata['variable_name']==row['variable_name']]['mean'].values[0]
    for index, row in dropped_metadata.iterrows():   ### NEW
        if (row['variable_name'] not in single_value_metadata['variable_name'].values) \
        and (row['variable_name'] not in zero_metadata['variable_name'].values) \
        and (row['variable_name'] not in empty_metadata['variable_name'].values):
            column_name = row['variable_name']
            mean = row['mean']
            standard_deviation = row['standard_deviation']
            min_value, max_value = row['values']
            a, b = (min_value - mean) / standard_deviation, (max_value - mean) / standard_deviation
            synthetic_data[column_name] = truncnorm.rvs(a, b, loc=mean, scale=standard_deviation, size=num_records)


    for column in synthetic_data.columns:
        # Find the corresponding datatype in the metadata
        datatype = metadata.loc[metadata['variable_name'] == column, 'datatype'].values
        if len(datatype) > 0 and "int" in str(datatype[0]).lower():   #if len(datatype) > 0 and np.issubdtype(datatype[0], np.integer):
            # Round the values in the column if the datatype is an integer
            synthetic_data[column] = round(synthetic_data[column]).astype('Int64')


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
        



    #Introduce missing values (NaNs) according to the completeness percentages
    for i, (index, row) in enumerate(metadata.iterrows()):
        completeness = row['completeness'] / 100  # Convert to a decimal
        num_valid_rows = int(num_rows * completeness)  # Number of valid rows based on completeness

        # Randomly set some of the data to NaN based on completeness
        if completeness < 1.0:
            nan_indices = np.random.choice(num_rows, size=(num_rows - num_valid_rows), replace=False)
            synthetic_data.iloc[nan_indices, i] = np.nan




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