import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import math
from distfit import distfit
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

# Detect if an object column can be turned into a numerical column
def detect_numerical_in_objects(data, non_numerical_columns):
    try:
        if not non_numerical_columns:  # Check if non_numerical_columns is empty
            return data, non_numerical_columns
        
        for column in tqdm(non_numerical_columns, desc="Processing Non-Numerical Columns"):
            # Attempt to convert the column to numeric
            data[column] = pd.to_numeric(data[column], errors="raise")
            
            # Remove the column from the list if successfully converted
            non_numerical_columns.remove(column)
        
        return data, non_numerical_columns  # Return the updated data and columns list

    except Exception as e:
        return data, non_numerical_columns  # Return data and an empty list as a fallback


def detect_datetime_in_objects(data, datetime_formats, non_numerical_columns):
    datetime_columns = []
    column_date_format = {}

    if datetime_formats is None:
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
        time_formats = ["%H:%M:%S.%f", "%H:%M:%S", "%H:%M", "%H:%M.%f"]
        datetime_formats = date_formats + time_formats + [f"{date} {time}" for date in date_formats for time in time_formats]

    for column in tqdm(data[non_numerical_columns], desc="Processing Datetime Columns"):
        if "date" in str(data[column].dtype):
            data[column] = data[column].astype("string")
            # data[column] = data[column].view(int) // 10**9
            # datetime_columns.append(column)
            # non_numerical_columns.remove(column)

        for datetime_format in datetime_formats:
            try:
                converted_column = pd.to_datetime(data[column], format=datetime_format)
                # should only convert to unix if for stats/corr version
                if converted_column.notna().any():
                    if any(converted_column.dt.date == pd.Timestamp("1900-01-01").date()):
                        data[column] = pd.to_timedelta(data[column])
                        data[column] = data[column].dt.total_seconds()
                    else:
                        data[column] = pd.to_datetime(data[column], format=datetime_format)
                        data[column] = data[column].astype("int64") // 10**9

                    column_date_format[column] = datetime_format
                    datetime_columns.append(column)
                    non_numerical_columns.remove(column)
            except:
                None

    data = data.replace(-9223372037, np.nan)
    return data, datetime_columns, non_numerical_columns, column_date_format


def detect_integer_in_floats(data):
    try:
        for column in tqdm(data.select_dtypes(include="float"), desc="Processing Integer Columns"):
            if (data[column].dropna() % 1 == 0).all():
                data[column] = data[column].astype("Int64")
    except:
        None
    return data


# identify string categories
def detect_categorical_strings(data, non_numerical_columns):
    categorical_string_columns = []
    for column in tqdm(data[non_numerical_columns].columns, desc="Processing String Columns"):

        if (
            data[non_numerical_columns][column].nunique()
            < len(data[non_numerical_columns]) * 0.2
        ) and (
            (data[non_numerical_columns][column].value_counts() >= 2).sum()
            >= (0.6 * len(data[non_numerical_columns][column].value_counts()))
        ):

            if data[non_numerical_columns][column].nunique() != len(
                data[non_numerical_columns][column]
            ):
                categorical_string_columns.append(column)

    non_categorical_string_columns = list(
        set(non_numerical_columns) - set(categorical_string_columns)
    )

    return data, categorical_string_columns, non_categorical_string_columns


def encode_data(data, orig_data, categorical_string_columns):
    data_encoded = data.copy()
    le = LabelEncoder()
    column_mappings = {}
    for column in tqdm(categorical_string_columns, desc="Encoding Categorical String Columns"):
        data_encoded[column] = data_encoded[column].astype(str)
        data_encoded[column] = le.fit_transform(data_encoded[column])

        mapping = dict(zip(le.fit_transform(data_encoded[column].unique()), orig_data[column].unique()))
        try:
            nan_key = next((key for key, value in mapping.items() if isinstance(value, float) and math.isnan(value)), None)
            data_encoded[column] = data_encoded[column].replace(nan_key, np.nan)
        except:
            None
        data_encoded[column] = data_encoded[column].astype("Int64")
        if nan_key is not None:
            del mapping[nan_key]
        column_mappings[column] = mapping
    return data_encoded, column_mappings


# identify numerical categories
def detect_categorical_numerical(data, numerical_columns):
    categorical_numerical_columns = []
    for column in tqdm(numerical_columns, desc="Identifying Categorical Numerical Columns"):
        if ((data[column].nunique() < data[column].notna().sum() * 0.2) 
            and ((data[column].value_counts() >= 2).sum()>= (0.7 * data[column].nunique()))
            and (data[column].notna().any())
            and (data[column].nunique() >= 2)
            and (data[column].nunique() != len(data[column]))
            and (data[column].nunique() < 50)):
            categorical_numerical_columns.append(column)
    return data, categorical_numerical_columns



################################# STATISTICS PROCESSING #################################



# Helper function to process a single column
def fit_distribution(col, column_data):
    column_data = column_data.replace([np.inf, -np.inf], np.nan).dropna()
    if column_data.empty:
        return col, None

    dfit = distfit(verbose=0)
    dfit.fit_transform(column_data)
    best_fit = dfit.model
    return col, {
        'dist': best_fit['name'],
        'params': best_fit['params'],
    }

# Parallelized best_fit function
def best_fit(data):
    distribution_metadata = {}

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(fit_distribution, col, data[col]): col
            for col in data.columns
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Identifying Best Fit Distributions"):
            col, result = future.result()
            if result:
                distribution_metadata[col] = result

    distribution_metadata_df = pd.DataFrame.from_dict(distribution_metadata, orient='index')
    return distribution_metadata_df



"""
# Original best_fit function without parallelization
# best fit identification
def best_fit(data):
    distribution_metadata = {}

    # Loop through each column in the dataset
    for col in tqdm(data.columns, desc="Identifying Best Fit Distributions"):
        # Clean the data by removing NaNs and Infs
        column_data = data[col].replace([np.inf, -np.inf], np.nan).dropna()

        # Skip the column if it's empty
        if column_data.empty:
            continue

        dfit = distfit(verbose=0)
        # Fit the distribution on the column data
        dfit.fit_transform(column_data)

        # Extract the best distribution and its parameters
        best_fit = dfit.model
        distribution_metadata[col] = {
            'dist': best_fit['name'],         # Best-fitting distribution name
            'params': best_fit['params'],     # Best-fitting parameters
        }

    # Convert the distribution metadata into a DataFrame for easy inspection
    distribution_metadata_df = pd.DataFrame.from_dict(distribution_metadata, orient='index')

    return distribution_metadata_df
"""