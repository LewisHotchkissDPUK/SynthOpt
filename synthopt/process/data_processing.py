import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import math
from distfit import distfit
from scipy import stats

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
        # Catch any unforeseen errors in the outer block
        print(f"An unexpected error occurred: {e}")
        return data, []  # Return data and an empty list as a fallback



def detect_datetime_in_objects(data, datetime_formats, non_numerical_columns):
    try:
        datetime_columns = []
        column_date_format = {}

        if datetime_formats is None:
            # Define default datetime formats if none are provided
            date_formats = [
                "%b-%m", "%b-%y", "%b-%Y",
                "%B-%m", "%B-%y", "%B-%Y",
                "%d-%b-%y", "%d-%b-%Y", "%b-%d-%y", "%b-%d-%Y",
                "%d-%B-%y", "%d-%B-%Y", "%B-%d-%y", "%B-%d-%Y",
                "%y-%m-%d", "%Y-%m-%d", "%m-%d-%y", "%m-%d-%Y", "%d-%m-%y", "%d-%m-%Y",
                "%b/%m", "%b/%y", "%b/%Y",
                "%B/%m", "%B/%y", "%B/%Y",
                "%d/%b/%y", "%d/%b/%Y", "%b/%d/%y", "%b/%d/%Y",
                "%d/%B/%y", "%d/%B/%Y", "%B/%d/%y", "%B/%d/%Y",
                "%y/%m/%d", "%Y/%m/%d", "%m/%d/%y", "%m/%d/%Y", "%d/%m/%y", "%d/%m/%Y",
                "%b.%m", "%b.%y", "%b.%Y",
                "%B.%m", "%B.%y", "%B.%Y",
                "%d.%b.%y", "%d.%b.%Y", "%b.%d.%y", "%b.%d.%Y",
                "%d.%B.%y", "%d.%B.%Y", "%B.%d.%y", "%B.%d.%Y",
                "%y.%m.%d", "%Y.%m.%d", "%m.%d.%y", "%m.%d.%Y", "%d.%m.%y", "%d.%m.%Y",
            ]
            time_formats = ["%H:%M:%S.%f", "%H:%M:%S", "%H:%M"]
            datetime_formats = date_formats + time_formats + [f"{date} {time}" for date in date_formats for time in time_formats]

        for column in tqdm(data[non_numerical_columns], desc="Processing Datetime Columns"):
            try:
                if "date" in str(data[column].dtype):
                    # Convert to string for uniform processing
                    data[column] = data[column].astype("string")

                for datetime_format in datetime_formats:
                    # Attempt conversion using the specified datetime format
                    converted_column = pd.to_datetime(data[column], format=datetime_format, errors='raise')
                    
                    if converted_column.notna().any():
                        if any(converted_column.dt.date == pd.Timestamp("1900-01-01").date()):
                            # Handle timedelta conversion
                            data[column] = pd.to_timedelta(data[column]).dt.total_seconds()
                        else:
                            # Handle datetime conversion to Unix time
                            data[column] = pd.to_datetime(data[column], format=datetime_format).astype("int64") // 10**9

                        column_date_format[column] = datetime_format
                        datetime_columns.append(column)
                        non_numerical_columns.remove(column)
                        break  # Stop checking other formats once successfully converted
            
            except Exception as e:
                # Handle errors related to individual columns
                print(f"Error processing column '{column}': {e}")
                continue

        # Replace problematic values (-9223372037) with NaN
        data = data.replace(-9223372037, np.nan)

    except Exception as e:
        # Catch-all for errors in the outer block
        print(f"An unexpected error occurred during processing: {e}")
        return data, datetime_columns, non_numerical_columns, None

    return data, datetime_columns, non_numerical_columns, column_date_format



def detect_integer_in_floats(data):
    try:
        # Process columns of float type
        for column in tqdm(data.select_dtypes(include="float"), desc="Processing Integer Columns"):
            # Check if all non-NA values are integers
            if (data[column].dropna() % 1 == 0).all() or data[column].apply(float.is_integer).all():
                # Convert the column to integer type
                data[column] = data[column].astype("Int64")

    except Exception as e:
        # Catch-all for outer-level errors
        print(f"An unexpected error occurred: {e}")

    return data



# identify string categories
def detect_categorical_strings(data, non_numerical_columns):
    try:
        categorical_string_columns = []

        for column in tqdm(data[non_numerical_columns].columns, desc="Processing String Columns"):
            # Check if column satisfies the categorical string conditions
            unique_count = data[non_numerical_columns][column].nunique()
            total_count = len(data[non_numerical_columns])
            value_counts = data[non_numerical_columns][column].value_counts()

            if (
                unique_count < total_count * 0.2  # Unique values are less than 20% of total rows
            ) and (
                (value_counts >= 2).sum() >= (0.6 * len(value_counts))  # At least 60% of values appear 2+ times
            ):
                if unique_count != total_count:  # Column is not fully unique
                    categorical_string_columns.append(column)

        # Identify non-categorical string columns
        non_categorical_string_columns = list(
            set(non_numerical_columns) - set(categorical_string_columns)
        )

    except Exception as e:
        # Catch-all for any unforeseen errors in the outer block
        print(f"An unexpected error occurred: {e}")
        return data, [], non_numerical_columns  # Return original structure with empty results as fallback

    return data, categorical_string_columns, non_categorical_string_columns



def encode_data(data, orig_data, categorical_string_columns):
    try:
        data_encoded = data.copy()
        le = LabelEncoder()
        column_mappings = {}

        for column in tqdm(categorical_string_columns, desc="Encoding Categorical String Columns"):
            try:
                # Convert column to string for consistent encoding
                data_encoded[column] = data_encoded[column].astype(str)

                # Perform label encoding
                data_encoded[column] = le.fit_transform(data_encoded[column])

                # Create a mapping of encoded values to original values
                mapping = dict(zip(le.transform(data_encoded[column].unique()), orig_data[column].unique()))

                try:
                    # Identify and handle NaN values in the mapping
                    nan_key = next(
                        (key for key, value in mapping.items() if isinstance(value, float) and math.isnan(value)), 
                        None
                    )
                    if nan_key is not None:
                        data_encoded[column] = data_encoded[column].replace(nan_key, np.nan)
                        del mapping[nan_key]  # Remove NaN from the mapping
                except Exception as e:
                    print(f"Error handling NaN for column '{column}': {e}")

                # Convert the column to integer type with support for missing values
                data_encoded[column] = data_encoded[column].astype("Int64")

                # Store the mapping for the column
                column_mappings[column] = mapping

            except Exception as e:
                # Log any errors encountered during encoding of the column
                print(f"Error encoding column '{column}': {e}")
                continue  # Proceed with the next column

        return data_encoded, column_mappings

    except Exception as e:
        # Catch-all for unexpected errors in the outer block
        print(f"An unexpected error occurred during encoding: {e}")
        return data, {}  # Return original data and an empty mappings dictionary as fallback



# identify numerical categories
def detect_categorical_numerical(data, numerical_columns):
    try:
        categorical_numerical_columns = []

        for column in tqdm(numerical_columns, desc="Identifying Categorical Numerical Columns"):
            # Check if the column meets all conditions to be considered categorical
            if (
                (data[column].nunique() < data[column].notna().sum() * 0.2)  # Unique values < 20% of non-NA values
                and ((data[column].value_counts() >= 2).sum() >= (0.7 * data[column].nunique()))  # >= 70% have 2+ counts
                and (data[column].notna().any())  # At least one non-NA value
                and (data[column].nunique() >= 2)  # At least 2 unique values
                and (data[column].nunique() != len(data[column]))  # Not fully unique
                and (data[column].nunique() < 50)  # Fewer than 50 unique values
            ):
                categorical_numerical_columns.append(column)

        return data, categorical_numerical_columns

    except Exception as e:
        # Catch-all for any unforeseen errors in the outer block
        print(f"An unexpected error occurred: {e}")
        return data, []  # Return the original data and an empty list as fallback




################################# STATISTICS PROCESSING #################################



# best fit identification
def best_fit(data):
    try:
        distribution_metadata = {}

        # Loop through each column in the dataset
        for col in tqdm(data.columns, desc="Identifying Best Fit Distributions"):
            # Clean the data by removing NaNs and Infs
            column_data = data[col].replace([np.inf, -np.inf], np.nan).dropna()

            # Skip the column if it's empty
            if column_data.empty:
                continue

            # Fit the distribution on the column data
            dfit = distfit(verbose=0)
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

    except Exception as e:
        # Catch-all for unforeseen errors
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame as fallback

