import pandas as pd
import numpy as np
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import load, load_from_file, save, save_to_file
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata
import random
from functools import reduce

def add_noise(data, scale, discrete_cols): # need to add constraints for integers (think ive done this)
    noised_data = data.copy()
    for column in data.columns:
        if not data[column].dropna().isin([0,1]).all():
            noise = np.random.laplace(loc=0, scale=scale, size=len(data))
            if column in discrete_cols:
                noised_data[column] = np.round(np.clip(noised_data[column] + noise, data[column].min(), data[column].max()))
            else:
                noised_data[column] = np.clip(noised_data[column] + noise, data[column].min(), data[column].max())
    return noised_data  

def create_metadata(data):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    return metadata

# imputation, categorical/string handling, outlier removal etc
def process(data, table_type='single'): #, subset_size=None
    if table_type == 'multi':
        imputer = KNNImputer(n_neighbors=3)
        processed_dataframes = []
        control_dataframes = []
        for df in data:
            imputed_data = imputer.fit_transform(df)
            processed_df = pd.DataFrame(imputed_data, columns=df.columns)
            processed_df, control_df = train_test_split(processed_df, test_size=0.1, random_state=42)
            #if subset_size != None:
            #    subset_size = subset_size * 0.9
            #    processed_df = processed_df.sample(n=subset_size)
            processed_dataframes.append(processed_df)
            control_dataframes.append(control_df)

        return processed_dataframes, control_dataframes

    if table_type == 'single':
        imputer = KNNImputer(n_neighbors=3)
        data_processed = imputer.fit_transform(data)
        data_processed = pd.DataFrame(data_processed, columns=data.columns)
        #if subset_size != None:
        #    subset_size = subset_size * 0.9
        #    data_processed = data_processed.sample(n=subset_size)
        data_processed, control_data = train_test_split(data_processed, test_size=0.1, random_state=42)

        return data_processed, control_data
    
    else:
        print("Please select an appropriate table type")
        return None

    

def generate_syntheticdata(data, identifier_column, prediction_column, sensitive_columns, sample_size, table_type='single', model_name='pategan', iterations=100, dp_epsilon=1, dp_delta=None, dp_lambda=0.001, save_location=None):
    try:
        # Check if data is a pandas DataFrame (for single table) or a list of DataFrames (for multi table)
        if table_type == 'single' and not isinstance(data, pd.DataFrame):
            raise ValueError("For single table type, data must be a pandas DataFrame.")
        elif table_type == 'multi' and not isinstance(data, list):
            raise ValueError("For multi table type, data must be a list of pandas DataFrames.")

        # Multi-table handling
        if table_type == 'multi':
            column_dict = {}
            for i, df in enumerate(data):
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(f"Element {i+1} in the data list is not a pandas DataFrame.")
                column_dict[f"DataFrame_{i+1}"] = df.columns.tolist()

            try:
                data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data)
            except KeyError as e:
                raise KeyError(f"Identifier column '{identifier_column}' not found in one or more DataFrames.") from e

        # Drop identifier column from the data
        if identifier_column not in data.columns:
            raise KeyError(f"Identifier column '{identifier_column}' not found in the data.")
        data = data.drop(columns=[identifier_column])

        # Check for object or string columns that are not allowed
        object_or_string_cols = data.select_dtypes(include=['object', 'string'])
        if not object_or_string_cols.empty:
            raise TypeError(f"Data must not contain string or object data types. Columns with object or string types: {list(object_or_string_cols.columns)}")

        # Create metadata and identify discrete columns
        metadata = create_metadata(data)
        available_columns = data.columns.tolist()
        discrete_columns = []
        for col, meta in metadata.columns.items():
            if ('sdtype' in meta and meta['sdtype'] == 'categorical') or (data[col].fillna(9999) % 1 == 0).all():
                discrete_columns.append(col)

        data_columns = data.columns

        # Validate the sample size
        if sample_size is None:
            sample_size = len(data)
        elif sample_size > len(data):
            raise ValueError("Sample size cannot be larger than the number of rows in the dataset.")

        # Check if the model name is valid and create the appropriate synthesizer
        try:
            if model_name == "ctgan":
                synthesizer = Plugins().get(model_name, n_iter=iterations)
            elif model_name == "dpgan":
                synthesizer = Plugins().get(model_name, n_iter=iterations, epsilon=dp_epsilon, delta=dp_delta)
            elif model_name == "pategan":
                synthesizer = Plugins().get(model_name, n_iter=iterations, epsilon=dp_epsilon, delta=dp_delta, lamda=dp_lambda)
            else:
                raise ValueError(f"Not a valid model name: '{model_name}'")
        except Exception as e:
            raise ValueError(f"Failed to initialize the synthesizer model '{model_name}'. Please check the model name and parameters.") from e

        # Ensure integer columns stay integers
        for column in data_columns:
            if (data[column] % 1).all() == 0:
                data[column] = data[column].astype(int)

        # Apply noise to data if using ctgan model
        if model_name == "ctgan":
            try:
                data = add_noise(data, dp_epsilon, discrete_columns)  # Noise added for ctgan only
            except Exception as e:
                raise ValueError("Error occurred while adding noise to the data.") from e

        # Convert data to GenericDataLoader
        try:
            data = GenericDataLoader(data, target_column=prediction_column, sensitive_columns=sensitive_columns)
        except Exception as e:
            raise ValueError("Failed to create GenericDataLoader. Please check the input data and columns.") from e

        # Fit the synthesizer to the data
        try:
            synthesizer.fit(data)
        except Exception as e:
            raise RuntimeError("Error occurred during model training. Please ensure the data and model are properly configured.") from e

        # Generate synthetic data
        try:
            synthetic_data = synthesizer.generate(count=sample_size).dataframe()
        except Exception as e:
            raise RuntimeError("Error occurred during synthetic data generation.") from e

        # Ensure the synthetic data has the correct columns
        synthetic_data.columns = data_columns
        synthetic_data.insert(0, identifier_column, range(1, len(synthetic_data) + 1))

        # Split synthetic data into multiple tables if using multi-table
        if table_type == 'multi':
            split_synthetic_dfs = []
            for key, columns in column_dict.items():
                split_synthetic_dfs.append(synthetic_data[columns])
            synthetic_data = split_synthetic_dfs

        # Save the model and/or synthetic data if a save location is provided
        if save_location is not None:
            try:
                save_to_file(save_location, synthesizer)
            except Exception as e:
                raise IOError(f"Failed to save the model to the specified location: {save_location}") from e

        return synthetic_data

    except ValueError as ve:
        print(f"ValueError: {str(ve)}")
    except KeyError as ke:
        print(f"KeyError: {str(ke)}")
    except TypeError as te:
        print(f"TypeError: {str(te)}")
    except IOError as ioe:
        print(f"IOError: {str(ioe)}")
    except RuntimeError as re:
        print(f"RuntimeError: {str(re)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")