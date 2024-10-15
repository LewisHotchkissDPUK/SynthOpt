import pandas as pd
import numpy as np
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import load, load_from_file, save, save_to_file
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata, MultiTableMetadata
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

def create_relational_metadata(data):
    metadata = MultiTableMetadata()
    metadata.detect_from_dataframes(data)
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
            processed_df, control_df = train_test_split(processed_df, test_size=0.1)
            #if subset_size != None:
            #    subset_size = subset_size * 0.9
            #    processed_df = processed_df.sample(n=subset_size)
            processed_dataframes.append(processed_df)
            control_dataframes.append(control_df)

        return processed_dataframes, control_dataframes

    elif table_type == 'single':
        imputer = KNNImputer(n_neighbors=3)
        data_processed = imputer.fit_transform(data)
        data_processed = pd.DataFrame(data_processed, columns=data.columns)
        #if subset_size != None:
        #    subset_size = subset_size * 0.9
        #    data_processed = data_processed.sample(n=subset_size)
        data_processed, control_data = train_test_split(data_processed, test_size=0.1, random_state=42)

        return data_processed, control_data
    
    elif table_type == "relational":
        # Detect relational metadata
        metadata = create_relational_metadata(data)

        # Step 1: Collect primary keys using get_table_metadata
        primary_keys = {table_name: metadata.get_table_metadata(table_name).primary_key for table_name in data.keys()}

        # Step 2: Collect foreign keys from relationships
        foreign_keys = {}
        for relationship in metadata.relationships:
            child_table = relationship["child_table_name"]
            foreign_key = relationship["child_foreign_key"]
            if child_table not in foreign_keys:
                foreign_keys[child_table] = set()
            foreign_keys[child_table].add(foreign_key)

        # Dictionary to hold the results
        result = {}

        # Step 3: Process each dataframe
        for table_name, df in data.items():
            keys_to_remove = set()

            # Get the primary key for this table (if it exists)
            primary_key = primary_keys.get(table_name)
            if primary_key:
                keys_to_remove.add(primary_key)

            # Get the foreign keys for this table (if they exist)
            table_foreign_keys = list(foreign_keys.get(table_name, set()))
            keys_to_remove.update(table_foreign_keys)

            # Step 4: Drop primary and foreign keys from the dataframe
            df_cleaned = df.drop(columns=keys_to_remove, errors='ignore')

            # Step 5: Impute missing values using KNNImputer
            if not df_cleaned.empty:
                imputer = KNNImputer(n_neighbors=3)
                df_imputed = imputer.fit_transform(df_cleaned)
                df_imputed = pd.DataFrame(df_imputed, columns=df_cleaned.columns)
            else:
                df_imputed = df_cleaned

            # Step 6: Split the dataframe into training and control sets
            train_df, control_df = train_test_split(df_imputed, test_size=0.1, random_state=42)

            # Step 7: Add to the result dictionary
            result[table_name] = {
                "training_data": train_df,
                "control_data": control_df,
                "primary_key": primary_key,
                "foreign_keys": table_foreign_keys,
            }

        # Step 8: Return the result dictionary
        return result
    
    else:
        print("Please select an appropriate table type")
        return None

def generate_syntheticdata(data, identifier_column=None, prediction_column=None, sensitive_columns=None, sample_size=None, table_type='single', model_name='pategan', iterations=100, dp_epsilon=1, dp_delta=None, dp_lambda=0.001, save_location=None):
    try:
        # Check if data is a pandas DataFrame (for single table) or a list of DataFrames (for multi table)
        if table_type == 'single' and not isinstance(data, pd.DataFrame):
            raise ValueError("For single table type, data must be a pandas DataFrame.")
        if table_type == 'multi' and not isinstance(data, list):
            raise ValueError("For multi table type, data must be a list of pandas DataFrames.")
        if table_type == 'relational' and not isinstance(data, dict):
            raise ValueError("For relational table type, data must be a dictionary of table name : pandas DataFrames pairs.")
        
        if sample_size == None:
            sample_size = len(data)

        if table_type == "relational":
            return generate_relational_syntheticdata(data, iterations) #save_location, model type etc
        
        metadata = create_metadata(data)

        # Soemtimes there may not be an identifier column, how should this be handled.
        #try:
        #    if identifier_column == None:
        #        identifier_column = metadata.primary_key
        #except Exception:
        #    print("Couldnt detect the identifier column, please specify")

        # Multi-table handling
        if table_type == 'multi':
            column_dict = {}
            for i, df in enumerate(data):
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(f"Element {i+1} in the data list is not a pandas DataFrame.")
                column_dict[f"DataFrame_{i+1}"] = df.columns.tolist()

            if identifier_column != None:
                try:
                    data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data)
                except KeyError as e:
                    raise KeyError(f"Identifier column '{identifier_column}' not found in one or more DataFrames.") from e

        if identifier_column != None and identifier_column in data.columns:
            data = data.drop(columns=[identifier_column])

        # Drop identifier column from the data
        #if identifier_column != None and identifier_column not in data.columns:
        #    raise KeyError(f"Identifier column '{identifier_column}' not found in the data.")
        #data = data.drop(columns=[identifier_column])

        # Check for object or string columns that are not allowed
        object_or_string_cols = data.select_dtypes(include=['object', 'string'])
        if not object_or_string_cols.empty:
            raise TypeError(f"Data must not contain string or object data types. Columns with object or string types: {list(object_or_string_cols.columns)}")
        
        
        data_columns = data.columns

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
        if identifier_column != None:
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


def generate_relational_syntheticdata(data, iterations):
    # Call the process_and_split_dataframes function to get processed data
    processed_data = process(data, 'relational')

    synthetic_data_dict = {}

    for table_name, table_info in processed_data.items():
        training_data = table_info['training_data']
        primary_key = table_info['primary_key']
        foreign_keys = table_info['foreign_keys']

        # Step 1: Generate synthetic data using synthcity
        print(f"Generating synthetic data for table: {table_name}")

        # Convert the training data into a format that synthcity can use
        data_loader = GenericDataLoader(training_data)

        # Choose a synthetic data generation plugin from synthcity
        plugin = Plugins().get("ctgan", n_iter=iterations)

        # Generate synthetic data (same number of records as the training data)
        synthetic_data = plugin.fit(data_loader).generate(len(training_data)).dataframe()

        # Convert the synthetic data back to a dataframe
        synthetic_df = synthetic_data

        # Step 2: Add unique primary keys to the synthetic data
        if primary_key:
            synthetic_df.insert(0, primary_key, np.arange(1, len(synthetic_df) + 1))

        # Step 3: Handle foreign keys while preserving the original frequency distribution
        for foreign_key in foreign_keys:
            if foreign_key in data[table_name].columns:
                # Get frequency distribution of foreign keys in the original data
                foreign_key_distribution = data[table_name][foreign_key].value_counts(normalize=True)

                # Generate synthetic foreign key values based on the distribution
                synthetic_foreign_keys = np.random.choice(
                    foreign_key_distribution.index,
                    size=len(synthetic_df),
                    p=foreign_key_distribution.values
                )

                # Add the foreign keys to the synthetic dataframe
                synthetic_df[foreign_key] = synthetic_foreign_keys
            else:
                print(f"Warning: Foreign key '{foreign_key}' not found in the original data for table '{table_name}'.")


        # Step 4: Add the synthetic dataframe to the result dictionary
        synthetic_data_dict[table_name] = synthetic_df

    return synthetic_data_dict
