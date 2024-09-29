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

# create a method to pass in a custom model (not a name but an actual model)
# add other model options from other packages like sdv and ydata
# allow option for single table, multi table and longitudinal
# handle string columns, maybe do encoding
def generate_syntheticdata(data, identifier_column, prediction_column, sensitive_columns, sample_size, table_type = 'single', model_name = 'pategan', iterations = 100, dp_epsilon = 1, dp_delta = None, dp_lambda = None, save_location=None):
    if table_type == 'multi':
        column_dict = {}
        for i, df in enumerate(data):
            column_dict[f"DataFrame_{i+1}"] = df.columns.tolist()
        data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data)
    
    data = data.drop(columns=[identifier_column])

    object_or_string_cols = data.select_dtypes(include=['object', 'string'])
    if not object_or_string_cols.empty:
        raise TypeError(f"Data must not contain string or object data types, please handle these. Columns with object or string types: {list(object_or_string_cols.columns)}")

    metadata = create_metadata(data)
    available_columns = data.columns.tolist()
    discrete_columns = []
    for col, meta in metadata.columns.items():
        if ('sdtype' in meta and meta['sdtype'] == 'categorical') or (data[col].fillna(9999) % 1 == 0).all():
            discrete_columns.append(col)
    data_columns = data.columns

    if sample_size == None:
        sample_size = len(data)

    if model_name != "ctgan":
        # add delta and lambda for dpgan and pategan
        synthesizer = Plugins().get(model_name, n_iter=iterations, epsilon=dp_epsilon)
    else:
        synthesizer = Plugins().get(model_name, n_iter=iterations)

    for column in data_columns:
        if (data[column] % 1).all() == 0:
            data[column] = data[column].astype(int)
            
    if model_name == "ctgan":
        data = add_noise(data, dp_epsilon, discrete_columns) # maybe should be the inverse of epsilon

    data = GenericDataLoader(data, target_column=prediction_column, sensitive_columns=sensitive_columns)
    synthesizer.fit(data)
    
    synthetic_data = synthesizer.generate(count=sample_size).dataframe()
    synthetic_data.columns = data_columns
    synthetic_data.insert(0, identifier_column, range(1, len(synthetic_data) + 1))

    if table_type == 'multi':
        split_synthetic_dfs = []
        for key, columns in column_dict.items():
            split_synthetic_dfs.append(synthetic_data[columns])
        synthetic_data = split_synthetic_dfs

    # Generate unique ten-digit identifiers
    #num_rows = len(synthetic_data)
    #unique_identifiers = set()
    #while len(unique_identifiers) < num_rows:
    #    identifier = random.randint(1000000000, 9999999999)
    #    unique_identifiers.add(identifier)
    # Convert the set to a list and insert it into the DataFrame
    #synthetic_data.insert(0, identifier_column, list(unique_identifiers))

    # save datasets if save location exists, and model is model save location exists
    if save_location != None:
        save_to_file(save_location, synthesizer)

    return synthetic_data