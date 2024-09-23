import pandas as pd
import numpy as np
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import load, load_from_file, save, save_to_file
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sdv.metadata import SingleTableMetadata
import random

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
# create method for optimisation
# allow option for single table, multi table and longitudinal
# pass in identifier column
# handle string columns, maybe do encoding
def generate_syntheticdata(model_name, data, control_data, prediction_column, sensitive_columns, key_columns, iterations, sample_size, dp_epsilon, dp_delta, dp_lambda):
    data = data.select_dtypes(exclude=['object']) # need to properly handle
    metadata = create_metadata(data)
    data, control_data = train_test_split(data, test_size=0.1, random_state=42)
    available_columns = data.columns.tolist()
    discrete_columns = []
    for col, meta in metadata.columns.items():
        if ('sdtype' in meta and meta['sdtype'] == 'categorical') or (data[col].fillna(9999) % 1 == 0).all():
            discrete_columns.append(col)
    data_columns = data.columns

    if sample_size == None:
        sample_size = len(data)

    if model_name != "ctgan":
        synthesizer = Plugins().get(model_name, n_iter=iterations, epsilon=dp_epsilon)
    else:
        synthesizer = Plugins().get(model_name, n_iter=iterations)

    DATA_PROCESSED = data
    
    imputer = KNNImputer(n_neighbors=3)
    DATA_PROCESSED = imputer.fit_transform(DATA_PROCESSED)
    DATA_PROCESSED = pd.DataFrame(DATA_PROCESSED, columns=data_columns)

    for column in data_columns:
        if (DATA_PROCESSED[column] % 1).all() == 0:
            DATA_PROCESSED[column] = DATA_PROCESSED[column].astype(int)
            
    if model_name == "ctgan":
        DATA_PROCESSED = add_noise(DATA_PROCESSED, dp_epsilon, discrete_columns) 

    DATA_PROCESSED = GenericDataLoader(DATA_PROCESSED, target_column=prediction_column, sensitive_columns=sensitive_columns)
    synthesizer.fit(DATA_PROCESSED)
    
    synthetic_data = synthesizer.generate(count=sample_size).dataframe()
    synthetic_data.columns = data_columns

    # save datasets if save location exists, and model is model save location exists
    
    
    return synthetic_data