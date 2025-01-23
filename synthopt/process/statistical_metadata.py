from synthopt.process.structural_metadata import process_structural_metadata
from synthopt.process.data_processing import best_fit
import pandas as pd
from tqdm import tqdm

def process_statistical_metadata(data, datetime_formats=None, table_name=None):
    if isinstance(data, dict):
        all_metadata = []
        for key, dataset in data.items():
            metadata, cleaned_data = process_structural_metadata(dataset, datetime_formats, key, return_data=True)
            metadata.index = metadata['variable_name']

            numerical_cleaned_data = cleaned_data.select_dtypes(include=['number'])
            best_fit_metadata = best_fit(numerical_cleaned_data)
            new_metadata = metadata.join(best_fit_metadata)
            new_metadata = new_metadata.reset_index(drop=True)

            all_metadata.append(new_metadata)
        final_combined_metadata = pd.concat(all_metadata, ignore_index=True)
        return final_combined_metadata
    else:
        metadata, cleaned_data = process_structural_metadata(data, datetime_formats, table_name, return_data=True)
        metadata.index = metadata['variable_name']

        numerical_cleaned_data = cleaned_data.select_dtypes(include=['number'])
        best_fit_metadata = best_fit(numerical_cleaned_data)
        new_metadata = metadata.join(best_fit_metadata)
        new_metadata = new_metadata.reset_index(drop=True)

        return new_metadata