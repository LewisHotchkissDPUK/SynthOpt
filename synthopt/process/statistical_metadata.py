from synthopt.process.structural_metadata import process_structural_metadata
from synthopt.process.data_processing import best_fit
import pandas as pd
from tqdm import tqdm

def process_statistical_metadata(data, datetime_formats=None, table_name=None, return_correlations=False):
    if isinstance(data, dict):
        all_metadata = []
        all_correlations = {}
        for key, dataset in data.items():
            metadata, cleaned_data = process_structural_metadata(dataset, datetime_formats, key, return_data=True)
            metadata.index = metadata['variable_name']
            table_name = metadata['table_name'].values[0]

            numerical_cleaned_data = cleaned_data.select_dtypes(include=['number'])

            if return_correlations:
                numerical_cleaned_data = numerical_cleaned_data.dropna(axis=1, how='all') # handles complete NaN columns
                numerical_cleaned_data = numerical_cleaned_data.loc[:, numerical_cleaned_data.nunique() > 1]

            best_fit_metadata = best_fit(numerical_cleaned_data)
            new_metadata = metadata.join(best_fit_metadata)
            new_metadata = new_metadata.reset_index(drop=True)

            if return_correlations:
                correlation_matrix = numerical_cleaned_data.astype(float).fillna(numerical_cleaned_data.mean()).corr()

            all_metadata.append(new_metadata)

            if return_correlations:
                all_correlations[table_name] = correlation_matrix

        final_combined_metadata = pd.concat(all_metadata, ignore_index=True)

        if return_correlations:
            return final_combined_metadata, all_correlations
        else:
            return final_combined_metadata
        
    else:
        metadata, cleaned_data = process_structural_metadata(data, datetime_formats, table_name, return_data=True)
        metadata.index = metadata['variable_name']

        numerical_cleaned_data = cleaned_data.select_dtypes(include=['number'])

        if return_correlations:
            numerical_cleaned_data = numerical_cleaned_data.dropna(axis=1, how='all') # handles complete NaN columns
            numerical_cleaned_data = numerical_cleaned_data.loc[:, numerical_cleaned_data.nunique() > 1]
        
        best_fit_metadata = best_fit(numerical_cleaned_data)
        new_metadata = metadata.join(best_fit_metadata)
        new_metadata = new_metadata.reset_index(drop=True)

        if return_correlations:
            correlation_matrix = numerical_cleaned_data.astype(float).fillna(numerical_cleaned_data.mean()).corr()
            return new_metadata, correlation_matrix

        return new_metadata