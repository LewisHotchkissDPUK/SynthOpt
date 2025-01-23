from synthopt.process.structural_metadata import process_structural_metadata
from synthopt.process.data_processing import best_fit
import pandas as pd
from tqdm import tqdm

def process_statistical_metadata(data, datetime_formats=None, table_name=None):
    try:
        if isinstance(data, dict):
            all_metadata = []

            for key, dataset in data.items():
                try:
                    # Process structural metadata for the current table
                    metadata, cleaned_data = process_structural_metadata(
                        dataset, datetime_formats, key, return_data=True
                    )
                    metadata.index = metadata['variable_name']

                    # Process numerical columns for best-fit distributions
                    try:
                        numerical_cleaned_data = cleaned_data.select_dtypes(include=['number'])
                        best_fit_metadata = best_fit(numerical_cleaned_data)
                    except Exception as e:
                        print(f"Error calculating best-fit distributions for table '{key}': {e}")
                        best_fit_metadata = pd.DataFrame()

                    # Join structural metadata with best-fit metadata
                    new_metadata = metadata.join(best_fit_metadata)
                    new_metadata = new_metadata.reset_index(drop=True)

                    # Append metadata for the current table
                    all_metadata.append(new_metadata)

                except Exception as e:
                    print(f"Error processing table '{key}': {e}")
                    continue  # Skip the current table and continue with the next one

            # Combine metadata from all tables
            final_combined_metadata = pd.concat(all_metadata, ignore_index=True) if all_metadata else pd.DataFrame()
            return final_combined_metadata

        else:
            # Process structural metadata for a single dataset
            metadata, cleaned_data = process_structural_metadata(data, datetime_formats, table_name, return_data=True)
            metadata.index = metadata['variable_name']

            # Process numerical columns for best-fit distributions
            try:
                numerical_cleaned_data = cleaned_data.select_dtypes(include=['number'])
                best_fit_metadata = best_fit(numerical_cleaned_data)
            except Exception as e:
                print(f"Error calculating best-fit distributions for the dataset: {e}")
                best_fit_metadata = pd.DataFrame()

            # Join structural metadata with best-fit metadata
            new_metadata = metadata.join(best_fit_metadata)
            new_metadata = new_metadata.reset_index(drop=True)

            return new_metadata

    except Exception as e:
        print(f"An unexpected error occurred while processing statistical metadata: {e}")
        return pd.DataFrame()  # Return an empty DataFrame as a fallback
