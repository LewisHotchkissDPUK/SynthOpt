from synthopt.generate.data_generation import generate_random_string, generate_from_distributions
from synthopt.generate.data_generation import generate_random_value, convert_datetime, decode_categorical_string, completeness, add_identifier
import pandas as pd
from tqdm import tqdm

def generate_statistical_synthetic_data(metadata, num_records=1000, identifier_column=None):
    def generate_data_for_column(column_metadata):
        data_type = column_metadata['datatype']
        if data_type == 'string':
            return [generate_random_string() for _ in range(num_records)]
        elif data_type == 'object':
            return None
        else:
            return generate_from_distributions(column_metadata, num_records)

    # Check if there is more than one table_name
    if metadata['table_name'].nunique() > 1:
        # Initialize a dictionary to hold table_name-to-dataframe pairs
        synthetic_data_by_table = {}

        # Group metadata by table_name to handle each table separately
        grouped_metadata = metadata.groupby('table_name')

        for table_name, table_metadata in grouped_metadata:
            synthetic_data = pd.DataFrame()
            # Iterate over columns in the table's metadata
            for index, column_metadata in tqdm(table_metadata.iterrows(), desc=f"Generating Data for Table: {table_name}"):
                column_name = column_metadata['variable_name']
                synthetic_data[column_name] = generate_data_for_column(column_metadata)

                if (column_metadata['datatype'] == 'categorical string') or (column_metadata['datatype'] == 'categorical integer') or (column_metadata['datatype'] == 'integer'):
                    synthetic_data[column_name] = synthetic_data[column_name].round().astype(int)

            # Post-processing steps
            synthetic_data = convert_datetime(table_metadata, synthetic_data)
            synthetic_data = decode_categorical_string(table_metadata, synthetic_data)
            synthetic_data = completeness(table_metadata, synthetic_data)

            if identifier_column is not None and identifier_column in synthetic_data.columns.tolist():
                synthetic_data = add_identifier(synthetic_data, table_metadata, identifier_column, num_records)
            
            # Add the synthetic data for the table to the dictionary
            synthetic_data_by_table[table_name] = synthetic_data

        return synthetic_data_by_table

    else:
        # If there is only one table, generate a single dataframe
        synthetic_data = pd.DataFrame()

        for index, column_metadata in tqdm(metadata.iterrows(), desc="Generating Synthetic Data"):
            column_name = column_metadata['variable_name']
            synthetic_data[column_name] = generate_data_for_column(column_metadata)

        # Post-processing steps
        synthetic_data = convert_datetime(metadata, synthetic_data)
        synthetic_data = decode_categorical_string(metadata, synthetic_data)
        synthetic_data = completeness(metadata, synthetic_data)

        if identifier_column is not None and identifier_column in synthetic_data.columns.tolist():
            synthetic_data = add_identifier(synthetic_data, metadata, identifier_column, num_records)

        return synthetic_data

# Usage example:
# synthetic_data_dict = generate_statistical_synthetic_data(metadata, num_records=1000)