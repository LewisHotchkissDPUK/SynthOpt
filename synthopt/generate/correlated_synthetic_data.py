from synthopt.generate.data_generation import generate_random_string, generate_from_distributions, generate_from_correlations
from synthopt.generate.data_generation import generate_random_value, convert_datetime, decode_categorical_string, completeness, add_identifier, enforce_categorical_validity
import pandas as pd
from tqdm import tqdm

def generate_correlated_synthetic_data(metadata, correlation_matrices, num_records=1000, identifier_column=None):
    def generate_data_for_non_object_columns(table_metadata, num_records, correlation_matrix):
        # Filter metadata for columns that are not 'object' or 'string'
        eligible_columns = table_metadata[
            ~table_metadata['datatype'].isin(['object', 'string']) & 
            table_metadata['dist'].notna()
        ]
        if eligible_columns.empty:
            return pd.DataFrame()

        # Generate data for all eligible columns at once using multivariate sampling
        column_names = eligible_columns['variable_name'].tolist()
        generated_data = generate_from_correlations(eligible_columns, num_records, correlation_matrix)
        return pd.DataFrame(generated_data, columns=column_names)

    synthetic_data_by_table = {}
    grouped_metadata = metadata.groupby('table_name')

    for table_name, table_metadata in grouped_metadata:
        synthetic_data = pd.DataFrame()

        # Generate data for non-object columns in one go
        if isinstance(correlation_matrices, dict):
            correlation_matrix = correlation_matrices.get(table_name, None)
        else:
            correlation_matrix = correlation_matrices

        non_object_data = generate_data_for_non_object_columns(table_metadata, num_records, correlation_matrix)
        synthetic_data = pd.concat([synthetic_data, non_object_data], axis=1)

        # Generate data for 'string' columns individually
        for _, column_metadata in tqdm(table_metadata.iterrows(), desc=f"Generating Data for Table: {table_name}"):
            column_name = column_metadata['variable_name']
            data_type = column_metadata['datatype']

            if data_type == 'string':
                synthetic_data[column_name] = [generate_random_string() for _ in range(num_records)]
            elif data_type == 'object':
                synthetic_data[column_name] = None

            if data_type in ['categorical string', 'categorical integer', 'integer']:
                try:
                    synthetic_data[column_name] = synthetic_data[column_name].round().astype(int)
                except Exception:
                    pass  # Skip conversion if rounding or casting fails

        # Post-processing
        synthetic_data = convert_datetime(table_metadata, synthetic_data)
        synthetic_data = enforce_categorical_validity(table_metadata, synthetic_data)
        synthetic_data = decode_categorical_string(table_metadata, synthetic_data)
        synthetic_data = completeness(table_metadata, synthetic_data)

        if identifier_column is not None and identifier_column in synthetic_data.columns.tolist():
            synthetic_data = add_identifier(
                synthetic_data, table_metadata, identifier_column, num_records
            )

        synthetic_data_by_table[table_name] = synthetic_data

    # If only one table, return DataFrame instead of dict
    if len(synthetic_data_by_table) == 1:
        return list(synthetic_data_by_table.values())[0]

    return synthetic_data_by_table