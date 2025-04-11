from synthopt.generate.data_generation import generate_random_string, generate_from_distributions, generate_from_correlations
from synthopt.generate.data_generation import generate_random_value, convert_datetime, decode_categorical_string, completeness, add_identifier, enforce_categorical_validity
import pandas as pd
from tqdm import tqdm

def generate_correlated_synthetic_data(metadata, num_records=1000, correlation_matrices=None, identifier_column=None):
    synthetic_data_by_table = {}
    grouped_metadata = metadata.groupby('table_name')

    is_single_table = len(grouped_metadata) == 1
    if is_single_table and isinstance(correlation_matrices, pd.DataFrame):
        # Wrap into dict for consistent logic
        table_name = list(grouped_metadata.groups.keys())[0]
        correlation_matrices = {table_name: correlation_matrices}

    for table_name, table_metadata in grouped_metadata:
        # Separate metadata by datatype
        non_string_metadata = table_metadata[
            ~table_metadata['datatype'].isin(['string', 'object'])
        ]
        string_metadata = table_metadata[
            table_metadata['datatype'].isin(['string', 'object'])
        ]

        non_string_vars = non_string_metadata['variable_name'].tolist()
        string_vars = string_metadata['variable_name'].tolist()

        # Get the correlation matrix for this table
        if correlation_matrices is None or table_name not in correlation_matrices:
            raise ValueError(f"No correlation matrix provided for table: {table_name}")
        table_corr_matrix = correlation_matrices[table_name].loc[non_string_vars, non_string_vars]

        # Generate correlated data for numeric/categorical variables
        synthetic_non_string = generate_from_correlations(non_string_metadata, num_records, table_corr_matrix)

        # Generate data for string/object variables
        synthetic_string = pd.DataFrame()
        for _, column_metadata in string_metadata.iterrows():
            var = column_metadata['variable_name']
            if column_metadata['datatype'] == 'string':
                synthetic_string[var] = [generate_random_string() for _ in range(num_records)]
            elif column_metadata['datatype'] == 'object':
                synthetic_string[var] = [None] * num_records

        # Combine in original order
        full_columns = table_metadata['variable_name'].tolist()
        combined_df = pd.concat([synthetic_non_string, synthetic_string], axis=1)
        combined_df = combined_df.reindex(columns=full_columns)

        # Post-process
        combined_df = convert_datetime(table_metadata, combined_df)
        combined_df = enforce_categorical_validity(table_metadata, combined_df)
        combined_df = decode_categorical_string(table_metadata, combined_df)
        combined_df = completeness(table_metadata, combined_df)

        if identifier_column is not None and identifier_column in combined_df.columns.tolist():
            combined_df = add_identifier(combined_df, table_metadata, identifier_column, num_records)

        synthetic_data_by_table[table_name] = combined_df

    # Return single DataFrame if just one table
    return list(synthetic_data_by_table.values())[0] if is_single_table else synthetic_data_by_table


