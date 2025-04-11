from synthopt.generate.data_generation import generate_random_string, generate_from_distributions
from synthopt.generate.data_generation import generate_random_value, convert_datetime, decode_categorical_string, completeness, add_identifier, enforce_categorical_validity
import pandas as pd
from tqdm import tqdm

def generate_statistical_synthetic_data(metadata, num_records=1000, identifier_column=None):
    def generate_data_for_column(column_metadata, num_records):
        data_type = column_metadata['datatype']
        if data_type == 'string':
            return [generate_random_string() for _ in range(num_records)]
        elif data_type == 'object':
            return None
        else:
            return generate_from_distributions(column_metadata, num_records)

    synthetic_data_by_table = {}
    grouped_metadata = metadata.groupby('table_name')

    for table_name, table_metadata in grouped_metadata:
        synthetic_data = pd.DataFrame()

        for _, column_metadata in tqdm(table_metadata.iterrows(), desc=f"Generating Data for Table: {table_name}"):
            column_name = column_metadata['variable_name']
            synthetic_data[column_name] = generate_data_for_column(column_metadata, num_records)

            if column_metadata['datatype'] in ['categorical string', 'categorical integer', 'integer']:
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