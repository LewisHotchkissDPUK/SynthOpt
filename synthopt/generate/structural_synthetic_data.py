from synthopt.generate.data_generation import generate_random_value, convert_datetime, decode_categorical_string, completeness, add_identifier
from tqdm import tqdm
import pandas as pd

def generate_structural_synthetic_data(metadata, num_records=1000, identifier_column=None):
    metadata = metadata.copy()

    # Initialize a dictionary to hold generated data for each table
    generated_data = {}

    # Create a mapping for each table to handle variable generation
    table_variable_mapping = {}

    for index, row in metadata.iterrows():
        table_name = row["table_name"]
        variable_name = row["variable_name"]

        # Initialize the table if it doesn't exist
        if table_name not in table_variable_mapping:
            table_variable_mapping[table_name] = []

        # Append variable row details to the specific table
        table_variable_mapping[table_name].append(row)

    # Loop through each table and generate its data
    for table_name, variables in table_variable_mapping.items():
        generated_data[table_name] = {}

        for row in tqdm(variables, desc="Generating Synthetic Data"):
            column_name = row["variable_name"]
            data = []

            # Generate data for the current variable
            for _ in range(num_records):
                value = generate_random_value(row)
                data.append(value)

            generated_data[table_name][column_name] = data

        # Create DataFrame for the current table and do conversions
        generated_data[table_name] = pd.DataFrame(generated_data[table_name])
        generated_data[table_name] = convert_datetime(metadata, generated_data[table_name])
        generated_data[table_name] = decode_categorical_string(
            metadata, generated_data[table_name]
        )
        generated_data[table_name] = completeness(metadata, generated_data[table_name])
        if identifier_column is not None:
            if identifier_column in generated_data[table_name].columns.tolist():
                generated_data[table_name] = add_identifier(
                    generated_data[table_name], metadata, identifier_column, num_records
                )

    if "None" in generated_data:
        generated_data = generated_data["None"]

    return generated_data
