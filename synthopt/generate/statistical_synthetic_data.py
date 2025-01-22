from synthopt.generate.data_generation import generate_random_string, generate_from_distributions
from tqdm import tqdm

def generate_statistical_synthetic_data(metadata, num_records=1000, identifier_column=None):
    def generate_data_for_column(column_metadata):
        try:
            data_type = column_metadata['datatype'].astype(str)
        except:
            data_type = column_metadata['datatype']
        if data_type not in ['string', 'object']:
            return generate_from_distributions(column_metadata, num_records)
        elif data_type == 'object':
            return None
        else:
            return [generate_random_string() for _ in range(num_records)]
        #else:
        #    raise ValueError(f"Unsupported data type: {data_type}")

    print()
    print("generating synthetic data")
    print(metadata)

    synthetic_data = {}
    for index, column_metadata in tqdm(metadata.iterrows(), desc="Generating Synthetic Data"):
        print(column_metadata)
        column_name = column_metadata['variable_name']
        synthetic_data[column_name] = generate_data_for_column(column_metadata)
    return synthetic_data



## Need to add handling to convert dates, categories, etc. to the correct format. ID column handling is also needed.
## if integer originally then convert back to integer