import pandas as pd
import numpy as np
import random
import random
import warnings
warnings.filterwarnings('ignore')
import ast
import string


def generate_random_string():
    return "".join(
        random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(5, 10))
    )


def generate_random_integer(value_range):
    if isinstance(value_range, (list)):
        return random.choice(value_range)
    elif isinstance(value_range, (tuple)):
        if value_range == (0, 1):  # if binary
            return random.choice([0, 1])
        else:
            return random.randint(value_range[0], value_range[1])
    else:
        return None


def generate_random_float(value_range):
    if isinstance(value_range, (list)):
        return random.choice(value_range)
    elif isinstance(value_range, (tuple)):
        return random.uniform(value_range[0], value_range[1])
    else:
        return None


def convert_datetime(metadata, generated_data):
    for column in generated_data.columns:
        if (
            "date"
            in metadata[metadata["variable_name"] == column]["datatype"].values[0]
        ):
            datetime_format = metadata[metadata["variable_name"] == column][
                "coding"
            ].values[0]
            generated_data[column] = pd.to_datetime(
                generated_data[column].astype("Int64"), unit="s"
            ).dt.strftime(datetime_format)
    return generated_data


def decode_categorical_string(metadata, data):
    for _, row in metadata.iterrows():
        if row["datatype"] == "categorical string":
            variable = row["variable_name"]
            if not isinstance(row["coding"], dict):
                coding = ast.literal_eval(row["coding"])
            else:
                coding = row["coding"]
            if variable in data.columns:
                data[variable] = data[variable].map(coding)
    return data


def completeness(metadata, data):
    adjusted_data = data.copy()
    num_rows = len(data)
    for _, row in metadata.iterrows():
        col_name = row["variable_name"]
        completeness_level = row["completeness"]
        if col_name in adjusted_data.columns and not pd.isna(completeness_level):
            retain_count = int((completeness_level / 100) * num_rows)
            if retain_count < num_rows:
                retained_indices = np.random.choice(
                    adjusted_data.index, retain_count, replace=False
                )
                adjusted_data.loc[
                    ~adjusted_data.index.isin(retained_indices), col_name
                ] = np.nan
    return adjusted_data


def add_identifier(data, metadata, identifier_column, num_records):
    if identifier_column != None:
        if (
            "integer"
            in metadata[metadata["variable_name"] == "id"]["datatype"].values[0]
        ):
            participant_ids_integer = random.sample(
                range(1_000_000_000, 10_000_000_000), num_records
            )
            data[identifier_column] = participant_ids_integer
        elif (
            "float" in metadata[metadata["variable_name"] == "id"]["datatype"].values[0]
        ):
            participant_ids_float = [
                random.uniform(1_000_000_000, 10_000_000_000)
                for _ in range(num_records)
            ]
            data[identifier_column] = participant_ids_float
        else:
            participant_ids_string = [
                "".join(random.choices(string.ascii_letters + string.digits, k=10))
                for _ in range(num_records)
            ]
            data[identifier_column] = participant_ids_string
    return data


def generate_random_value(row):
    dtype = row["datatype"]
    value_range = row["values"]
    if "string" in str(dtype) and "categorical" not in str(dtype):
        return generate_random_string()
    else:
        try:
            if isinstance(value_range, str):
                value_range = eval(
                    value_range
                )  # Evaluate the string representation of a tuple/list
            if isinstance(value_range, (tuple, list)):
                if (
                    "int" in str(dtype)
                    or "date" in str(dtype)
                    or "categorical" in str(dtype)
                ):
                    return generate_random_integer(value_range)
                elif "float" in str(dtype):
                    return generate_random_float(value_range)
        except Exception as e:
            return None
