from synthopt.process.data_processing import detect_numerical_in_objects, detect_datetime_in_objects, detect_integer_in_floats, detect_categorical_strings, detect_categorical_numerical, encode_data
import pandas as pd
from tqdm import tqdm

def process_structural_metadata(data, datetime_formats=None, table_name=None, return_data=False):
    def process_single_dataframe(data, datetime_formats=None, table_name=None):
        ### prepare the data ###
        orig_data = data.copy()

        non_numerical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
        data, non_numerical_columns = detect_numerical_in_objects(data, non_numerical_columns)

        data, datetime_columns, non_numerical_columns, column_date_format = detect_datetime_in_objects(data, datetime_formats, non_numerical_columns)

        data = detect_integer_in_floats(data)

        data, categorical_string_columns, non_categorical_string_columns = detect_categorical_strings(data, non_numerical_columns)

        data, column_mappings = encode_data(data, orig_data, categorical_string_columns) # should only happen for stats/corr versions

        numerical_columns = list(set(data.columns) - set(non_numerical_columns) - set(datetime_columns))
        data, categorical_numerical_columns = detect_categorical_numerical(data, numerical_columns)


        ### Create the Metadata ###

        metadata = pd.DataFrame(columns=['variable_name', 'datatype', 'completeness', 'values', 'coding', 'table_name'])

        for column in tqdm(data.columns, desc="Creating Metadata"):
            completeness = (orig_data[column].notna().sum() / len(orig_data)) * 100

            if column in non_categorical_string_columns:
                value_range = None
            else:
                try:
                    if (column in categorical_numerical_columns) or (column in categorical_string_columns):
                        value_range = data[column].dropna().unique().tolist()
                    else:
                        value_range = (data[column].min(), data[column].max())
                except Exception:
                    value_range = None

            if column in datetime_columns:
                datatype = "datetime"
            elif column in categorical_string_columns:
                datatype = "categorical string"
            elif column in non_categorical_string_columns:
                datatype = "string"
            elif column in numerical_columns:
                if "float" in str(data[column].dtype):
                    if column in categorical_numerical_columns:
                        datatype = "categorical float"
                    else:
                        datatype = "float"
                else:
                    if column in categorical_numerical_columns:
                        datatype = "categorical integer"
                    else:
                        datatype = "integer"
            else:
                datatype = "object"

            if data[column].isna().all():
                datatype = "object"
                value_range = None

            if column in column_mappings:
                coding = column_mappings[column]
            elif column in column_date_format:
                coding = column_date_format[column]
            else:
                coding = None

            new_row = pd.DataFrame({
                    "variable_name": [column],
                    "datatype": datatype,
                    "completeness": [completeness],
                    "values": [value_range],
                    "coding": [coding],
                    "table_name": [table_name] if table_name else ["None"]})
            metadata = pd.concat([metadata, new_row], ignore_index=True)

        return metadata, data

    ### new stats specific code ###
    if isinstance(data, dict):
        combined_metadata = pd.DataFrame()

        for table_name, df in tqdm(data.items(), desc="Processing Tables"):
            table_metadata, table_data = process_single_dataframe(df, datetime_formats, table_name)
            combined_metadata = pd.concat([combined_metadata, table_metadata], ignore_index=True)
            combined_data = {table_name: table_data}
    else:
        combined_metadata, combined_data = process_single_dataframe(data, datetime_formats, table_name)

    if return_data == True:
        return combined_metadata, combined_data
    else:
        return combined_metadata