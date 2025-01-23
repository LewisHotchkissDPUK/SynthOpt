from synthopt.process.data_processing import detect_numerical_in_objects, detect_datetime_in_objects, detect_integer_in_floats, detect_categorical_strings, detect_categorical_numerical, encode_data
import pandas as pd
from tqdm import tqdm

def process_structural_metadata(data, datetime_formats=None, table_name=None, return_data=False):
    def process_single_dataframe(data, datetime_formats=None, table_name=None):
        try:
            ### Prepare the Data ###
            orig_data = data.copy()

            non_numerical_columns = data.select_dtypes(exclude=['number']).columns.tolist()

            # Detect numerical columns
            try:
                data, non_numerical_columns = detect_numerical_in_objects(data, non_numerical_columns)
            except Exception as e:
                print(f"Error detecting numerical columns: {e}")

            # Detect datetime columns
            try:
                data, datetime_columns, non_numerical_columns, column_date_format = detect_datetime_in_objects(
                    data, datetime_formats, non_numerical_columns
                )
            except Exception as e:
                print(f"Error detecting datetime columns: {e}")
                datetime_columns, column_date_format = [], {}

            # Detect integers in float columns
            try:
                data = detect_integer_in_floats(data)
            except Exception as e:
                print(f"Error detecting integers in float columns: {e}")

            # Detect categorical string columns
            try:
                data, categorical_string_columns, non_categorical_string_columns = detect_categorical_strings(
                    data, non_numerical_columns
                )
            except Exception as e:
                print(f"Error detecting categorical string columns: {e}")
                categorical_string_columns, non_categorical_string_columns = [], []

            # Encode categorical string columns
            try:
                data, column_mappings = encode_data(data, orig_data, categorical_string_columns)
            except Exception as e:
                print(f"Error encoding data: {e}")
                column_mappings = {}

            # Detect categorical numerical columns
            try:
                numerical_columns = list(set(data.columns) - set(non_numerical_columns) - set(datetime_columns))
                data, categorical_numerical_columns = detect_categorical_numerical(data, numerical_columns)
            except Exception as e:
                print(f"Error detecting categorical numerical columns: {e}")
                categorical_numerical_columns = []

            ### Create the Metadata ###
            metadata = pd.DataFrame(columns=['variable_name', 'datatype', 'completeness', 'values', 'coding', 'table_name'])

            for column in tqdm(data.columns, desc="Creating Metadata"):
                try:
                    completeness = (orig_data[column].notna().sum() / len(orig_data)) * 100

                    # Determine value range
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

                    # Determine datatype
                    if column in datetime_columns:
                        datatype = "datetime"
                    elif column in categorical_string_columns:
                        datatype = "categorical string"
                    elif column in non_categorical_string_columns:
                        datatype = "string"
                    elif column in numerical_columns:
                        if "float" in str(data[column].dtype):
                            datatype = "categorical float" if column in categorical_numerical_columns else "float"
                        else:
                            datatype = "categorical integer" if column in categorical_numerical_columns else "integer"
                    else:
                        datatype = "object"

                    # Handle columns with all NaN values
                    if data[column].isna().all():
                        datatype = "object"
                        value_range = None

                    # Set coding
                    if column in column_mappings:
                        coding = column_mappings[column]
                    elif column in column_date_format:
                        coding = column_date_format[column]
                    else:
                        coding = None

                    # Append metadata
                    new_row = pd.DataFrame({
                        "variable_name": [column],
                        "datatype": datatype,
                        "completeness": [completeness],
                        "values": [value_range],
                        "coding": [coding],
                        "table_name": [table_name] if table_name else ["None"]
                    })
                    metadata = pd.concat([metadata, new_row], ignore_index=True)

                except Exception as e:
                    print(f"Error creating metadata for column '{column}': {e}")
                    continue

            return metadata, data

        except Exception as e:
            print(f"An error occurred while processing the dataframe: {e}")
            return pd.DataFrame(), data

    ### New Stats-Specific Code ###
    try:
        if isinstance(data, dict):
            combined_metadata = pd.DataFrame()
            combined_data = {}

            for table_name, df in tqdm(data.items(), desc="Processing Tables"):
                try:
                    table_metadata, table_data = process_single_dataframe(df, datetime_formats, table_name)
                    combined_metadata = pd.concat([combined_metadata, table_metadata], ignore_index=True)
                    combined_data[table_name] = table_data
                except Exception as e:
                    print(f"Error processing table '{table_name}': {e}")
                    continue
        else:
            combined_metadata, combined_data = process_single_dataframe(data, datetime_formats, table_name)

        if return_data:
            return combined_metadata, combined_data
        else:
            return combined_metadata

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure
