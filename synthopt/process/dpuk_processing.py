from datetime import datetime
import pandas as pd
import numpy as np
import json
import json
from jinja2 import Template

def create_dpuk_metadata(metadata):
    metadata_temp = metadata.copy()
    
    def convert_timestamps(value):
        if isinstance(value, tuple) and metadata_temp[metadata_temp['values'] == value]['datatype'].iloc[0] == "datetime":
            converted = tuple(datetime.utcfromtimestamp(v).strftime('%Y-%m-%d %H:%M:%S') for v in value)
            cleaned = tuple(
                v.replace('1970-01-01 ', '').replace(' 00:00:00', '') if '1970-01-01' in v or '00:00:00' in v else v
                for v in converted)
            return cleaned
        return value
    
    metadata_temp['values'] = metadata_temp['values'].apply(convert_timestamps)
    
    for index, row in metadata_temp.iterrows():
        if row['datatype'] == 'categorical string' and row['coding'] is not None:
            coding_dict = eval(str(row['coding']))
            metadata_temp.at[index, 'values'] = list(coding_dict.values())
            metadata_temp.at[index, 'coding'] = None
        if isinstance(metadata_temp.at[index, 'coding'], str):
            metadata_temp.at[index, 'coding'] = metadata_temp.at[index, 'coding'].replace('%', '')
            metadata_temp.at[index, 'coding'] = metadata_temp.at[index, 'coding'].upper()
    
    def replace_commas_in_parentheses(value):
        if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
            value = value.replace(",", " to")
            value = value.replace("(", "")
            value = value.replace(")", "")
            return value
        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
            value = value.replace("[", "")
            value = value.replace("]", "")
        return value
    
    metadata_temp["values"] = metadata_temp["values"].astype(str)
    metadata_temp["values"] = metadata_temp["values"].apply(replace_commas_in_parentheses)
    metadata_temp["values"] = metadata_temp["values"].str.replace("'", "")
    metadata_temp["completeness"] = metadata_temp["completeness"].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
    
    # Insert additional columns
    metadata_temp.insert(1, "var_label", "")
    metadata_temp.insert(6, "wave", "")
    
    # Rename columns
    metadata_temp = metadata_temp.rename(columns={
        "variable_name": "var_name",
        "datatype": "data_type",
        "table_name": "filename",
    })
    
    # Ensure 'values' is set to NaN for datetime data type
    metadata_temp.loc[metadata_temp['data_type'] == 'datetime', 'values'] = float('nan')
    
    # Define column order
    columns_order = ["var_name", "var_label", "data_type", "values", "completeness", "coding", "wave", "filename"]
    metadata_temp = metadata_temp[columns_order]
    
    return metadata_temp


# REPORT GENERATION


def convert_to_native_type(value):
    if isinstance(value, np.generic):
        return value.item()
    elif isinstance(value, pd.Timestamp):
        return value.isoformat()
    elif isinstance(value, float) and pd.isna(value):
        return None
    return value


def compare_dataframes(df1, df2, file1_name, file2_name, description):
    comparison_results = {
        'file_1': file1_name,
        'file_2': file2_name,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'description': description,
        'changes': []
    }
    value_change_counts = {}

    for col in df1.columns:
        if col in df2.columns:
            dtype_df1 = str(df1[col].dtype)
            dtype_df2 = str(df2[col].dtype)
            if dtype_df1 != dtype_df2:
                comparison_results['changes'].append({
                    'type': 'column_dtype_change',
                    'column': col,
                    'from': dtype_df1,
                    'to': dtype_df2
                })
        else:
            comparison_results['changes'].append({'type': 'column_removed', 'column': col})

    for col in df2.columns:
        if col not in df1.columns:
            comparison_results['changes'].append({'type': 'column_added', 'column': col})

    for col in df1.columns:
        if col in df2.columns:
            for idx in df1.index:
                val_df1 = convert_to_native_type(df1.at[idx, col])
                val_df2 = convert_to_native_type(df2.at[idx, col])

                if val_df1 != val_df2:
                    # Convert both values to strings to avoid comparison issues
                    key = tuple([str(val_df1), str(val_df2)])

                    if col not in value_change_counts:
                        value_change_counts[col] = {}

                    value_change_counts[col][key] = value_change_counts[col].get(key, 0) + 1

    for col, changes in value_change_counts.items():
        comparison_results['changes'].append({
            'type': 'value_changes',
            'column': col,
            'changes': [{'original': orig, 'new': new, 'count': count} for (orig, new), count in changes.items()]
        })

    return comparison_results

def save_comparison_to_json(comparison_results, output_filename):
    comparison_results = json.loads(json.dumps(comparison_results, default=convert_to_native_type))
    with open(output_filename, 'w') as f:
        json.dump(comparison_results, f, indent=4)
        
        
def generate_html_report(json_data, output_file="report.html"):
    template = Template("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Transformation Report</title>
        <style>
            body {
                font-family: 'Arial', sans-serif;
                margin: 20px;
                padding: 20px;
                background-color: #f4f4f4;
            }
            .info-box {
                background: #e0e0e0;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .changes-container {
                display: flex;
                gap: 20px;
            }
            .changes-box {
                flex: 1;
                background: #e0e0e0;
                padding: 15px;
                border-radius: 8px;
            }
            h2 {
                color: #592981;
            }
            h3 {
                color: #333;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                background: white;
                border-radius: 8px;
                overflow: hidden;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }
            th {
                background-color: #592981;
                color: white;
            }
            select {
                margin-top: 10px;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            /* New style for Value Changes box */
            .value-changes-box {
                border: 2px solid #582981;
                padding: 10px;
                margin-top: 20px;
                border-radius: 8px;
                background-color: transparent;  /* No fill color */
            }
        </style>
        <script>
            function showChanges() {
                var selectBox = document.getElementById("columnSelect");
                var selectedColumn = selectBox.value;
                var changeTables = document.getElementsByClassName("changeTable");

                for (var i = 0; i < changeTables.length; i++) {
                    changeTables[i].style.display = "none";
                }

                if (selectedColumn) {
                    document.getElementById("changes_" + selectedColumn).style.display = "table";
                }
            }
        </script>
    </head>
    <body>
        <img src="https://portal.dementiasplatform.uk/wp-content/uploads/2023/06/DPUK-Data-Portal-Logo-300x55.png" alt="DPUK Data Portal Logo" style="max-width: 300px; margin-bottom: 20px;">
        <h2>Data Transformation Report</h2>
        <div class="info-box">
            <p><strong>Input File:</strong> {{ json_data['file_1'] }}</p>
            <p><strong>Output File:</strong> {{ json_data['file_2'] }}</p>
            <p><strong>Date:</strong> {{ json_data['date'] }}</p>
            <p><strong>Description:</strong> {{ json_data['description'] }}</p>
        </div>

        <div class="changes-container">
            <div class="changes-box">
                <h3>Column Type Changes</h3>
                <ul>
                    {% for change in json_data['changes'] if change['type'] == 'column_dtype_change' %}
                        <li><strong>{{ change['column'] }}:</strong> {{ change['from'] }} → {{ change['to'] }}</li>
                    {% endfor %}
                </ul>
            </div>
            <div class="changes-box">
                <h3>Added Columns</h3>
                <ul>
                    {% for change in json_data['changes'] if change['type'] == 'column_added' %}
                        <li><strong>{{ change['column'] }}</strong> was added</li>
                    {% endfor %}
                </ul>
            </div>
            <div class="changes-box">
                <h3>Removed Columns</h3>
                <ul>
                    {% for change in json_data['changes'] if change['type'] == 'column_removed' %}
                        <li><strong>{{ change['column'] }}</strong> was removed</li>
                    {% endfor %}
                </ul>
            </div>
        </div>

        <div class="value-changes-box">
            <h3>Value Changes</h3>
            <label for="columnSelect"><strong>Select Column:</strong></label>
            <select id="columnSelect" onchange="showChanges()">
                <option value="">-- Select --</option>
                {% for change in json_data['changes'] if change['type'] == 'value_changes' %}
                    <option value="{{ change['column'] }}">{{ change['column'] }}</option>
                {% endfor %}
            </select>

            {% for change in json_data['changes'] if change['type'] == 'value_changes' %}
                <table id="changes_{{ change['column'] }}" class="changeTable" style="display: none; margin-top: 10px;">
                    <tr>
                        <th>Original Value</th>
                        <th>New Value</th>
                        <th>Count</th>
                    </tr>
                    {% for val_change in change['changes'] %}
                        <tr>
                            <td>{{ val_change['original'] }}</td>
                            <td>{{ val_change['new'] }}</td>
                            <td>{{ val_change['count'] }}</td>
                        </tr>
                    {% endfor %}
                </table>
            {% endfor %}
        </div>
    </body>
    </html>
    """)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(template.render(json_data=json_data))
    print(f"Report generated: {output_file}")
    
def generate_report(df1, df2, file1_name, file2_name, description, output_file):
    comparison_results = compare_dataframes(df1, df2, file1_name, file2_name, description)
    #save_comparison_to_json(comparison_results, "comparison_results.json")
    generate_html_report(comparison_results, output_file)