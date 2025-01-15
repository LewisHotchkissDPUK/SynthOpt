import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta
import os
from scipy.stats import truncnorm
from scipy.stats import skew
from scipy.stats import skewnorm, multivariate_normal
from numpy.linalg import cholesky
from sklearn.preprocessing import LabelEncoder
import random
import string
from datetime import datetime
import calendar
import warnings
warnings.filterwarnings('ignore')
from statsmodels.distributions.copula.api import GaussianCopula, CopulaDistribution
from scipy.stats import norm
import scipy.stats as stats
from tqdm import tqdm


def calculate_average_length(df, columns):
  results = []
  for column in columns:
    char_lengths = []
    space_lengths = []
    for value in df[column]:
      if isinstance(value, str):
        char_lengths.append(len(value))
        space_lengths.append(value.count(" "))
    avg_char_length = sum(char_lengths) / len(char_lengths) if char_lengths else 0
    avg_space_length = sum(space_lengths) / len(space_lengths) if space_lengths else 0

    results.append({
        "column": column,
        "avg_char_length": avg_char_length,
        "avg_space_length": avg_space_length,
    })
  return results

def best_fit_distribution(data, bins=200):
    DISTRIBUTIONS = [
        stats.norm, stats.expon, stats.lognorm, stats.gamma,
        stats.beta, stats.uniform, stats.weibull_min, stats.poisson
    ]

    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    best_distribution = None
    best_params = None
    best_sse = np.inf

    for distribution in DISTRIBUTIONS:
        try:
            params = distribution.fit(data)
            pdf = distribution.pdf(bin_centers, *params)
            sse = np.sum(np.power(hist - pdf, 2.0))

            if best_sse > sse > 0:
                best_distribution = distribution
                best_params = params
                best_sse = sse
        except Exception:
            pass

    return best_distribution, best_params

def identify_best_fit_distributions(df, discrete_threshold=10): # change this discrete identification
    #df = df.fillna(df.mean())
    
    result = {}

    for column in df.columns:
        data = df[column].dropna()
        data = data[~data.isin([np.inf, -np.inf])]

        if data.nunique() <= discrete_threshold:
            try:
                mu = data.mean()
                result[column] = (stats.poisson, (mu,))
            except Exception:
                result[column] = (None, None)
        else:
            best_distribution, best_params = best_fit_distribution(data)
            result[column] = (best_distribution, best_params)

    return result





def metadata_process(data, identifier_column=None, type="correlated"):

    def process_single_dataframe(data, table_name=None):
        
        orig_data_completeness = data.copy()
        data = data.copy()
        metadata = pd.DataFrame(columns=['variable_name', 'datatype', 'completeness', 'values', 'mean', 'standard_deviation', 'table_name'])

        non_numerical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
        #for column in non_numerical_columns:
        for column in tqdm(non_numerical_columns, desc="Processing Non Numerical Columns"):
            # IF THE OBJECT CANT BE CONVERTED TO NUMBER
            try:
                data[column] = pd.to_numeric(data[column], errors='raise')
                non_numerical_columns = non_numerical_columns.remove(column)

                # If all values are integers, convert to Int64 or int
                if data[column].apply(float.is_integer).all():
                    data[column] = data[column].astype('Int64')  # Uses Int64 for nullable support
            except:
                None
        if non_numerical_columns == None:
            non_numerical_columns = []
            
        # Convert floats that are actually integers
        #for column in data.select_dtypes(include='float'):
        for column in tqdm(data.select_dtypes(include='float'), desc="Processing Integer Columns"):
            if (data[column].dropna() % 1 == 0).all():
                data[column] = data[column].astype("Int64")
                if data[column].notna().any():
                    data[column] = data[column].fillna(round(data[column].mean()))

        # fill na of numerical columns with mean
        float_columns = data.select_dtypes(include=['float']).columns
        data[float_columns] = data[float_columns].fillna(data[float_columns].mean())
        
        

        
        
        
        ###########################################################################################
        non_numerical_columns = data.select_dtypes(exclude=['number']).columns.tolist()
        
        date_formats = [
            "%b-%m","%b-%y","%b-%Y",
            "%B-%m","%B-%y","%B-%Y",
            "%d-%b-%y","%d-%b-%Y","%b-%d-%y","%b-%d-%Y",
            "%d-%B-%y","%d-%B-%Y","%B-%d-%y","%B-%d-%Y",
            "%y-%m-%d","%Y-%m-%d","%m-%d-%y","%m-%d-%Y","%d-%m-%y","%d-%m-%Y",
            "%b/%m","%b/%y","%b/%Y",
            "%B/%m","%B/%y","%B/%Y",
            "%d/%b/%y","%d/%b/%Y","%b/%d/%y","%b/%d/%Y",
            "%d/%B/%y","%d/%B/%Y","%B/%d/%y","%B/%d/%Y",
            "%y/%m/%d","%Y/%m/%d","%m/%d/%y","%m/%d/%Y","%d/%m/%y","%d/%m/%Y",
            "%b.%m","%b.%y","%b.%Y",
            "%B.%m","%B.%y","%B.%Y",
            "%d.%b.%y","%d.%b.%Y","%b.%d.%y","%b.%d.%Y",
            "%d.%B.%y","%d.%B.%Y","%B.%d.%y","%B.%d.%Y",
            "%y.%m.%d","%Y.%m.%d","%m.%d.%y","%m.%d.%Y","%d.%m.%y","%d.%m.%Y",
        ]

        time_formats = ["%H:%M:%S.%f", "%H:%M:%S", "%H:%M"]




        # NEW DATE TIME, check if values contain both date and time. If they do then split out.
        from dateutil.parser import parse

        # Sample function to check if a column contains both date and time
        def contains_date_and_time(series):
            for value in series.dropna():
                try:
                    # Attempt to parse the value
                    dt = parse(value, fuzzy=True)
                    # Ensure both date and time are present
                    if dt.date() and (dt.time() != dt.min.time()) and (dt.date() != datetime.today().date()):  # Time is not midnight (00:00:00) and date is not current date
                        return True
                except (ValueError, TypeError):
                    continue
            return False

        # Function to identify and convert columns containing date and time
        def identify_datetime_columns(df, non_numerical_columns):
            #df = df.copy()
            #for column in non_numerical_columns:
            for column in tqdm(non_numerical_columns, desc="Processing Date + Time Columns"):
                try:
                    if contains_date_and_time(df[column].astype(str)):
                        # split into date and time
                        #print(df[column])
                        df[column] = pd.to_datetime(df[column], errors='coerce', infer_datetime_format=True)        # This needs fixing as its converting some values to none
                        #print(df[column])
                        date_column = df[column].dt.date
                        time_column = df[column].dt.time
                        df.insert(df.columns.get_loc(column), f'{column}_synthoptdate', date_column)
                        df.insert(df.columns.get_loc(column) + 1, f'{column}_synthopttime', time_column)
                        df.drop(column, axis=1, inplace=True)
                        #print(f"Column '{column}' was split into '{column}_date' and '{column}_time'.")

                        # Update the non_numerical_columns list
                        if column in non_numerical_columns:
                            non_numerical_columns.remove(column)
                        non_numerical_columns.insert(0, f'{column}_synthoptdate')
                        non_numerical_columns.insert(0, f'{column}_synthopttime')
                        #break  # Exit after processing the column
                except Exception:
                    None

            return df
        
        data = identify_datetime_columns(data, non_numerical_columns)


        ######




        date_columns = []

        #for column in non_numerical_columns:
        for column in tqdm(non_numerical_columns, desc="Processing Date Columns"):
            for date_format in date_formats:
                try:
                    converted_column = pd.to_datetime(data[column], format=date_format)
                    if converted_column.notna().any() and converted_column.dt.date.nunique != 1:
                        date_columns.append(column)
                        data[column] = pd.to_datetime(data[column], format=date_format)
                        break
                except ValueError:
                    continue
                                        
        # go through non_numerical_columns which are not in date_columns and check for time formats
        
        time_columns = []
        #for column in non_numerical_columns:
        for column in tqdm(non_numerical_columns, desc="Processing Time Columns"):
            if column not in date_columns:
                try:
                    pd.to_datetime(data[column], format="%H:%M:%S") #, errors='coerce'
                    if pd.to_datetime(data[column], format="%H:%M:%S").notna().sum() != 0:
                        time_columns.append(column)
                        data[column] = pd.to_datetime(data[column], format="%H:%M:%S", errors="coerce")
                except Exception:
                    try:
                        pd.to_datetime(data[column], format="%H:%M") #, errors='coerce'
                        if pd.to_datetime(data[column], format="%H:%M").notna().sum() != 0:
                            time_columns.append(column)
                            data[column] = pd.to_datetime(data[column], format="%H:%M", errors="coerce")
                    except Exception:
                        try:
                            pd.to_datetime(data[column], format="%H:%M:%S.%f", errors="coerce") #, errors='coerce'
                            if pd.to_datetime(data[column], format="%H:%M:%S.%f", errors="coerce").notna().sum() != 0:
                                time_columns.append(column)
                                data[column] = pd.to_datetime(data[column], format="%H:%M:%S.%f", errors="coerce")
                        except Exception:
                            pass
                            
        ###########################################################################################
        
        
        # add code to try convert strings to numbers

        # Identify string/object columns
        all_string_columns = list(set(non_numerical_columns) - set(date_columns))
        all_string_columns = list(set(all_string_columns) - set(time_columns))

        # identify string categories
        categorical_string_columns = []
        #for column in data[all_string_columns].columns:
        for column in tqdm(data[all_string_columns].columns, desc="Processing String Columns"):
            if (data[all_string_columns][column].nunique() < len(data[all_string_columns]) * 0.2) and ((data[all_string_columns][column].value_counts() >= 2).sum() >= (0.6 * len(data[all_string_columns][column].value_counts()))):
                if data[all_string_columns][column].nunique() != len(data[all_string_columns][column]):
                    categorical_string_columns.append(column)
        non_categorical_string_columns = list(set(all_string_columns) - set(categorical_string_columns))
        
        # Calculate average lengths of non-categorical strings
        average_lengths_df = calculate_average_length(data, non_categorical_string_columns)
        # Encode categorical strings
        orig_data = data.copy()
        le = LabelEncoder()
        #for column in categorical_string_columns:
        for column in tqdm(categorical_string_columns, desc="Processing Categorical String Columns"):
            data[column] = data[column].astype(str)
            data[column] = le.fit_transform(data[column])

        
        
        ###########################################################################################
        #for column in date_columns:
        for column in tqdm(date_columns, desc="Splitting Date Columns"):
            data[column + '_synthoptyear'] = data[column].dt.year
            if data[column + '_synthoptyear'].notna().any():
                orig_data_completeness[column + '_synthoptyear'] = data[column + '_synthoptyear']
                data[column + '_synthoptyear'] = data[column + '_synthoptyear'].fillna(round(data[column + '_synthoptyear'].min()))
                data[column + '_synthoptyear'] = data[column + '_synthoptyear'].astype('Int64')

            data[column + '_synthoptmonth'] = data[column].dt.month
            if data[column + '_synthoptmonth'].notna().any():
                orig_data_completeness[column + '_synthoptmonth'] = data[column + '_synthoptmonth']
                data[column + '_synthoptmonth'] = data[column + '_synthoptmonth'].fillna(round(data[column + '_synthoptmonth'].min()))
                data[column + '_synthoptmonth'] = data[column + '_synthoptmonth'].astype('Int64')

            data[column + '_synthoptday'] = data[column].dt.day
            if data[column + '_synthoptday'].notna().any():
                orig_data_completeness[column + '_synthoptday'] = data[column + '_synthoptday']
                data[column + '_synthoptday'] = data[column + '_synthoptday'].fillna(round(data[column + '_synthoptday'].min()))
                data[column + '_synthoptday'] = data[column + '_synthoptday'].astype('Int64')

            data.insert(data.columns.get_loc(column) + 1, column + '_synthoptyear', data.pop(column + '_synthoptyear'))
            data.insert(data.columns.get_loc(column) + 2, column + '_synthoptmonth', data.pop(column + '_synthoptmonth'))
            data.insert(data.columns.get_loc(column) + 3, column + '_synthoptday', data.pop(column + '_synthoptday'))

            data = data.drop(columns=[column], axis=1)
        ###########################################################################################


        ###########################################################################################
        #for column in time_columns:
        for column in tqdm(time_columns, desc="Splitting Time Columns"):
            data[column + '_synthopthour'] = data[column].dt.hour
            if data[column + '_synthopthour'].notna().any():
                orig_data_completeness[column + '_synthopthour'] = data[column + '_synthopthour']
                data[column + '_synthopthour'] = data[column + '_synthopthour'].fillna(round(data[column + '_synthopthour'].min()))
                data[column + '_synthopthour'] = data[column + '_synthopthour'].astype('Int64')

            data[column + '_synthoptminute'] = data[column].dt.minute
            if data[column + '_synthoptminute'].notna().any():
                orig_data_completeness[column + '_synthoptminute'] = data[column + '_synthoptminute']
                data[column + '_synthoptminute'] = data[column + '_synthoptminute'].fillna(round(data[column + '_synthoptminute'].min()))
                data[column + '_synthoptminute'] = data[column + '_synthoptminute'].astype('Int64')

            data[column + '_synthoptsecond'] = data[column].dt.second
            if data[column + '_synthoptsecond'].notna().any():
                orig_data_completeness[column + '_synthoptsecond'] = data[column + '_synthoptsecond']
                data[column + '_synthoptsecond'] = data[column + '_synthoptsecond'].fillna(round(data[column + '_synthoptsecond'].min()))
                data[column + '_synthoptsecond'] = data[column + '_synthoptsecond'].astype('Int64')

            data.insert(data.columns.get_loc(column) + 1, column + '_synthopthour', data.pop(column + '_synthopthour'))
            data.insert(data.columns.get_loc(column) + 2, column + '_synthoptminute', data.pop(column + '_synthoptminute'))
            data.insert(data.columns.get_loc(column) + 3, column + '_synthoptsecond', data.pop(column + '_synthoptsecond'))

            data = data.drop(columns=[column], axis=1)
        ###########################################################################################
            
            
            
            
            
        
        #for column in data.columns:
        for column in tqdm(data.columns, desc="Handling Completeness"):
            completeness = (orig_data_completeness[column].notna().sum() / len(data)) * 100
            
            if column in non_categorical_string_columns: #or column in non_numerical_columns
                value_range = None
                mean = next((item['avg_char_length'] for item in average_lengths_df if item['column'] == column), None)
                std_dev = next((item['avg_space_length'] for item in average_lengths_df if item['column'] == column), None)
            else:
                try:
                    # identify if numbers are categorical, if so return list, otherwise return tuple   (should be something like < data[column].notna().sum() * 0.3)) used to be (data[column].nunique() < len(data) * 0.3)
                    # if the number of unique values is less than 20% of non missing values AND if atleast 60% of the data has more than two counts AND completeness is not 0 and there are atleast two unique values AND the number of unique values are not the same length as the whole data
                    if ((data[column].nunique() < data[column].notna().sum() * 0.1) \
                        and ((data[column].value_counts() >= 2).sum() >= (0.7 * data[column].nunique())) \
                        and (completeness != 0) \
                        and (data[column].nunique() >= 2) \
                        and (data[column].nunique() != len(data[column])) \
                        and (data[column].nunique() < 50)): # and (data[column].nunique() < 30)
                        
                        
                        value_range = data[column].unique().tolist()
                    else:
                        value_range = (data[column].min(), data[column].max())
                except Exception:
                    value_range = None
                try:
                    mean = data[column].mean()
                    std_dev = data[column].std()
                except Exception:
                    mean = None
                    std_dev = None
                
            new_row = pd.DataFrame({
                'variable_name': [column],
                'datatype': [data[column].dtype],
                'completeness': [completeness],
                'values': [value_range],
                'mean': [mean],
                'standard_deviation': [std_dev],
                'table_name': [table_name] if table_name else [None]
            })
            metadata = pd.concat([metadata, new_row], ignore_index=True)

        # Create label mapping for categorical variables with table name prefix
        label_mapping = {}
        #for column in categorical_string_columns:
        for column in tqdm(categorical_string_columns, desc="Handling Label Mapping"):
            prefixed_column = f"{table_name}.{column}" if table_name else column  # Add table name prefix
            orig_data[column] = orig_data[column].astype(str)
            label_mapping[prefixed_column] = dict(zip(le.fit_transform(orig_data[column].unique()), orig_data[column].unique()))

        return metadata, label_mapping, data




    # If the input is a dictionary, process each table individually
    if isinstance(data, dict):
        combined_metadata = pd.DataFrame()
        combined_label_mapping = {}
        combined_data = pd.DataFrame()

        #for table_name, df in data.items():
        for table_name, df in tqdm(data.items(), desc="Processing Tables"):
            table_metadata, table_label_mapping, processed_data = process_single_dataframe(df, table_name)
            combined_metadata = pd.concat([combined_metadata, table_metadata], ignore_index=True)
            
            # Update with the new label mapping, flattening it
            for key, value in table_label_mapping.items():
                combined_label_mapping[key] = value  # Add the prefixed key directly

            # Prefix columns with table name to prevent conflicts
            processed_data.columns = [f"{table_name}.{col}" for col in processed_data.columns]

            processed_data.columns = [identifier_column if identifier_column in col else col for col in processed_data.columns] # remove prefix for ID
            try:
                combined_data = pd.merge(combined_data, processed_data, on=identifier_column, how='outer')
            except:
                combined_data = pd.concat([combined_data, processed_data], axis=1)
                
        # Correlation across combined numerical data
        combined_numerical_data = combined_data.select_dtypes(include=['number'])
        combined_numerical_data = combined_numerical_data.dropna(axis=1)
        combined_numerical_data = combined_numerical_data.loc[:, combined_numerical_data.nunique() > 1]
        
        if type == "correlated":
            correlation_matrix = np.corrcoef(combined_numerical_data.astype(float).values, rowvar=False)
            correlation_matrix = pd.DataFrame(correlation_matrix, index=combined_numerical_data.columns, columns=combined_numerical_data.columns)
            
            best_fit_distributions = identify_best_fit_distributions(combined_numerical_data)
            marginals = []
            for column in combined_numerical_data.columns:
                dist, params = best_fit_distributions[column]
                if dist and params:
                    marginals.append(dist(*params))
                else:
                    marginals.append(norm(loc=np.mean(combined_numerical_data[column]), scale=np.std(combined_numerical_data[column])))

        if type == "correlated":
            return combined_metadata, combined_label_mapping, correlation_matrix, marginals
        elif type == "structural":
            return combined_metadata[['variable_name', 'datatype', 'completeness', 'values', 'table_name']], combined_label_mapping
        else:
            return combined_metadata, combined_label_mapping
    else:
        # If input is a single DataFrame, process directly
        metadata, label_mapping, processed_data = process_single_dataframe(data)

        numerical_data = processed_data.select_dtypes(include=['number'])

        # needed so it matches the removal of empty and single values in generation method
        numerical_data = numerical_data.dropna(axis=1)
        numerical_data = numerical_data.loc[:, numerical_data.nunique() > 1]
        #numerical_data = numerical_data.loc[:, numerical_data.mean() != 0]
        

        if type == "correlated":
            correlation_matrix = np.corrcoef(numerical_data.astype(float).values, rowvar=False)
            correlation_matrix = pd.DataFrame(correlation_matrix, index=numerical_data.columns, columns=numerical_data.columns)
            
            best_fit_distributions = identify_best_fit_distributions(numerical_data)
            marginals = []
            for column in numerical_data.columns:
                dist, params = best_fit_distributions[column]
                if dist and params:
                    marginals.append(dist(*params))
                else:
                    marginals.append(norm(loc=np.mean(numerical_data[column]), scale=np.std(numerical_data[column])))
            #correlation_matrix = processed_data.corr() if type == "correlated" else None

        # if statistical or structural then only return metadata with columns needed (metadata[columns])

        if type == "correlated":
            return metadata, label_mapping, correlation_matrix, marginals
        elif type == "structural":
            return metadata[['variable_name', 'datatype', 'completeness', 'values', 'table_name']], label_mapping
        else:
            return metadata, label_mapping