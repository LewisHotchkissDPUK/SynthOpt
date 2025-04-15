import pandas as pd

from synthopt.process.metadata import metadata_process
from synthopt.generate.statistical import structural_data

### Structural Metadata Process Testing
"""
DATA = pd.read_csv("examples/NEW_TESTING_DATASET.csv")
METADATA, LABEL_MAPPING = metadata_process(DATA, "id", "structural") #, CORRELATION_MATRIX, MARGINALS

print(METADATA)
print()
print(LABEL_MAPPING)
"""
###

### Correlated Metadata Process Testing
"""
DATA = pd.read_csv("examples/NEW_TESTING_DATASET.csv")
METADATA, LABEL_MAPPING, CORRELATION_MATRIX, MARGINALS = metadata_process(DATA, "id", "correlated") #, CORRELATION_MATRIX, MARGINALS

print(METADATA)
print()
print(LABEL_MAPPING)
print()
print(CORRELATION_MATRIX)

SYNTHETIC_DATA = structural_data(METADATA, LABEL_MAPPING, len(DATA), identifier_column='id')

print()
print(SYNTHETIC_DATA)

SYNTHETIC_DATA.to_csv("output/NEW_TESTING_DATASET_SYNTHETIC_STRUCTURAL.csv")
"""

### New Structural Metadata Process Testing ###
from synthopt.process.structural_metadata import process_structural_metadata
DATA = pd.read_csv("examples/new_new_testing_dataset.csv") #NEW_TESTING_DATASET.csv
DATA['bool'] = DATA['bool'].astype('bool')
DATA2 = pd.read_csv("examples/new_healthcare_testing_dataset.csv") #healthcare_dataset.csv

DATASETS = {"Test": DATA, "Healthcare": DATA2}
#date_formats = ["%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y", "%H:%M:%S", "%d/%m/%Y %H:%M", "%Y-%m-%d"]
date_formats = None
"""
METADATA = process_structural_metadata(DATASETS, date_formats)
print(METADATA)


### New Structural Generation Testing ###
from synthopt.generate.structural_synthetic_data import generate_structural_synthetic_data
SYNTHETIC_DATA = generate_structural_synthetic_data(METADATA, num_records=1000, identifier_column="id")
print(SYNTHETIC_DATA)

METADATA.to_csv("output/new_structural_test_metadata.csv")
SYNTHETIC_DATA['Test'].to_csv("output/new_structural_test.csv")
SYNTHETIC_DATA['Healthcare'].to_csv("output/new_structural_healthcare.csv")

"""


### New Statistical Metadata Process Testing ###
from synthopt.process.statistical_metadata import process_statistical_metadata
STATS_METADATA = process_statistical_metadata(DATASETS, date_formats, "Test")
print("STATS METADATA")
print(STATS_METADATA)

### New Statistical Generation Testing ###
from synthopt.generate.statistical_synthetic_data import generate_statistical_synthetic_data
STATS_SYNTHETIC_DATA = generate_statistical_synthetic_data(STATS_METADATA, num_records=1000, identifier_column="id")
print(STATS_SYNTHETIC_DATA['Test'])
print(STATS_SYNTHETIC_DATA['Healthcare'])

STATS_METADATA.to_csv("output/NEW_TESTING_STATS_METADATA.csv")
STATS_SYNTHETIC_DATA['Test'].to_csv("output/NEW_TESTING_DATASET_SYNTHETIC_STATS.csv")
STATS_SYNTHETIC_DATA['Healthcare'].to_csv("output/HEALTHCARE_SYNTHETIC_STATS.csv")



"""
from synthopt.process.statistical_metadata import process_statistical_metadata
data = pd.read_csv("examples/california_housing_test.csv")
STATS_METADATA = process_statistical_metadata(data)
STATS_SYNTHETIC_DATA = generate_statistical_synthetic_data(STATS_METADATA, num_records=1000)
print(STATS_SYNTHETIC_DATA)
"""


### New Correlation Generation Testing ###
#from synthopt.generate.correlated_synthetic_data import generate_correlated_synthetic_data

print()
print()
### New Statistical Metadata Process Testing ###
from synthopt.process.statistical_metadata import process_statistical_metadata
STATS_METADATA, CORR_MATRIX = process_statistical_metadata(DATASETS, date_formats, return_correlations=True)
print(STATS_METADATA)
print()
print(CORR_MATRIX)
#CORR_MATRIX['Healthcare'].to_csv("output/NEW_HEALTHCARE_TESTING_CORR_MATRIX.csv")

#numerical_metadata = STATS_METADATA[~STATS_METADATA['datatype'].isin(['string', 'object'])]
#variable_names = numerical_metadata['variable_name'].tolist()
#correlation_matrix = DATA[variable_names].corr().values.tolist()

### New Statistical Generation Testing ###
from synthopt.generate.correlated_synthetic_data import generate_correlated_synthetic_data
CORR_SYNTHETIC_DATA = generate_correlated_synthetic_data(STATS_METADATA, CORR_MATRIX, num_records=1000, identifier_column="id")
print(CORR_SYNTHETIC_DATA)


STATS_METADATA.to_csv("output/NEW_TESTING_CORR_METADATA.csv")
CORR_SYNTHETIC_DATA['Test'].to_csv("output/NEW_TESTING_DATASET_SYNTHETIC_CORR.csv")
CORR_SYNTHETIC_DATA['Healthcare'].to_csv("output/HEALTHCARE_SYNTHETIC_CORR.csv")


from synthopt.evaluate.quality2 import evaluate_quality

qs = evaluate_quality(DATASETS, CORR_SYNTHETIC_DATA, STATS_METADATA, identifier_column="id", table_type = 'multi')
print(qs)