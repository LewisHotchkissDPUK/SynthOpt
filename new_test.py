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
DATA = pd.read_csv("examples/NEW_TESTING_DATASET.csv")
DATA['bool'] = DATA['bool'].astype('bool')
DATA2 = pd.read_csv("examples/healthcare_dataset.csv")

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

SYNTHETIC_DATA['Test'].to_csv("output/new_structural_test.csv")
SYNTHETIC_DATA['Healthcare'].to_csv("output/new_structural_healthcare.csv")

"""

### New Statistical Metadata Process Testing ###
from synthopt.process.statistical_metadata import process_statistical_metadata
STATS_METADATA = process_statistical_metadata(DATASETS, date_formats, "Test")
print("STATS METADATA")
print(STATS_METADATA)
#STATS_METADATA['Test'].to_csv("output/NEW_TESTING_DATASET_STATS_METADATA.csv")
#STATS_METADATA['Healthcare'].to_csv("output/HEALTHCARE_STATS_METADATA.csv")

### New Statistical Generation Testing ###
from synthopt.generate.statistical_synthetic_data import generate_statistical_synthetic_data
STATS_SYNTHETIC_DATA = generate_statistical_synthetic_data(STATS_METADATA, num_records=1000, identifier_column="id")
print(STATS_SYNTHETIC_DATA['Test'])
print(STATS_SYNTHETIC_DATA['Healthcare'])
#STATS_SYNTHETIC_DATA['Test'].to_csv("output/NEW_TESTING_DATASET_SYNTHETIC_STATS.csv")
#STATS_SYNTHETIC_DATA['Healthcare'].to_csv("output/HEALTHCARE_SYNTHETIC_STATS.csv")


"""
from synthopt.process.statistical_metadata import process_statistical_metadata
data = pd.read_csv("examples/california_housing_test.csv")
STATS_METADATA = process_statistical_metadata(data)
print("STATS METADATA")
print(STATS_METADATA)
"""