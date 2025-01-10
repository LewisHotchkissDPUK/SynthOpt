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