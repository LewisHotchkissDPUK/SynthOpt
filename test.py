from generate.metadata import generate_metadata
from generate.syntheticdata import generate_syntheticdata

from evaluate.privacy import evaluate_privacy

from sklearn.model_selection import train_test_split
import pandas as pd

## METADATA GENERATION TESTING ##
"""
METADATA_FILENAME = "/workspaces/SynthOpt/examples/Cam-Can_Metadata.csv"
SAMPLE_SIZE = 800
SAVE_LOCATION = "/workspaces/SynthOpt/examples"

GENERATED_METADATA_DATASETS = generate_metadata(METADATA_FILENAME, SAMPLE_SIZE, SAVE_LOCATION)

for filename, df in GENERATED_METADATA_DATASETS.items():
    print(f"\nData for {filename}:")
    print(df.head())
"""
##
## SYNTHETIC DATA GENERATION TESTING ##

MODEL = "dpgan"
DATA_NAME = "/workspaces/SynthOpt/examples/ADNI_cleaned.csv"
PREDICTION_COLUMN = "Combined Depression"
SENSITIVE_COLUMNS = ["Combined Depression"]
KEY_COLUMNS = ["PTDOBYY","PTGENDER"]
ITERATIONS = 10
SAMPLE_SIZE = 1000
EPSILON = 2.56

SYNTHETIC_DATA = generate_syntheticdata(MODEL, DATA_NAME, PREDICTION_COLUMN, SENSITIVE_COLUMNS, KEY_COLUMNS, 
                                        ITERATIONS, SAMPLE_SIZE, EPSILON, None, None)

##
## SYNTHETIC DATA PRIVACY EVALUATION TESTING ##

DATA = pd.read_csv(DATA_NAME)
DATA, CONTROL_DATA = train_test_split(DATA, test_size=0.1, random_state=42) #random state is 42 in syntheticdata module

evaluate_privacy(DATA, SYNTHETIC_DATA, SENSITIVE_COLUMNS, KEY_COLUMNS, CONTROL_DATA)