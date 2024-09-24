from generate.metadata import generate_metadata
from generate.syntheticdata import generate_syntheticdata

from evaluate.privacy import evaluate_privacy
from evaluate.quality import evaluate_quality

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
DATA = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned.csv")
DATA, CONTROL_DATA = train_test_split(DATA, test_size=0.1)
PREDICTION_COLUMN = "Combined Depression"
SENSITIVE_COLUMNS = ["Combined Depression"]
KEY_COLUMNS = ["PTDOBYY","PTGENDER"]
ITERATIONS = 10
SAMPLE_SIZE = 1000
EPSILON = 2.56

SYNTHETIC_DATA = generate_syntheticdata(MODEL, DATA, CONTROL_DATA, PREDICTION_COLUMN, SENSITIVE_COLUMNS, KEY_COLUMNS, 
                                        ITERATIONS, SAMPLE_SIZE, EPSILON, None, None)

##
## SYNTHETIC DATA PRIVACY EVALUATION TESTING ##

evaluate_privacy(DATA, SYNTHETIC_DATA, SENSITIVE_COLUMNS, KEY_COLUMNS, CONTROL_DATA)

evaluate_quality(DATA, SYNTHETIC_DATA)