from generate.metadata import generate_metadata
from generate.syntheticdata import generate_syntheticdata

from evaluate.privacy import evaluate_privacy
from evaluate.quality import evaluate_quality
from evaluate.utility import evaluate_utility

from evaluate.visualisation import table_vis

from sklearn.model_selection import train_test_split
import pandas as pd

## METADATA GENERATION TESTING ##
"""
METADATA_FILENAME = "/workspaces/SynthOpt/examples/Cam-Can_Metadata.csv"
SAMPLE_SIZE = 800
SAVE_LOCATION = "/workspaces/SynthOpt/output"

GENERATED_METADATA_DATASETS = generate_metadata(METADATA_FILENAME, SAMPLE_SIZE, SAVE_LOCATION)

for filename, df in GENERATED_METADATA_DATASETS.items():
    print(f"\nData for {filename}:")
    print(df.head())
"""
##
## SYNTHETIC DATA GENERATION TESTING ##

TYPE = "single" #multi, temporal
MODEL = "pategan"
DATA = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned.csv")
DATA, CONTROL_DATA = train_test_split(DATA, test_size=0.1)
IDENTIFIER_COLUMN = "RID"
PREDICTION_COLUMN = "Combined Depression"
SENSITIVE_COLUMNS = ["Combined Depression"]
KEY_COLUMNS = ["PTDOBYY","PTGENDER"]
ITERATIONS = 10
SAMPLE_SIZE = 800
EPSILON = 5

SYNTHETIC_DATA = generate_syntheticdata(TYPE, MODEL, DATA, IDENTIFIER_COLUMN, PREDICTION_COLUMN, SENSITIVE_COLUMNS, 
                                        ITERATIONS, SAMPLE_SIZE, EPSILON, None, None)

SYNTHETIC_DATA.to_csv("/workspaces/SynthOpt/output/example_synthetic_data.csv")

##
## SYNTHETIC DATA PRIVACY EVALUATION TESTING ##

# maybe add a risk appetite level for determining how many attacks to run etc and thresholds for evaluations

privacy_scores = evaluate_privacy(DATA, SYNTHETIC_DATA, SENSITIVE_COLUMNS, KEY_COLUMNS, CONTROL_DATA)
quality_scores = evaluate_quality(DATA, SYNTHETIC_DATA)
utility_scores = evaluate_utility(DATA, SYNTHETIC_DATA, PREDICTION_COLUMN)

table_vis(privacy_scores, quality_scores, utility_scores, DATA.columns)
