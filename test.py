from generate.metadata import generate_metadata
from generate.syntheticdata import generate_syntheticdata

from evaluate.privacy import evaluate_privacy
from evaluate.quality import evaluate_quality
from evaluate.utility import evaluate_utility

from evaluate.visualisation import table_vis, attribute_vis

from evaluate.report import create_pdf_report

from sklearn.model_selection import train_test_split
import pandas as pd
from functools import reduce


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
## SYNTHETIC DATA GENERATION TESTING (SINGLE) ##
"""
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
PREDICTION_TYPE = 'binary' #multi, regression

SYNTHETIC_DATA = generate_syntheticdata(TYPE, MODEL, DATA, IDENTIFIER_COLUMN, PREDICTION_COLUMN, SENSITIVE_COLUMNS, 
                                        ITERATIONS, SAMPLE_SIZE, EPSILON, None, None)
SYNTHETIC_DATA.to_csv("/workspaces/SynthOpt/output/example_synthetic_data.csv", index=False)
"""
##
## SYNTHETIC DATA GENERATION TESTING (MULTI) ##

TYPE = "multi"
MODEL = "pategan"
DATA1 = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned_subset1.csv")
DATA1, CONTROL_DATA1 = train_test_split(DATA1, test_size=0.1, random_state=42)
DATA2 = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned_subset2.csv")
DATA2, CONTROL_DATA2 = train_test_split(DATA2, test_size=0.1, random_state=42)
DATA3 = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned_subset3.csv")
DATA3, CONTROL_DATA3 = train_test_split(DATA3, test_size=0.1, random_state=42)
DATA = [DATA1,DATA2,DATA3]
CONTROL_DATA = [CONTROL_DATA1,CONTROL_DATA2,CONTROL_DATA3]
IDENTIFIER_COLUMN = "RID"
PREDICTION_COLUMN = "Combined Depression"
SENSITIVE_COLUMNS = ["Combined Depression"]
KEY_COLUMNS = ["PTDOBYY","PTGENDER"]
ITERATIONS = 1
SAMPLE_SIZE = 800
EPSILON = 5
PREDICTION_TYPE = 'binary'

SYNTHETIC_DATA = generate_syntheticdata(TYPE, MODEL, DATA, IDENTIFIER_COLUMN, PREDICTION_COLUMN, SENSITIVE_COLUMNS, 
                                        ITERATIONS, SAMPLE_SIZE, EPSILON, None, None)
SYNTHETIC_DATA[0].to_csv("/workspaces/SynthOpt/output/example_synthetic_data_subset1.csv", index=False)
SYNTHETIC_DATA[1].to_csv("/workspaces/SynthOpt/output/example_synthetic_data_subset2.csv", index=False)
SYNTHETIC_DATA[2].to_csv("/workspaces/SynthOpt/output/example_synthetic_data_subset3.csv", index=False)


##
## SYNTHETIC DATA PRIVACY EVALUATION TESTING ##

# maybe add a risk appetite level for determining how many attacks to run etc and thresholds for evaluations

privacy_scores = evaluate_privacy(TYPE, DATA, SYNTHETIC_DATA, IDENTIFIER_COLUMN, SENSITIVE_COLUMNS, KEY_COLUMNS, CONTROL_DATA)
quality_scores = evaluate_quality(TYPE, DATA, SYNTHETIC_DATA, IDENTIFIER_COLUMN)
utility_scores = evaluate_utility(TYPE, DATA, SYNTHETIC_DATA, IDENTIFIER_COLUMN, PREDICTION_TYPE, PREDICTION_COLUMN)

table_vis(privacy_scores, quality_scores, utility_scores)

vis_data = reduce(lambda left, right: pd.merge(left, right, on=IDENTIFIER_COLUMN), DATA)
vis_data = vis_data.drop(columns=[IDENTIFIER_COLUMN])

DATA_COLUMNS = vis_data.columns

attribute_vis(privacy_scores, quality_scores, utility_scores, DATA_COLUMNS) # maybe pass in data instead of columns to handle the identifier column and multi 

create_pdf_report(privacy_scores, quality_scores, utility_scores, DATA_COLUMNS)