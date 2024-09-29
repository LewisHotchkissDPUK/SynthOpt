from generate.metadata import generate_metadata
from generate.syntheticdata import generate_syntheticdata
from generate.syntheticdata import process

from evaluate.privacy import evaluate_privacy
from evaluate.quality import evaluate_quality
from evaluate.utility import evaluate_utility

from evaluate.visualisation import table_vis, attribute_vis

from evaluate.report import create_pdf_report

from sklearn.model_selection import train_test_split
import pandas as pd
from functools import reduce
import seaborn as sns
from sklearn.impute import KNNImputer

from optimise.optimise import optimize_epsilon


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
DATA2 = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned_subset2.csv")
DATA3 = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned_subset3.csv")
DATA = [DATA1,DATA2,DATA3]
#SUBSET_SIZE = 800
DATA_PROCESSED, CONTROL_DATA = process(DATA, TYPE) # ,SUBSET_SIZE
IDENTIFIER_COLUMN = "RID"
PREDICTION_COLUMN = "Combined Depression"
SENSITIVE_COLUMNS = ["Combined Depression"]
KEY_COLUMNS = ["PTDOBYY","PTGENDER"]
ITERATIONS = 1
SAMPLE_SIZE = 800
EPSILON = 5
PREDICTION_TYPE = 'binary'



SYNTHETIC_DATA = generate_syntheticdata(DATA_PROCESSED, IDENTIFIER_COLUMN, PREDICTION_COLUMN, SENSITIVE_COLUMNS, 
                                        SAMPLE_SIZE, TYPE, MODEL, ITERATIONS, EPSILON)
SYNTHETIC_DATA[0].to_csv("/workspaces/SynthOpt/output/example_synthetic_data_subset1.csv", index=False)
SYNTHETIC_DATA[1].to_csv("/workspaces/SynthOpt/output/example_synthetic_data_subset2.csv", index=False)
SYNTHETIC_DATA[2].to_csv("/workspaces/SynthOpt/output/example_synthetic_data_subset3.csv", index=False)

##
## SYNTHETIC DATA PRIVACY EVALUATION TESTING ##

# maybe add a risk appetite level for determining how many attacks to run etc and thresholds for evaluations

privacy_scores = evaluate_privacy(DATA, SYNTHETIC_DATA, IDENTIFIER_COLUMN, SENSITIVE_COLUMNS, KEY_COLUMNS, CONTROL_DATA, TYPE)
quality_scores = evaluate_quality(DATA, SYNTHETIC_DATA, IDENTIFIER_COLUMN, TYPE)
utility_scores = evaluate_utility(DATA, SYNTHETIC_DATA, CONTROL_DATA, IDENTIFIER_COLUMN, PREDICTION_COLUMN, TYPE, PREDICTION_TYPE)

table_vis(privacy_scores, quality_scores, utility_scores)

# because multi
vis_data = reduce(lambda left, right: pd.merge(left, right, on=IDENTIFIER_COLUMN), DATA)
vis_data = vis_data.drop(columns=[IDENTIFIER_COLUMN])
DATA_COLUMNS = vis_data.columns
DATA_COLUMNS = list(DATA_COLUMNS)
#

create_pdf_report(privacy_scores, quality_scores, utility_scores, TYPE, IDENTIFIER_COLUMN, DATA, SYNTHETIC_DATA, DATA_COLUMNS)




"""
# Example weights
weights = {
    'quality': 0.5,
    'utility': 0.3,
    'privacy': 0.2
}

# Call the optimization function
optimized_epsilon, optimized_score = optimize_epsilon(
    DATA_PROCESSED, MODEL, TYPE, IDENTIFIER_COLUMN, PREDICTION_COLUMN, PREDICTION_TYPE, SENSITIVE_COLUMNS, KEY_COLUMNS,
    CONTROL_DATA, SAMPLE_SIZE, ITERATIONS, weights
)

print("Optimized Epsilon:", optimized_epsilon)
print("Optimized Score:", optimized_score)
"""