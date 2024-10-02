from synthopt.generate.metadata import generate_metadata
from synthopt.generate.syntheticdata import generate_syntheticdata
from synthopt.generate.syntheticdata import process

from synthopt.evaluate.privacy import evaluate_privacy
from synthopt.evaluate.quality import evaluate_quality
from synthopt.evaluate.utility import evaluate_utility

from synthopt.evaluate.visualisation import table_vis, attribute_vis

from synthopt.evaluate.report import create_pdf_report

from sklearn.model_selection import train_test_split
import pandas as pd
from functools import reduce
import seaborn as sns
from sklearn.impute import KNNImputer

from synthopt.optimise.optimise import optimise_epsilon


## METADATA GENERATION TESTING ##
"""
METADATA_FILENAME = "/workspaces/SynthOpt/examples/example_metadata.csv"
SAMPLE_SIZE = 800
SAVE_LOCATION = "/workspaces/SynthOpt/output"

GENERATED_METADATA_DATASETS = generate_metadata(METADATA_FILENAME, SAMPLE_SIZE, SAVE_LOCATION)
"""
##
## SYNTHETIC DATA GENERATION TESTING (MULTI) ##
"""
TYPE = "multi"
MODEL = "pategan"
DATA1 = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned_subset1.csv")
DATA2 = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned_subset2.csv")
DATA3 = pd.read_csv("/workspaces/SynthOpt/examples/ADNI_cleaned_subset3.csv")
DATA = [DATA1,DATA2,DATA3]
#SUBSET_SIZE = 800
DATA_PROCESSED, CONTROL_DATA = process(DATA, TYPE) # ,SUBSET_SIZE
IDENTIFIER_COLUMN = "RID"
PREDICTION_COLUMN = "Depression"
SENSITIVE_COLUMNS = ["Depression"]
KEY_COLUMNS = ["PTDOBYY","PTGENDER"]
ITERATIONS = 1
SAMPLE_SIZE = 1000
EPSILON = 5
PREDICTION_TYPE = 'binary'


SYNTHETIC_DATA = generate_syntheticdata(DATA_PROCESSED, IDENTIFIER_COLUMN, PREDICTION_COLUMN, SENSITIVE_COLUMNS, 
                                        SAMPLE_SIZE, TYPE, MODEL, ITERATIONS, EPSILON)
SYNTHETIC_DATA[0].to_csv("/workspaces/SynthOpt/output/example_synthetic_data_subset1.csv", index=False)
SYNTHETIC_DATA[1].to_csv("/workspaces/SynthOpt/output/example_synthetic_data_subset2.csv", index=False)
SYNTHETIC_DATA[2].to_csv("/workspaces/SynthOpt/output/example_synthetic_data_subset3.csv", index=False)

##
## SYNTHETIC DATA PRIVACY EVALUATION TESTING ##

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


create_pdf_report(privacy_scores, quality_scores, utility_scores, TYPE, IDENTIFIER_COLUMN, DATA, SYNTHETIC_DATA, DATA_COLUMNS,"/workspaces/SynthOpt/output/EvaluationReport.pdf")

"""


data={
        'bio': pd.read_csv("/workspaces/SynthOpt/examples/bio.csv"),
        'dispat': pd.read_csv("/workspaces/SynthOpt/examples/dispat.csv"),
        'indis': pd.read_csv("/workspaces/SynthOpt/examples/indis.csv"),
        'inf': pd.read_csv("/workspaces/SynthOpt/examples/inf.csv"),
        'rel11': pd.read_csv("/workspaces/SynthOpt/examples/rel11.csv"),
        'rel12': pd.read_csv("/workspaces/SynthOpt/examples/rel12.csv"),
        'rel13': pd.read_csv("/workspaces/SynthOpt/examples/rel13.csv")
    }
from synthopt.generate.syntheticdata import generate_relational_syntheticdata
synthetic_data_dict = generate_relational_syntheticdata(data)

for table_name, synthetic_df in synthetic_data_dict.items():
    # Define the filename (you can adjust the path as needed)
    filename = f"/workspaces/SynthOpt/output/{table_name}_synthetic_data.csv"
    
    # Save the DataFrame to a CSV file
    synthetic_df.to_csv(filename, index=False)
    print(f"Saved {table_name} synthetic data to {filename}")

print(synthetic_data_dict)

# Loop through the real data dictionary to perform evaluations
#for table_name, real_df in data.items():
#    # Check if the table exists in the synthetic data dictionary
#    if table_name in synthetic_data_dict:
#        synthetic_df = synthetic_data_dict[table_name]



"""
# Example weights
weights = {
    'utility': 0.8,
    'privacy': 0.2
}

# Call the optimisation function
optimised_epsilon, optimised_score = optimise_epsilon(
    DATA_PROCESSED, MODEL, TYPE, IDENTIFIER_COLUMN, PREDICTION_COLUMN, PREDICTION_TYPE, SENSITIVE_COLUMNS, KEY_COLUMNS,
    CONTROL_DATA, SAMPLE_SIZE, ITERATIONS, weights
)

print()
print("Optimized Epsilon:", optimised_epsilon)
print("Optimized Score:", optimised_score)
"""