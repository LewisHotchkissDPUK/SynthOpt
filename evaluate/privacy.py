import pandas as pd
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.utils.serialization import load, load_from_file, save, save_to_file
from sklearn.impute import KNNImputer
from sdmetrics.single_column import BoundaryAdherence,CategoryAdherence,KSComplement,TVComplement,StatisticSimilarity,RangeCoverage,CategoryCoverage
from sdmetrics.column_pairs import CorrelationSimilarity,ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis,LogisticDetection,BinaryDecisionTreeClassifier,CategoricalCAP,CategoricalKNN,NumericalMLP
from sdv.metadata import SingleTableMetadata
from itertools import combinations
from anonymeter.evaluators import SinglingOutEvaluator,LinkabilityEvaluator,InferenceEvaluator
from sklearn.model_selection import train_test_split
def evaluate_privacy(DATA, DATA_COLUMNS, SYNTHETIC_DATA, METADATA, PREDICTION_COLUMN, SENSITIVE_COLUMNS, KEY_COLUMNS, CONTROL_DATA):
    #== Exact Matches ==#
    exact_matches_score = NewRowSynthesis.compute(real_data=DATA, synthetic_data=SYNTHETIC_DATA, metadata=METADATA, numerical_match_tolerance=0.1, synthetic_sample_size=5000)

    #== Detection ==#        
    detection_score = LogisticDetection.compute(real_data=DATA, synthetic_data=SYNTHETIC_DATA, metadata=METADATA)

    #== Inference Attack Protection ==#        
    inference_protection_score = CategoricalCAP.compute(real_data=DATA, synthetic_data=SYNTHETIC_DATA, key_fields=KEY_COLUMNS, sensitive_fields=SENSITIVE_COLUMNS)

    #== Singling Out ==#    
    singling_evaluator = SinglingOutEvaluator(ori=DATA,syn=SYNTHETIC_DATA,control=CONTROL_DATA,n_attacks=500)
    singling_evaluator.evaluate(mode='univariate')
    singling_risk = singling_evaluator.risk().value

    #== Linkability ==#    
    linkability_evaluator = LinkabilityEvaluator(ori=DATA,syn=SYNTHETIC_DATA,control=CONTROL_DATA,n_attacks=500,aux_cols=KEY_COLUMNS,n_neighbors=10)
    linkability_evaluator.evaluate() #n_jobs=-2
    linkability_risk = linkability_evaluator.risk().value

    #== Inference ==#    
    #inference_evaluator = InferenceEvaluator(ori=DATA,syn=SYNTHETIC_DATA,control=CONTROL_DATA,n_attacks=500,aux_cols=KEY_COLUMNS,secret=SENSITIVE_COLUMNS)
    #inference_evaluator.evaluate() #n_jobs=-2
    #inference_risk = inference_evaluator.risk().value

    print(exact_matches_score)
    print(detection_score)
    print(inference_protection_score)
    print(singling_risk)
    print(linkability_risk)