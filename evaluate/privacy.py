import pandas as pd
from sdmetrics.single_column import BoundaryAdherence,CategoryAdherence,KSComplement,TVComplement,StatisticSimilarity,RangeCoverage,CategoryCoverage
from sdmetrics.column_pairs import CorrelationSimilarity,ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis,LogisticDetection,BinaryDecisionTreeClassifier,CategoricalCAP,CategoricalKNN,NumericalMLP
from sdv.metadata import SingleTableMetadata
from itertools import combinations
from anonymeter.evaluators import SinglingOutEvaluator,LinkabilityEvaluator,InferenceEvaluator
from generate.syntheticdata import create_metadata

def evaluate_privacy(data, synthetic_data, sensitive_columns, key_columns, control_data):
    METADATA = create_metadata(data)

    #== Exact Matches ==#
    exact_matches_score = NewRowSynthesis.compute(real_data=data, synthetic_data=synthetic_data, metadata=METADATA, numerical_match_tolerance=0.1) # , synthetic_sample_size=5000

    #== Detection ==#        
    detection_score = LogisticDetection.compute(real_data=data, synthetic_data=synthetic_data, metadata=METADATA)

    #== Inference Attack Protection ==#        
    inference_protection_score = CategoricalCAP.compute(real_data=data, synthetic_data=synthetic_data, key_fields=key_columns, sensitive_fields=sensitive_columns)

    #== Singling Out ==#    
    print("running singling out attacks")
    singling_evaluator = SinglingOutEvaluator(ori=data,syn=synthetic_data,control=control_data,n_attacks=500)
    singling_evaluator.evaluate(mode='univariate')
    singling_risk = singling_evaluator.risk().value

    #== Linkability ==#    
    print("running linkability attacks")
    linkability_evaluator = LinkabilityEvaluator(ori=data,syn=synthetic_data,control=control_data,n_attacks=500,aux_cols=key_columns,n_neighbors=10)
    linkability_evaluator.evaluate() #n_jobs=-2
    linkability_risk = linkability_evaluator.risk().value

    #== Inference ==#    
    #inference_evaluator = InferenceEvaluator(ori=data,syn=synthetic_data,control=CONTROL_DATA,n_attacks=500,aux_cols=KEY_COLUMNS,secret=sensitive_columns)
    #inference_evaluator.evaluate() #n_jobs=-2
    #inference_risk = inference_evaluator.risk().value

    print(f"exact matches score:        {exact_matches_score}")
    print(f"detection score:            {detection_score}")
    print(f"inference protection score: {inference_protection_score}")
    print(f"singling out score:         {singling_risk}")
    print(f"linkability score:          {linkability_risk}")

    