import pandas as pd
import numpy as np
from sdmetrics.single_column import BoundaryAdherence,CategoryAdherence,KSComplement,TVComplement,StatisticSimilarity,RangeCoverage,CategoryCoverage
from sdmetrics.column_pairs import CorrelationSimilarity,ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis,LogisticDetection,BinaryDecisionTreeClassifier,CategoricalCAP,CategoricalKNN,NumericalMLP
from sdv.metadata import SingleTableMetadata
from itertools import combinations
from anonymeter.evaluators import SinglingOutEvaluator,LinkabilityEvaluator,InferenceEvaluator
from synthopt.generate.syntheticdata import create_metadata
from functools import reduce

def evaluate_privacy(data, synthetic_data, identifier_column, sensitive_columns, key_columns, control_data, table_type = 'single'):
    if table_type == 'multi':
        data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data)
        control_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), control_data)
        synthetic_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), synthetic_data)
    if identifier_column != None:
        data = data.drop(columns=[identifier_column])
        control_data = control_data.drop(columns=[identifier_column])
        synthetic_data = synthetic_data.drop(columns=[identifier_column])

    number_attacks = round(control_data.shape[0])

    METADATA = create_metadata(data)

    #== Exact Matches ==#
    exact_matches_score = NewRowSynthesis.compute(real_data=data, synthetic_data=synthetic_data, metadata=METADATA, numerical_match_tolerance=1) # , synthetic_sample_size=5000

    #== Detection ==#        
    detection_score = LogisticDetection.compute(real_data=data, synthetic_data=synthetic_data, metadata=METADATA)

    #== Inference Attack Protection ==#        
    inference_protection_score = CategoricalCAP.compute(real_data=data, synthetic_data=synthetic_data, key_fields=key_columns, sensitive_fields=sensitive_columns)

    #== Singling Out ==#    
    print("[SynthOpt] conducting singling out attacks (this may take a while)")
    singling_evaluator = SinglingOutEvaluator(ori=data,syn=synthetic_data,control=control_data,n_attacks=number_attacks) # number of records to attack (CONVERT TO PERCENTAGE)
    singling_evaluator.evaluate(mode='univariate')
    singling_risk = singling_evaluator.risk().value

    #== Linkability ==#    
    print("[SynthOpt] conducting linkability attacks (this may take a while)")
    #aux_cols should be a list of two seperate lists represetning two different sets of key attributs which could be linked
    #split up the list into two
    linkability_evaluator = LinkabilityEvaluator(ori=data,syn=synthetic_data,control=control_data,n_attacks=number_attacks,aux_cols=key_columns,n_neighbors=10)
    linkability_evaluator.evaluate(n_jobs=-2)
    linkability_risk = linkability_evaluator.risk().value

    #== Inference ==#    
    print("[SynthOpt] conducting inference attacks (this may take a while)")
    columns = data.columns
    results = []
    for secret in columns:
        aux_cols = [col for col in columns if col != secret]
        inference_evaluator = InferenceEvaluator(ori=data,syn=synthetic_data,control=control_data,n_attacks=number_attacks,aux_cols=key_columns,secret=sensitive_columns)
        inference_evaluator.evaluate(n_jobs=-2)
        #results.append((secret, inference_evaluator.results()))
        results.append(inference_evaluator.results().risk().value)
    inference_risk = np.mean(results)

    print()
    print("== PRIVACY SCORES ==")

    print(f"exact matches score: {exact_matches_score}")
    print(f"detection score: {detection_score}")
    print(f"inference protection score: {inference_protection_score}")
    print(f"singling out score: {singling_risk}")
    print(f"linkability score: {linkability_risk}")
    print(f"inference score: {inference_risk}")

    privacy_scores = {
        'Exact Matches Total': exact_matches_score,
        'Detection Total': round(detection_score, 2),
        'Inference Protection Total': round(inference_protection_score, 2),
        'Singling Risk Total': round(singling_risk, 2),
        'Linkability Risk Total': round(linkability_risk, 2),
        'Inference Risk Total': round(inference_risk, 2)
    }

    return privacy_scores
