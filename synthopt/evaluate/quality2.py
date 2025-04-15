import pandas as pd
import numpy as np
from sdmetrics.single_column import BoundaryAdherence,CategoryAdherence,KSComplement,TVComplement,StatisticSimilarity,RangeCoverage,CategoryCoverage
from sdmetrics.column_pairs import CorrelationSimilarity,ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis,LogisticDetection,BinaryDecisionTreeClassifier,CategoricalCAP,CategoricalKNN,NumericalMLP
from sdv.metadata import SingleTableMetadata
from itertools import combinations
from anonymeter.evaluators import SinglingOutEvaluator,LinkabilityEvaluator,InferenceEvaluator
from functools import reduce

def evaluate_quality(data, synthetic_data, metadata, identifier_column = None, table_type = 'single'):
    
    if table_type == 'multi' and isinstance(data, dict) and isinstance(synthetic_data, dict):
        data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data.values())
        synthetic_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), synthetic_data.values())
    if identifier_column != None:
        data = data.drop(columns=[identifier_column])
        synthetic_data = synthetic_data.drop(columns=[identifier_column])

    # Process columns based on datatype in metadata
    for column in data.columns:
        variable_name = column
        datatype_row = metadata[metadata['variable_name'] == variable_name]
        datatype = datatype_row['datatype'].values[0]
        
        if datatype == 'categorical string':
            # Encode categorical string values
            unique_values = data[column].astype(str).unique()
            encoding_map = {value: idx for idx, value in enumerate(unique_values)}
            data[column] = data[column].map(encoding_map)
            synthetic_data[column] = synthetic_data[column].map(encoding_map)
        
        #elif datatype == 'datetime':
        #    # Convert datetime to Unix timestamp
        #    data[column] = pd.to_datetime(data[column], format=metadata[metadata['variable_name']==column]['coding'].values[0]).astype('int64') // 10**9
        #    synthetic_data[column] = pd.to_datetime(synthetic_data[column], format=metadata[metadata['variable_name']==column]['coding'].values[0]).astype('int64') // 10**9

    discrete_columns = metadata[metadata['datatype'].isin(['categorical integer', 'categorical string'])]['variable_name'].tolist()
    other_numerical_columns = metadata[metadata['datatype'].isin(['integer', 'float'])]['variable_name'].tolist()  #, 'datetime'

    boundary_adherence_scores = []
    coverage_scores = []
    complement_scores = []
    
    for column in data.columns:
        if column in other_numerical_columns:
            if data[column].dropna().empty:
                continue
            #== Boundary Adherence ==#
            adherence_score = BoundaryAdherence.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            boundary_adherence_scores.append(adherence_score)
            #== Coverage ==#
            coverage_score = RangeCoverage.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            coverage_scores.append(coverage_score)
            #== Complement ==#
            complement_score = KSComplement.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            complement_scores.append(complement_score)
        if column in discrete_columns:
            if data[column].dropna().empty:
                continue
            #== Boundary Adherence ==#
            adherence_score = CategoryAdherence.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            boundary_adherence_scores.append(adherence_score)
            #== Coverage ==#
            coverage_score = CategoryCoverage.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            coverage_scores.append(coverage_score)
            #== Complement ==#
            complement_score = TVComplement.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            complement_scores.append(complement_score)

    avg_boundary_adherence_score = np.round(np.nanmean(boundary_adherence_scores), 2)
    avg_coverage_score = np.round(np.nanmean(coverage_scores), 2)
    avg_complement_score = np.round(np.nanmean(complement_scores), 2)

    print()
    print("== QUALITY SCORES ==")
    print(f"boundary adherence score: {avg_boundary_adherence_score}")
    print(f"coverage score: {avg_coverage_score}")
    print(f"complement score: {avg_complement_score}")

    quality_scores = {
        'Boundary Adherence Total': avg_boundary_adherence_score,
        'Boundary Adherence Individual': boundary_adherence_scores,
        'Coverage Total': avg_coverage_score,
        'Coverage Individual': coverage_scores,
        'Complement Total': avg_complement_score,
        'Complement Individual': complement_scores,
    }

    return quality_scores
