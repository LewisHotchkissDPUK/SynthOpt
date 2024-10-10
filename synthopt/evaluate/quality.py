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

def evaluate_quality(data, synthetic_data, identifier_column, table_type = 'single'):
    if table_type == 'multi':
        data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data)
        synthetic_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), synthetic_data)
    if identifier_column != None:
        data = data.drop(columns=[identifier_column])
        synthetic_data = synthetic_data.drop(columns=[identifier_column])

    metadata = create_metadata(data)

    discrete_columns = []
    for col, meta in metadata.columns.items():
        if ('sdtype' in meta and meta['sdtype'] == 'categorical'):
            discrete_columns.append(col)
    data_columns = data.columns

    boundary_adherence_scores = []
    coverage_scores = []
    complement_scores = []
    for column in data_columns:
        if column not in discrete_columns:
            #== Boundary Adherence ==#
            adherence_score = BoundaryAdherence.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            boundary_adherence_scores.append(adherence_score)
            #== Coverage ==#
            coverage_score = RangeCoverage.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            coverage_scores.append(coverage_score)
            #== Complement ==#
            complement_score = KSComplement.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            complement_scores.append(complement_score)
        else:
            #== Boundary Adherence ==#
            adherence_score = CategoryAdherence.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            boundary_adherence_scores.append(adherence_score)
            #== Coverage ==#
            coverage_score = CategoryCoverage.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            coverage_scores.append(coverage_score)
            #== Complement ==#
            complement_score = TVComplement.compute(real_data=data[column], synthetic_data=synthetic_data[column])
            complement_scores.append(complement_score)

    avg_boundary_adherence_score = np.round(np.mean(boundary_adherence_scores), 2)
    avg_coverage_score = np.round(np.mean(coverage_scores), 2)
    avg_complement_score = np.round(np.mean(complement_scores), 2)

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
