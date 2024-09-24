import pandas as pd
import numpy as np
from sdmetrics.single_column import BoundaryAdherence,CategoryAdherence,KSComplement,TVComplement,StatisticSimilarity,RangeCoverage,CategoryCoverage
from sdmetrics.column_pairs import CorrelationSimilarity,ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis,LogisticDetection,BinaryDecisionTreeClassifier,CategoricalCAP,CategoricalKNN,NumericalMLP
from sdv.metadata import SingleTableMetadata
from itertools import combinations
from anonymeter.evaluators import SinglingOutEvaluator,LinkabilityEvaluator,InferenceEvaluator
from generate.syntheticdata import create_metadata

def evaluate_quality(data, synthetic_data):
    metadata = create_metadata(data)

    discrete_columns = []
    for col, meta in metadata.columns.items():
        if ('sdtype' in meta and meta['sdtype'] == 'categorical') or (data[col].fillna(9999) % 1 == 0).all():
            discrete_columns.append(col)
    data_columns = data.columns

    boundary_adherence_scores = []
    coverage_scores = []
    complement_scores = []
    similarity_scores = []
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
            #== Statistic Similarity ==#
            similarity_score = StatisticSimilarity.compute(real_data=data[column], synthetic_data=synthetic_data[column], statistic='mean')
            similarity_scores.append(similarity_score)
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
    avg_similarity_score = np.round(np.mean(similarity_scores), 2)

    print()
    print("== QUALITY SCORES ==")
    print(f"boundary adherence score: {avg_boundary_adherence_score}")
    print(f"coverage score: {avg_coverage_score}")
    print(f"complement score: {avg_complement_score}")
    print(f"statistic similarity score: {avg_similarity_score}")