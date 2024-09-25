import pandas as pd
import numpy as np
from sdmetrics.single_column import BoundaryAdherence,CategoryAdherence,KSComplement,TVComplement,StatisticSimilarity,RangeCoverage,CategoryCoverage
from sdmetrics.column_pairs import CorrelationSimilarity,ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis,LogisticDetection,BinaryDecisionTreeClassifier,CategoricalCAP,CategoricalKNN,NumericalMLP
from sdv.metadata import SingleTableMetadata
from itertools import combinations
from anonymeter.evaluators import SinglingOutEvaluator,LinkabilityEvaluator,InferenceEvaluator
from generate.syntheticdata import create_metadata
import random
from scipy import stats
from functools import reduce

def evaluate_utility(type, data, synthetic_data, identifier_column, prediction_type, prediction_column):
    if type == 'multi':
        data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data)
        synthetic_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), synthetic_data)
    data = data.drop(columns=[identifier_column])

    metadata = create_metadata(data)

    discrete_columns = []
    for col, meta in metadata.columns.items():
        if ('sdtype' in meta and meta['sdtype'] == 'categorical') or (data[col].fillna(9999) % 1 == 0).all():
            discrete_columns.append(col)
    data_columns = data.columns

    #== Statistic Similarity ==# (in quality as well but should it be?)
    similarity_scores = []
    for column in data_columns:
        if column not in discrete_columns:
            similarity_score = StatisticSimilarity.compute(real_data=data[column], synthetic_data=synthetic_data[column], statistic='mean')
            similarity_scores.append(similarity_score)

    #== Correlation ==#
    #print()
    #print("[SynthOpt] calculating correlation scores (this may take a while)")
    correlation_scores = []
    if not synthetic_data.columns[synthetic_data.nunique()==1].tolist():
        column_pairs = list(combinations(data_columns, 2))
        #column_pairs = random.sample(column_pairs, 10)    # For testing!, takes random sample of column pairs to speed up time

        for col1, col2 in column_pairs:        
            correlation_score = data[col1].corr(data[col2]) - synthetic_data[col1].corr(synthetic_data[col2])
            #print(f"(corr) real data correlation : {data[col1].corr(data[col2])} | synthetic data correlation : {synthetic_data[col1].corr(synthetic_data[col2])}")
            #print(correlation_score)
            correlation_scores.append(correlation_score)


    # Calculate the Pearson correlation matrices for both datasets
    #real_corr_matrix = data.corr(method='pearson')
    #synth_corr_matrix = synthetic_data.corr(method='pearson')


    #== ML Efficacy ==# (maybe create own with optimisation of hyperparams (as option)) (SHOULD BE ABLE TO CHOOSE REGRESSION / CLASSIFICATION / MULTI-CLASS)
    #print("[SynthOpt] training & evaluating performance of machine learning classifiers (this may take a while)")   
    ml_efficacy_score = BinaryDecisionTreeClassifier.compute(test_data=data, train_data=synthetic_data, target=prediction_column, metadata=metadata)

    avg_similarity_score = np.round(np.mean(similarity_scores), 2)
    avg_correlation_score = np.round(np.mean(correlation_scores), 2) # the lower the better

    print()
    print("== UTILITY SCORES ==")
    print(f"statistic similarity score: {avg_similarity_score}")
    print(f"correlation score: {avg_correlation_score}")
    print(f"ml efficacy score: {ml_efficacy_score}")

    utility_scores = {
        'Similarity Total': avg_similarity_score,
        'Similarity Individual': similarity_scores,
        'Correlation Total': avg_correlation_score,
        'Correlation Individual': correlation_scores,
        'ML Efficacy': ml_efficacy_score
    }

    return utility_scores