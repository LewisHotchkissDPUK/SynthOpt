import pandas as pd
import numpy as np
from sdmetrics.single_column import BoundaryAdherence,CategoryAdherence,KSComplement,TVComplement,StatisticSimilarity,RangeCoverage,CategoryCoverage
from sdmetrics.column_pairs import CorrelationSimilarity,ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis,LogisticDetection,BinaryDecisionTreeClassifier,CategoricalCAP,CategoricalKNN,NumericalMLP
from sdv.metadata import SingleTableMetadata
from itertools import combinations
from anonymeter.evaluators import SinglingOutEvaluator,LinkabilityEvaluator,InferenceEvaluator
from generate.syntheticdata import create_metadata
import progressbar
import random

def evaluate_utility(data, synthetic_data, prediction_column):
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
    print()
    print("[SynthOpt] calculating correlation scores (this may take a while)")
    correlation_scores = []
    if not synthetic_data.columns[synthetic_data.nunique()==1].tolist():
        column_pairs = list(combinations(data_columns, 2))
        column_pairs = random.sample(column_pairs, 10)    # For testing!, takes random sample of column pairs to speed up time

        i = 0
        #widgets = [' [',progressbar.Timer(format='elapsed time: %(elapsed)s'),'] ',
        #    progressbar.Bar('*'),' (',progressbar.ETA(), ') ',]
        #bar = progressbar.ProgressBar(maxval=len(column_pairs), widgets=widgets).start()

        for col1, col2 in column_pairs:
            if col1 not in discrete_columns and col2 not in discrete_columns:
                correlation_score = CorrelationSimilarity.compute(real_data=data[[col1,col2]], synthetic_data=synthetic_data[[col1,col2]])
                correlation_scores.append(correlation_score)
            else:
                correlation_score = ContingencySimilarity.compute(real_data=data[[col1,col2]], synthetic_data=synthetic_data[[col1,col2]])
                correlation_scores.append(correlation_score)
            i += 1
            #bar.update(i)
            

    #== ML Efficacy ==# (maybe create own with optimisation of hyperparams (as option)) (SHOULD BE ABLE TO CHOOSE REGRESSION / CLASSIFICATION / MULTI-CLASS)
    print("[SynthOpt] training & evaluating performance of machine learning classifiers (this may take a while)")   
    ml_efficacy_score = BinaryDecisionTreeClassifier.compute(test_data=data, train_data=synthetic_data, target=prediction_column, metadata=metadata)

    avg_similarity_score = np.round(np.mean(similarity_scores), 2)
    avg_correlation_score = np.round(np.mean(correlation_scores), 2)

    print()
    print("== UTILITY SCORES ==")
    print(f"statistic similarity score: {avg_similarity_score}")
    print(f"correlation score: {avg_correlation_score}")
    print(f"ml efficacy score: {ml_efficacy_score}")