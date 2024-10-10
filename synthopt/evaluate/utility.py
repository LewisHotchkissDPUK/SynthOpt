import pandas as pd
import numpy as np
from sdmetrics.single_column import BoundaryAdherence,CategoryAdherence,KSComplement,TVComplement,StatisticSimilarity,RangeCoverage,CategoryCoverage
from sdmetrics.column_pairs import CorrelationSimilarity,ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis,LogisticDetection,BinaryDecisionTreeClassifier,CategoricalCAP,CategoricalKNN,NumericalMLP
from sdv.metadata import SingleTableMetadata
from itertools import combinations
from anonymeter.evaluators import SinglingOutEvaluator,LinkabilityEvaluator,InferenceEvaluator
from synthopt.generate.syntheticdata import create_metadata
import random
from scipy import stats
from functools import reduce
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

def classifier_performance(real_data, synthetic_data, control_data, prediction_column, prediction_type):
    # Prepare the data
    X_real = real_data.drop(columns=[prediction_column])
    y_real = real_data[prediction_column]

    X_synthetic = synthetic_data.drop(columns=[prediction_column])
    y_synthetic = synthetic_data[prediction_column]

    X_control = control_data.drop(columns=[prediction_column])
    y_control = control_data[prediction_column]

     # Hyperparameter distributions for RandomizedSearchCV
    param_distributions_classifier = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': randint(5, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': [None, 'sqrt', 'log2']
    }

    param_distributions_regressor = {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter': ['best', 'random'],
        'max_depth': randint(5, 50),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': [None, 'sqrt', 'log2']
    }

    # Initialize variables to store results
    f1_real = None
    f1_synthetic = None
    r2_real = None
    r2_synthetic = None
    score_difference = None

    # Train and test models based on prediction_type
    if prediction_type == 'binary' or prediction_type == 'multiclass':
        # Use DecisionTreeClassifier for classification
        classifier_real = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions_classifier)
        classifier_real.fit(X_real, y_real)
        y_pred_real = classifier_real.predict(X_control)

        # Calculate F1 score or accuracy depending on the prediction_type
        if prediction_type == 'binary':
            f1_real = f1_score(y_control, y_pred_real, average='binary')
        else:
            f1_real = f1_score(y_control, y_pred_real, average='weighted')

        classifier_synthetic = RandomizedSearchCV(DecisionTreeClassifier(), param_distributions_classifier)
        classifier_synthetic.fit(X_synthetic, y_synthetic)
        y_pred_synthetic = classifier_synthetic.predict(X_control)

        if prediction_type == 'binary':
            f1_synthetic = f1_score(y_control, y_pred_synthetic, average='binary')
        else:
            f1_synthetic = f1_score(y_control, y_pred_synthetic, average='weighted')

        # Calculate the difference in F1 scores
        score_difference = f1_real - f1_synthetic

        # Output performance comparison
        print(f"F1 Score (Real Data): {f1_real:.4f}")
        print(f"F1 Score (Synthetic Data): {f1_synthetic:.4f}")
        print(f"Difference in F1 Scores (Synthetic - Real): {score_difference:.4f}")

    elif prediction_type == 'regression':
        # Use DecisionTreeRegressor for regression
        regressor_real = RandomizedSearchCV(DecisionTreeRegressor(), param_distributions_regressor)
        regressor_real.fit(X_real, y_real)
        y_pred_real = regressor_real.predict(X_control)

        r2_real = r2_score(y_control, y_pred_real)

        regressor_synthetic = DecisionTreeRegressor()
        regressor_synthetic.fit(X_synthetic, y_synthetic)
        y_pred_synthetic = regressor_synthetic.predict(X_control)

        r2_synthetic = r2_score(y_control, y_pred_synthetic)

        # Calculate the difference in R-squared values
        score_difference =  r2_real - r2_synthetic

        # Output performance comparison
        print(f"R-squared (Real Data): {r2_real:.4f}")
        print(f"R-squared (Synthetic Data): {r2_synthetic:.4f}")
        print(f"Difference in R-squared (Synthetic - Real): {score_difference:.4f}")

    else:
        raise ValueError("Invalid prediction_type. Use 'binary', 'multiclass', or 'regression'.")

    return 1-score_difference

def evaluate_utility(data, synthetic_data, control_data, identifier_column, prediction_column, table_type = 'single', prediction_type = 'binary'):
    if table_type == 'multi':
        data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data)
        synthetic_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), synthetic_data)
        control_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), control_data)
    if identifier_column != None:
        data = data.drop(columns=[identifier_column])
        synthetic_data = synthetic_data.drop(columns=[identifier_column])
        control_data = control_data.drop(columns=[identifier_column])

    metadata = create_metadata(data)

    discrete_columns = []
    for col, meta in metadata.columns.items():
        #if ('sdtype' in meta and meta['sdtype'] == 'categorical') or (data[col].fillna(9999) % 1 == 0).all():
        if ('sdtype' in meta and meta['sdtype'] == 'categorical'):
            discrete_columns.append(col)
    data_columns = data.columns

    #== Statistic Similarity ==# (in quality as well but should it be?)
    similarity_scores = []
    for column in data_columns:
        #if column not in discrete_columns:
        #    # only does mean, maybe add standard deviation and median etc
        #    similarity_score = StatisticSimilarity.compute(real_data=data[column], synthetic_data=synthetic_data[column], statistic='mean')
        #    similarity_scores.append(similarity_score)
        #meant to only be for continous not discrete, but there doesnt seem to be discrete alternative
        similarity_score = StatisticSimilarity.compute(real_data=data[column], synthetic_data=synthetic_data[column], statistic='mean')
        similarity_scores.append(similarity_score)


    """
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
    """

    #== Correlation ==#
    print()
    print("[SynthOpt] calculating correlation scores (this may take a while)")
    correlation_scores = []
    if not synthetic_data.columns[synthetic_data.nunique()==1].tolist():
        column_pairs = list(combinations(data_columns, 2))
        num = min(40, len(data.columns))
        column_pairs = random.sample(column_pairs, num)    # For testing!, takes random sample of column pairs to speed up time
        for col1, col2 in column_pairs:
            if col1 not in discrete_columns and col2 not in discrete_columns:
                correlation_score = CorrelationSimilarity.compute(real_data=data[[col1,col2]], synthetic_data=synthetic_data[[col1,col2]])
                correlation_scores.append(correlation_score)
            else:
                correlation_score = ContingencySimilarity.compute(real_data=data[[col1,col2]], synthetic_data=synthetic_data[[col1,col2]])
                correlation_scores.append(correlation_score)


    #== ML Efficacy ==# (maybe create own with optimisation of hyperparams (as option)) (SHOULD BE ABLE TO CHOOSE REGRESSION / CLASSIFICATION / MULTI-CLASS)
    #print("[SynthOpt] training & evaluating performance of machine learning classifiers (this may take a while)")   
    #ml_efficacy_score_real = BinaryDecisionTreeClassifier.compute(test_data=control_data, train_data=data, target=prediction_column, metadata=metadata)
    #print(f"real ml = {ml_efficacy_score_real}")
    #ml_efficacy_score_synth = BinaryDecisionTreeClassifier.compute(test_data=control_data, train_data=synthetic_data, target=prediction_column, metadata=metadata)
    #print(f"synthetic ml = {ml_efficacy_score_synth}")
    #ml_efficacy_score = ml_efficacy_score_real - ml_efficacy_score_synth

    # add multi class and regression prediction types
    #ml_efficacy_score = BinaryDecisionTreeClassifier.compute(test_data=control_data, train_data=synthetic_data, target=prediction_column, metadata=metadata)
    ml_efficacy_score = classifier_performance(data, synthetic_data, control_data, prediction_column, prediction_type)

    avg_similarity_score = np.round(np.mean(similarity_scores), 2)
    avg_correlation_score = np.round(np.mean(correlation_scores), 2) # the lower the better

    print()
    print("== UTILITY SCORES ==")
    print(f"statistic similarity score: {avg_similarity_score}")
    print(f"correlation score: {avg_correlation_score}")
    print(f"ml efficacy score: {ml_efficacy_score}")

    utility_scores = {
        'Statistic Similarity Total': avg_similarity_score,
        'Statistic Similarity Individual': similarity_scores,
        'Correlation Total': avg_correlation_score,
        'Correlation Individual': correlation_scores,
        'ML Efficacy Total': round(ml_efficacy_score, 2)
    }

    return utility_scores
