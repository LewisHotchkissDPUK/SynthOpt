import pandas as pd
import numpy as np
from sdmetrics.single_column import BoundaryAdherence, CategoryAdherence, KSComplement, TVComplement, StatisticSimilarity, RangeCoverage, CategoryCoverage
from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity
from sdmetrics.single_table import NewRowSynthesis, LogisticDetection, BinaryDecisionTreeClassifier, CategoricalCAP, CategoricalKNN, NumericalMLP
from sdv.metadata import SingleTableMetadata
from itertools import combinations
from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator
from synthopt.generate.syntheticdata import create_metadata
import random
from scipy import stats
from functools import reduce
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


def classifier_performance(real_data, synthetic_data, control_data, prediction_column, prediction_type):
    
    # Prepare the data
    scaler = StandardScaler()
    real_data = scaler.fit_transform(real_data)
    X_real = real_data.drop(columns=[prediction_column])
    y_real = real_data[prediction_column]

    synthetic_data = scaler.fit_transform(synthetic_data)
    X_synthetic = synthetic_data.drop(columns=[prediction_column])
    y_synthetic = synthetic_data[prediction_column]

    X_control = control_data.drop(columns=[prediction_column])
    y_control = control_data[prediction_column]

    # Hyperparameter distributions for RandomizedSearchCV (XGBoost specific)
    param_distributions_classifier = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 1.0),
    'colsample_bytree': uniform(0.5, 1.0),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(1, 3),
    }

    param_distributions_regressor = {
    'n_estimators': randint(100, 400),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.4),
    'subsample': uniform(0.5, 2.0),
    'colsample_bytree': uniform(0.5, 2.0),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 3),
    'reg_lambda': uniform(1, 4),
    }

    # Initialize variables to store results
    f1_real = None
    f1_synthetic = None
    r2_real = None
    r2_synthetic = None
    score_difference = None

    # Train and test models based on prediction_type
    if prediction_type == 'binary' or prediction_type == 'multiclass':
        # Use XGBClassifier for classification
        classifier_real = RandomizedSearchCV(XGBClassifier(eval_metric='logloss'), param_distributions_classifier, n_iter= 50)
        classifier_real.fit(X_real, y_real)
        y_pred_real = classifier_real.predict(X_control)

        # Calculate F1 score or accuracy depending on the prediction_type
        if prediction_type == 'binary':
            f1_real = f1_score(np.round(y_control), np.round(y_pred_real), average='binary')
        else:
            f1_real = f1_score(np.round(y_control), np.round(y_pred_real), average='weighted')

        classifier_synthetic = RandomizedSearchCV(XGBClassifier(eval_metric='logloss'), param_distributions_classifier, n_iter= 50)
        classifier_synthetic.fit(X_synthetic, y_synthetic)
        y_pred_synthetic = classifier_synthetic.predict(X_control)

        if prediction_type == 'binary':
            f1_synthetic = f1_score(np.round(y_control), np.round(y_pred_synthetic), average='binary')
        else:
            f1_synthetic = f1_score(np.round(y_control), np.round(y_pred_synthetic), average='weighted')

        # Calculate the difference in F1 scores
        score_difference = f1_real - f1_synthetic
        
        
        # Feature importances from classifier on real and synthetic data
        feature_importance_real_clf = classifier_real.best_estimator_.feature_importances_
        feature_importance_synth_clf = classifier_synthetic.best_estimator_.feature_importances_
        
        # Calculate the difference and take the average for classification
        importance_diff_clf = feature_importance_real_clf - feature_importance_synth_clf
        avg_importance_diff_clf = np.mean(np.abs(importance_diff_clf))


        # Output performance comparison
        print(f"F1 Score (Real Data): {f1_real:.4f}")
        print(f"F1 Score (Synthetic Data): {f1_synthetic:.4f}")
        print(f"Difference in F1 Scores (Real - Synthetic): {score_difference:.4f}")
        print(f"Average Feature Importance Difference (Classification): {avg_importance_diff_clf}")


    elif prediction_type == 'regression':
        # Use XGBRegressor for regression
        regressor_real = RandomizedSearchCV(XGBRegressor(), param_distributions_regressor, n_iter= 50)
        regressor_real.fit(X_real, y_real)
        y_pred_real = regressor_real.predict(X_control)

        r2_real = r2_score(np.round(y_control), np.round(y_pred_real))

        regressor_synthetic = RandomizedSearchCV(XGBRegressor(), param_distributions_regressor, n_iter= 50)
        regressor_synthetic.fit(X_synthetic, y_synthetic)
        y_pred_synthetic = regressor_synthetic.predict(X_control)

        r2_synthetic = r2_score(np.round(y_control), np.round(y_pred_synthetic))

        # Calculate the difference in R-squared values
        score_difference = r2_real - r2_synthetic

        feature_importance_real_reg = regressor_real.best_estimator_.feature_importances_
        feature_importance_synth_reg = regressor_synthetic.best_estimator_.feature_importances_
        importance_diff_reg = feature_importance_real_reg - feature_importance_synth_reg
        avg_importance_diff_reg = np.mean(np.abs(importance_diff_reg))

        # Output performance comparison
        print(f"R-squared (Real Data): {r2_real:.4f}")
        print(f"R-squared (Synthetic Data): {r2_synthetic:.4f}")
        print(f"Difference in R-squared (Real - Synthetic): {score_difference:.4f}")
        print(f"Average Feature Importance Difference (Regression): {avg_importance_diff_reg}")

    else:
        raise ValueError("Invalid prediction_type. Use 'binary', 'multiclass', or 'regression'.")

    return 1 - score_difference


def evaluate_utility(data, synthetic_data, control_data, identifier_column, prediction_column, table_type='single', prediction_type='binary'):
    if table_type == 'multi':
        data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), data)
        synthetic_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), synthetic_data)
        control_data = reduce(lambda left, right: pd.merge(left, right, on=identifier_column), control_data)

    if identifier_column is not None:
        data = data.drop(columns=[identifier_column])
        synthetic_data = synthetic_data.drop(columns=[identifier_column])
        control_data = control_data.drop(columns=[identifier_column])

    metadata = create_metadata(data)

    discrete_columns = []
    for col, meta in metadata.columns.items():
        if 'sdtype' in meta and meta['sdtype'] == 'categorical':
            discrete_columns.append(col)
    data_columns = data.columns

    # Statistic Similarity
    similarity_scores = []
    for column in data_columns:
        similarity_score = StatisticSimilarity.compute(real_data=data[column], synthetic_data=synthetic_data[column], statistic='mean')
        similarity_scores.append(similarity_score)

    # Correlation Similarity
    print("[SynthOpt] calculating correlation scores (this may take a while)")
    correlation_scores = []
    if not synthetic_data.columns[synthetic_data.nunique() == 1].tolist():
        column_pairs = list(combinations(data_columns, 2))
        num = min(40, len(data.columns))
        column_pairs = random.sample(column_pairs, num)  # Random sample to speed up
        for col1, col2 in column_pairs:
            if col1 not in discrete_columns and col2 not in discrete_columns:
                correlation_score = CorrelationSimilarity.compute(real_data=data[[col1, col2]], synthetic_data=synthetic_data[[col1, col2]])
            else:
                correlation_score = ContingencySimilarity.compute(real_data=data[[col1, col2]], synthetic_data=synthetic_data[[col1, col2]])
            correlation_scores.append(correlation_score)

    # ML Efficacy
    ml_efficacy_score = classifier_performance(data, synthetic_data, control_data, prediction_column, prediction_type)

    avg_similarity_score = np.round(np.mean(similarity_scores), 2)
    avg_correlation_score = np.round(np.mean(correlation_scores), 2)

    print("\n== UTILITY SCORES ==")
    print(f"Statistic Similarity Score: {avg_similarity_score}")
    print(f"Correlation Score: {avg_correlation_score}")
    print(f"ML Efficacy Score: {ml_efficacy_score}")

    utility_scores = {
        'Statistic Similarity Total': avg_similarity_score,
        'Statistic Similarity Individual': similarity_scores,
        'Correlation Total': avg_correlation_score,
        'Correlation Individual': correlation_scores,
        'ML Efficacy Total': round(ml_efficacy_score, 2)
    }

    return utility_scores
