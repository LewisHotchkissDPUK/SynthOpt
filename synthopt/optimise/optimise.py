from synthopt.generate.syntheticdata import generate_syntheticdata
from synthopt.evaluate.privacy import evaluate_privacy
from synthopt.evaluate.utility import evaluate_utility
from synthopt.evaluate.quality import evaluate_quality
import numpy as np
from scipy.optimize import minimize



def objective_function(epsilon, weights, data, model, table_type, identifier_column, prediction_column, prediction_type, sensitive_columns, key_columns, control_data, sample_size, iterations):
    # Generate synthetic data with the current epsilon value
    synthetic_data = generate_syntheticdata(
        data,
        identifier_column,
        prediction_column,
        sensitive_columns,
        sample_size,
        table_type,
        model_name=model,
        iterations=iterations,
        dp_epsilon=epsilon
    )

    print(f"Trying epsilon value {epsilon}")

    # Evaluate privacy, quality, and utility scores
    privacy_scores = evaluate_privacy(data, synthetic_data, identifier_column, sensitive_columns, key_columns, control_data, table_type)
    quality_scores = evaluate_quality(data, synthetic_data, identifier_column, table_type)
    utility_scores = evaluate_utility(data, synthetic_data, control_data, identifier_column, prediction_column, table_type, prediction_type)

    privacy_scores['Detection Total'] = 1 - privacy_scores['Detection Total']
    privacy_scores['Singling Risk Total'] = 1 - privacy_scores['Singling Risk Total']
    privacy_scores['Linkability Risk Total'] = 1 - privacy_scores['Linkability Risk Total']
    privacy_scores['Inference Risk Total'] = 1 - privacy_scores['Inference Risk Total']

    privacy_total_scores = [value for key, value in privacy_scores.items() if 'Total' in key]
    privacy_mean_total_score = sum(privacy_total_scores) / len(privacy_total_scores)

    utility_total_scores = [value for key, value in utility_scores.items() if 'Total' in key]
    utility_mean_total_score = sum(utility_total_scores) / len(utility_total_scores)

    quality_total_scores = [value for key, value in quality_scores.items() if 'Total' in key]
    quality_mean_total_score = sum(quality_total_scores) / len(quality_total_scores)

    utility_quality_total_scores = utility_total_scores + quality_total_scores
    utility_quality_mean_total_score = sum(utility_quality_total_scores) / len(utility_quality_total_scores)

    # Calculate a weighted score
    total_score = (
        weights['privacy'] * privacy_mean_total_score +
        weights['utility'] * utility_quality_mean_total_score
    )

    print(f"Got privacy score {privacy_mean_total_score}")
    print(f"Got utility score {utility_quality_mean_total_score}")
    print()
    
    return -total_score  # Minimize the negative score

def optimise_epsilon(data, model, table_type, identifier_column, prediction_column, prediction_type, sensitive_columns, key_columns, control_data, sample_size, iterations, weights):
    # Define the bounds for epsilon
    bounds = [(0.01, 10)]  # Epsilon can vary between 0.1 and 10

    # Use minimize to optimize epsilon
    result = minimize(
        objective_function,
        x0=[5],  # Initial guess for epsilon
        args=(weights, data, model, table_type, identifier_column, prediction_column, prediction_type, sensitive_columns, key_columns, control_data, sample_size, iterations),
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 0.001}
    )

    optimal_epsilon = result.x[0]
    optimal_score = -result.fun  # Since we minimized the negative score

    return optimal_epsilon, optimal_score