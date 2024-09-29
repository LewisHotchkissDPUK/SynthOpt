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

    # Evaluate privacy, quality, and utility scores
    privacy_scores = evaluate_privacy(data, synthetic_data, identifier_column, sensitive_columns, key_columns, control_data, table_type)
    quality_scores = evaluate_quality(data, synthetic_data, identifier_column, table_type)
    utility_scores = evaluate_utility(data, synthetic_data, control_data, identifier_column, prediction_column, table_type, prediction_type)

    # Convert scores to numpy arrays and filter out non-numeric values
    privacy_values = np.array(list(privacy_scores.values()), dtype=float)
    quality_values = np.array(list(quality_scores.values()), dtype=float)
    utility_values = np.array(list(utility_scores.values()), dtype=float)

    # Check for NaN values and remove them
    privacy_values = privacy_values[~np.isnan(privacy_values)]
    quality_values = quality_values[~np.isnan(quality_values)]
    utility_values = utility_values[~np.isnan(utility_values)]

    # Calculate means safely
    mean_privacy = np.mean(privacy_values) if privacy_values.size > 0 else 0
    mean_quality = np.mean(quality_values) if quality_values.size > 0 else 0
    mean_utility = np.mean(utility_values) if utility_values.size > 0 else 0

    # Calculate a weighted score
    total_score = (
        weights['privacy'] * mean_privacy +
        weights['quality'] * mean_quality +
        weights['utility'] * mean_utility
    )
    
    return -total_score  # Minimize the negative score

def optimize_epsilon(data, model, table_type, identifier_column, prediction_column, prediction_type, sensitive_columns, key_columns, control_data, sample_size, iterations, weights):
    # Define the bounds for epsilon
    bounds = [(0.1, 10)]  # Epsilon can vary between 0.1 and 10

    # Use minimize to optimize epsilon
    result = minimize(
        objective_function,
        x0=[5],  # Initial guess for epsilon
        args=(weights, data, model, table_type, identifier_column, prediction_column, prediction_type, sensitive_columns, key_columns, control_data, sample_size, iterations),
        bounds=bounds,
        method='L-BFGS-B'
    )

    optimal_epsilon = result.x[0]
    optimal_score = -result.fun  # Since we minimized the negative score

    return optimal_epsilon, optimal_score