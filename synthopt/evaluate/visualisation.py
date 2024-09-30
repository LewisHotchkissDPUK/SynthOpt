import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from synthopt.generate.syntheticdata import create_metadata

def combine_dicts(*dicts):
    combined = {}
    for d in dicts:
        for key, value in d.items():
            combined[key] = value  # Directly assign the value to the key
    return combined

def table_vis(privacy_scores, quality_scores, utility_scores):

    fig, ax = plt.subplots(figsize=(8, 5))

    combined = combine_dicts(privacy_scores, quality_scores, utility_scores)
    total_combined = {key: value for key, value in combined.items() if 'Total' in key}

    x = list(total_combined.keys())
    y = list(total_combined.values())

    # Create a plot
    #plt.figure(figsize=(8, 5))
    ax.barh(x, y, color='b')

    ax.set_xlabel('Score')
    ax.set_ylabel('Metric')
    ax.set_title('Summary of Scores for Each Metric')

    fig.tight_layout()
    #fig.savefig("/workspaces/SynthOpt/output/table_vis.png")

    return fig

def attribute_vis(metric_name, scores, data_columns):
    y = scores.get(metric_name, [])
    
    # Check if the lengths of data_columns and y match
    if len(data_columns) != len(y):
        raise ValueError("Length of data_columns and the values for the selected score_name do not match.")
    
    # Combine the variables and values into a list of tuples
    combined = list(zip(data_columns, y))
    
    # Sort the combined list by the values in descending order
    combined_sorted = sorted(combined, key=lambda pair: pair[1], reverse=True)
    
    # Split into top 10 and bottom 10
    top_10 = combined_sorted[:10]
    bottom_10 = combined_sorted[-10:]
    
    # Extract names and values for top 10 and bottom 10
    top_10_names, top_10_values = zip(*top_10)
    bottom_10_names, bottom_10_values = zip(*bottom_10)
    
    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the top 10 variables
    axs[0].barh(top_10_names, top_10_values, color='blue')
    axs[0].set_title(f'Top 10 Variables for {metric_name}')
    axs[0].invert_yaxis()  # Highest values at the top
    axs[0].set_xlabel('Value')
    axs[0].set_xlim(0, 1.1)
    
    # Plot the bottom 10 variables
    axs[1].barh(bottom_10_names, bottom_10_values, color='red')
    axs[1].set_title(f'Bottom 10 Variables for {metric_name}')
    axs[1].invert_yaxis()  # Lowest values at the top
    axs[1].set_xlabel('Value')
    axs[1].set_xlim(0, 1.1)

    # Adjust layout for clarity
    plt.tight_layout()

    return fig

def distribution_vis(data, synthetic_data, data_columns):
    num_columns = min(12, len(data_columns)) ## use the same method for plotting individual metric scores top and bottom 10 or min
    selected_columns = random.sample(data_columns, num_columns)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()  # Flatten the axes array for easier iteration
    
    for i, col in enumerate(selected_columns):
        # Plot real data distribution (blue)
        #sns.kdeplot(data[col], ax=axes[i], color='blue', shade=True, label='Real', alpha=0.5)
        # Plot synthetic data distribution (red)
        #sns.kdeplot(synthetic_data[col], ax=axes[i], color='red', shade=True, label='Synthetic', alpha=0.5)

        min_val = min(data[col].min(), synthetic_data[col].min())
        max_val = max(data[col].max(), synthetic_data[col].max())
        bins = np.linspace(min_val, max_val, 30)
        sns.histplot(data[col], ax=axes[i], color='blue', label='Real', alpha=0.5, bins=bins, stat='density')
        sns.histplot(synthetic_data[col], ax=axes[i], color='red', label='Synthetic', alpha=0.5, bins=bins, stat='density')
        
        sns.kdeplot(data[col], ax=axes[i], color='blue', lw=2, fill=False)
        sns.kdeplot(synthetic_data[col], ax=axes[i], color='red', lw=2, fill=False)
        
        # Add title and legend
        axes[i].set_title(f'Distribution of {col}')
        axes[i].legend()

    # Turn off the remaining empty axes (if any)
    for j in range(i + 1, 9):
        axes[j].axis('off')

    # Adjust layout for better appearance
    plt.tight_layout()
    
    # Return the figure object
    return fig

def correlation_vis(data, synthetic_data, data_columns):
    metadata = create_metadata(data)
    cat_columns = []
    for col, meta in metadata.columns.items():
        if ('sdtype' in meta and meta['sdtype'] == 'categorical'):
            cat_columns.append(col)
    continuous_columns = [col for col in data_columns if col not in cat_columns]
    num_columns = len(continuous_columns)
    if num_columns < 2:
        print("Not enough continuous columns to plot correlations.")
        return

    num_plots = min(12, num_columns * (num_columns - 1) // 2)  # Max number of unique pairs
    column_pairs = random.sample([(x, y) for x in continuous_columns for y in continuous_columns if x != y], num_plots)

    # Ensure synthetic data has the same number of samples as the real data
    if len(synthetic_data) > len(data):
        synthetic_data = synthetic_data.sample(n=len(data), random_state=42).reset_index(drop=True)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    for i, (col_x, col_y) in enumerate(column_pairs):
        # Plot scatter plot and regression line for real data (blue)
        sns.regplot(x=data[col_x], y=data[col_y], ax=axes[i], scatter_kws={'alpha': 0.5}, 
                    line_kws={'color': 'blue'}, color='blue', label='Real')
        
        # Plot scatter plot and regression line for synthetic data (red)
        sns.regplot(x=synthetic_data[col_x], y=synthetic_data[col_y], ax=axes[i], scatter_kws={'alpha': 0.5}, 
                    line_kws={'color': 'red'}, color='red', label='Synthetic')
        
        # Set title and labels
        axes[i].set_title(f'{col_x} vs {col_y}')
        axes[i].legend()

    # Turn off the remaining empty axes (if any)
    for j in range(i + 1, 12):
        axes[j].axis('off')

    # Adjust layout for better appearance
    plt.tight_layout()

    # Return the figure object
    return fig


def reduction_vis(real_data, synthetic_data):
    # Ensure real_data and synthetic_data have the same number of columns (features)
    if real_data.shape[1] != synthetic_data.shape[1]:
        raise ValueError("Real and synthetic data must have the same number of features (columns).")
    
    data_columns = real_data.columns
    real_data = real_data
    imputer = KNNImputer(n_neighbors=3) ## Maybe improve this or add other options (hyperimpute maybe)
    real_data = imputer.fit_transform(real_data)
    real_data = pd.DataFrame(real_data, columns=data_columns)
    
    # Combine real and synthetic data
    combined_data = pd.concat([real_data, synthetic_data], axis=0)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(combined_data)
    
    # Split PCA results back into real and synthetic components
    real_pca = pca_results[:real_data.shape[0], :]
    synthetic_pca = pca_results[real_data.shape[0]:, :]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(real_pca[:, 0], real_pca[:, 1], color='blue', label='Real Data', alpha=0.6)
    ax.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], color='red', label='Synthetic Data', alpha=0.6)
    ax.set_title('PCA Plot of Real and Synthetic Data')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.legend()
    ax.grid(True)
    return fig