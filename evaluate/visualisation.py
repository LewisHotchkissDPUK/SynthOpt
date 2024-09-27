import matplotlib.pyplot as plt

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
    fig.savefig("/workspaces/SynthOpt/output/table_vis.png")

    return fig

def attribute_vis(metric_name, scores, data_columns):
    # SHOW TOP 20 AND BOTTOM 20
    # Maybe add x and y to a dictionary to then be able to sort

    #total_combined = {key: value for key, value in combined.items() if 'Individual' in key}
    
    #boundary_adherence = total_combined['Boundary Adherence Individual']
    #x = list(data_columns)
    #y = list(boundary_adherence)
    #plt.figure(figsize=(5, 9))
    #plt.barh(x, y, color='b')
    #plt.tight_layout()
    #plt.savefig("/workspaces/SynthOpt/output/attribute_boundary_adherence_vis.png")

    # MAKE SURE THE BOTTOM AXIS ALWAYS GOES UP TO SCORE 1


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
    axs[0].set_xlim(0, 1)
    
    # Plot the bottom 10 variables
    axs[1].barh(bottom_10_names, bottom_10_values, color='red')
    axs[1].set_title(f'Bottom 10 Variables for {metric_name}')
    axs[1].invert_yaxis()  # Lowest values at the top
    axs[1].set_xlabel('Value')
    axs[1].set_xlim(0, 1)

    # Adjust layout for clarity
    plt.tight_layout()

    return fig