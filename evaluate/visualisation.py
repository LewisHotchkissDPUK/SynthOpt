import matplotlib.pyplot as plt

def combine_dicts(*dicts):
    combined = {}
    for d in dicts:
        for key, value in d.items():
            combined[key] = value  # Directly assign the value to the key
    return combined

def table_vis(privacy_scores, quality_scores, utility_scores, data_columns):

    combined = combine_dicts(privacy_scores, quality_scores, utility_scores)
    total_combined = {key: value for key, value in combined.items() if 'Total' in key}

    x = list(total_combined.keys())
    y = list(total_combined.values())

    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.barh(x, y, color='b')

    plt.tight_layout()
    plt.savefig("/workspaces/SynthOpt/output/table_vis.png")

#def attribute_vis():