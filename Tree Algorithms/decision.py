import pandas as pd
import numpy as np
from collections import Counter

# Function to calculate entropy
def entropy(column):
    counts = Counter(column)
    total = len(column)
    return -sum((count / total) * np.log2(count / total) for count in counts.values())

# Function to calculate information gain
def information_gain(data, split_attr, target_attr):
    total_entropy = entropy(data[target_attr])
    values = data[split_attr].unique()
    
    weighted_entropy = 0
    for val in values:
        subset = data[data[split_attr] == val]
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy(subset[target_attr])
    
    return total_entropy - weighted_entropy

# ID3 algorithm
def id3(data, target_attr, features):
    # If all values are the same, return that value
    if len(data[target_attr].unique()) == 1:
        return data[target_attr].iloc[0]

    # If no features left, return majority class
    if len(features) == 0:
        return data[target_attr].mode()[0]
    
    # Choose the best feature
    gains = {feature: information_gain(data, feature, target_attr) for feature in features}
    best_feature = max(gains, key=gains.get)
    
    tree = {best_feature: {}}
    
    for val in data[best_feature].unique():
        subset = data[data[best_feature] == val]
        if subset.empty:
            tree[best_feature][val] = data[target_attr].mode()[0]
        else:
            remaining_features = [f for f in features if f != best_feature]
            subtree = id3(subset, target_attr, remaining_features)
            tree[best_feature][val] = subtree
    
    return tree

# Helper to print the tree
def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "->", tree)
        return
    for feature, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{indent}{feature} = {value}")
            print_tree(subtree, indent + "  ")

# Example dataset
data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny',
                'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild',
                    'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High',
                 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak',
             'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
                   'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

# Build tree
features = list(data.columns)
features.remove('PlayTennis')
tree = id3(data, 'PlayTennis', features)

# Print tree
print("Decision Tree:")
print_tree(tree)