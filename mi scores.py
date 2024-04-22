from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

# Assuming X is your feature matrix and y is your target variable
# discrete_features is a list of indices of discrete features in X

df = pd.read_csv("new_dataset.csv")

X = df.drop(columns=['label', 'URL_length', 'URL'])

Y = df['label']
Y = pd.DataFrame(Y)

discrete_features = X.dtypes == int
# Calculate MI scores
mi_scores = mutual_info_classif(X, Y, discrete_features=discrete_features)

# Create a pandas Series with feature names as index
mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)

# Sort MI scores in descending order
mi_scores = mi_scores.sort_values(ascending=False)

def plot_mi_scores(scores, filename):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("MI Scores")
    plt.savefig(filename + ".png")  # Save the plot as a PNG file
    plt.close()

# Plot and save the MI scores
plt.figure(dpi=100, figsize=(12, 12))
plot_mi_scores(mi_scores, "mi_scores_plot")

# Save MI scores to a text file
with open("mi_scores.txt", "w") as f:
    f.write(str(mi_scores))

