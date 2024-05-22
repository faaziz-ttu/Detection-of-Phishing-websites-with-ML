import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("new_dataset.csv")

# Drop unnecessary columns, experiment 1
data = df.drop(columns=['URL_length', 'URL', 'containing_shortUrl', 'using_ip_address'])

# Function to plot correlation heatmap
def corr_heatmap(data, idx_s, idx_e):
    y = data['label']
    temp = data.iloc[:, idx_s:idx_e]
    if 'id' in temp.columns:
        del temp['id']
    temp['label'] = y
    sns.heatmap(temp.corr(), annot=True, fmt='.2f')
    plt.savefig('exp3.png')  # Save the plot as an image
    plt.show()

# Plot correlation heatmap and save it as an image
corr_heatmap(data, 0, 10)

# Split data into features and target
X = df.drop(columns=['label', 'URL_length', 'URL', 'containing_shortUrl', 'using_ip_address'])
Y = df['label']

# Split data into train and test sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=2)

# Logistic Regression (model_1)
logreg = LogisticRegression()
model_1 = logreg.fit(train_X, train_Y)
logreg_predict = model_1.predict(test_X)
print('Logistic Regression Accuracy:', accuracy_score(logreg_predict, test_Y))
print(classification_report(logreg_predict, test_Y))

# Random Forest Classifier (model_2)
rfc = RandomForestClassifier()
model_2 = rfc.fit(train_X, train_Y)
rfc_predict = model_2.predict(test_X)
print('Random Forest Classifier Accuracy:', accuracy_score(rfc_predict, test_Y))
print(classification_report(rfc_predict, test_Y))

# Decision Tree Classifier (model_3)
dtree = DecisionTreeClassifier()
model_3 = dtree.fit(train_X, train_Y)
dtree_predict = model_3.predict(test_X)
print('Decision Tree Classifier Accuracy:', accuracy_score(dtree_predict, test_Y))
print(classification_report(dtree_predict, test_Y))

# Support Vector Machine (model_4)
svc = SVC()
model_4 = svc.fit(train_X, train_Y)
svm_predict = model_4.predict(test_X)
print('Support Vector Machine Accuracy:', accuracy_score(svm_predict, test_Y))
print(classification_report(svm_predict, test_Y))

# Adaboost Classifier (model_5)
adc = AdaBoostClassifier(n_estimators=5, learning_rate=1)
model_5 = adc.fit(train_X, train_Y)
adc_predict = model_5.predict(test_X)
print('Adaboost Classifier Accuracy:', accuracy_score(adc_predict, test_Y))
print(classification_report(adc_predict, test_Y))

# Gradient Boosting Classifier (model_6)
clf = GradientBoostingClassifier(n_estimators=300, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(train_X, train_Y)
gb_predict = clf.predict(test_X)
print('Gradient Boosting Classifier Accuracy:', accuracy_score(gb_predict, test_Y))
print(classification_report(gb_predict, test_Y))

# K-Nearest Neighbors Classifier (model_7)
knn = KNeighborsClassifier(n_neighbors=3)
model_7 = knn.fit(train_X, train_Y)
knn_predict = model_7.predict(test_X)
print('K-Nearest Neighbour Accuracy:', accuracy_score(knn_predict, test_Y))
print(classification_report(knn_predict, test_Y))

# Calculate precision scores for each model
precision_scores = {
    'Logistic Regression': precision_score(test_Y, logreg_predict),
    'Random Forest Classifier': precision_score(test_Y, rfc_predict),
    'Decision Tree Classifier': precision_score(test_Y, dtree_predict),
    'Support Vector Machine': precision_score(test_Y, svm_predict),
    'Adaboost Classifier': precision_score(test_Y, adc_predict),
    'Gradient Boosting Classifier': precision_score(test_Y, gb_predict),
    'K-Nearest Neighbors Classifier': precision_score(test_Y, knn_predict)
}

# Create a DataFrame from precision scores
precision_df = pd.DataFrame(list(precision_scores.items()), columns=['Model', 'Precision Score'])

# Plot precision scores
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Precision Score', data=precision_df)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Machine Learning Model')
plt.ylabel('Precision Score')
plt.title('Precision Scores of Machine Learning Models')
plt.show()
