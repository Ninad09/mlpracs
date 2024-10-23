# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Load the Iris dataset
data = load_iris()
# print(dir(data))
print(dir(load_iris()))
print(data['feature_names'])

X = data['data']
y = data['target']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create a Decision Tree Classifier using ID3 (criterion='entropy')
id3_clf = DecisionTreeClassifier(criterion='gini', random_state=42)

# Train the classifier on the training data
id3_clf.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(10,8))  # Set plot size for better visibility
plot_tree(id3_clf, 
          feature_names=data.feature_names,  
          class_names=data.target_names, 
          filled=True, 
          rounded=True,
          proportion=True,
        )

# Add a title to the plot
plt.title("Decision Tree using ID3 Algorithm (Entropy as Split Criterion)")
plt.show()
