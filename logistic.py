# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, :2]  # Select only the first two features (sepal length and sepal width)
y = (iris.target == 0).astype(int)  # Binary classification: Setosa vs Non-setosa

# Introduce noise to the features
noise = np.random.normal(0, 0.5, X.shape)  # Gaussian noise with mean=0 and std=0.5
X_noisy = X + noise  # Add noise to the features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

# Create and train a Logistic Regression model
log_reg = LogisticRegression(solver='liblinear')  # Use liblinear solver for small datasets
log_reg.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_reg.predict(X_test)

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot the data points
plt.figure(figsize=(10, 6))
plt.scatter(X_noisy[:, 0], X_noisy[:, 1], c=y, cmap=plt.cm.coolwarm, marker='o', edgecolor='k', s=50)

# Create a grid to plot the decision boundary
x_min, x_max = X_noisy[:, 0].min() - 1, X_noisy[:, 0].max() + 1
y_min, y_max = X_noisy[:, 1].min() - 1, X_noisy[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict on the grid points
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# Labels and title
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Logistic Regression Decision Boundary')
plt.show()
