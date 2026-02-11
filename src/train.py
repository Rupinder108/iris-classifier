import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
print("Features:", iris.feature_names)
print("Target names:", iris.target_names)
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)

print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)
# Train KNN
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)
print("k-NN Accuracy:", accuracy_score(y_test, y_pred2))
# Confusion Matrix (using Decision Tree predictions)
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.close()
# Save trained Decision Tree model
joblib.dump(model, "outputs/model.joblib")
print("Confusion matrix saved to outputs/confusion_matrix.png")
print("Model saved to outputs/model.joblib")

import joblib
import numpy as np

# Verify saved model
loaded_model = joblib.load("outputs/model.joblib")

# Make a sample prediction (using first row of test set as example)
sample_input = X_test[0].reshape(1, -1)  # reshape to 2D
sample_prediction = loaded_model.predict(sample_input)

print("\n Model verification:")
print("Sample input:", sample_input)
print("Prediction from loaded model:", sample_prediction)
print("True label:", y_test[0])
