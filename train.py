import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Load dataset
data = pd.read_csv("Iris.csv")

# Display information
print(data.info())

# Check species distribution
print(data["Species"].value_counts())

# Separate features and target
X = data.drop(columns=["Id", "Species"])
y = data["Species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=200)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="micro")
recall = recall_score(y_test, y_pred, average="micro")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Save trained model
joblib.dump(model, "iris_model.pkl")

print("Model saved as iris_model.pkl")
