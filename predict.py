import joblib

# Load the trained model
model = joblib.load("iris_model.pkl")

# Example input:
# [Sepal Length, Sepal Width, Petal Length, Petal Width]
sample = [[5.1, 3.5, 1.4, 0.2]]

# Make prediction
prediction = model.predict(sample)

print("Predicted Species:", prediction[0])
