from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

# Load dataset
data = pd.read_csv("asl_dataset.csv")  # Replace with your dataset
X = data.iloc[:, :-1].values  # Landmarks
y = data.iloc[:, -1].values   # Labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open("asl_hand_model.pkl", "wb") as file:
    pickle.dump(model, file)