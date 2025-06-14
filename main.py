# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('car_data.csv')
print("Initial Data:")
print(df.head())

# Feature engineering
df['car_age'] = 2025 - df['year']
df.drop(['year', 'name'], axis=1, inplace=True)

# Encode categorical features
le_fuel = LabelEncoder()
le_trans = LabelEncoder()
le_owner = LabelEncoder()

df['fuel'] = le_fuel.fit_transform(df['fuel'])
df['transmission'] = le_trans.fit_transform(df['transmission'])
df['owner'] = le_owner.fit_transform(df['owner'])

print("\nProcessed Data:")
print(df.head())

# Split dataset
X = df.drop('selling_price', axis=1)
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")

# Optional: Predict with saved model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

sample_input = np.array([[70000, 0, 1, 0, 11]])  # kms_driven, fuel, transmission, owner, car_age
predicted_price = loaded_model.predict(sample_input)
print("\nPredicted price for sample input:", predicted_price[0])