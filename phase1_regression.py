import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("Starting Phase 1: Training Numerical Regression Model...\n")

# 1. Load the generated dataset
try:
    df = pd.read_csv('student_performance_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find 'student_performance_data.csv'. Make sure you ran generate_data.py first!")
    exit()

# 2. Select Features (X) and Target (y)
# We only want the numeric inputs for this specific phase
features = ['Hours_Studied', 'Attendance_Pct', 'Past_Grade']
X = df[features]
y = df['Final_Score']

# 3. Train/Test Split
# We keep 80% of data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

# 4. Feature Scaling (Standardization)
# This ensures all features have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Notice we only transform the test set to prevent data leakage!

# 5. Initialize and Train the Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 6. Make Predictions on the unseen Test Set
y_pred = model.predict(X_test_scaled)

# 7. Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} points")
print(f"R-squared (R2) Score:      {r2:.4f} (1.0 is perfect)")

# 8. Let's look inside the model to see what it learned
print("\n--- Feature Importance (Coefficients) ---")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nPhase 1 Complete! The model successfully learned the mathematical relationship between the features and the final score.")