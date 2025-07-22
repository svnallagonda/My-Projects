import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
file_path = "/Users/snigdha/Downloads/salary_prediction_data (1).csv"  # Update path if needed
df = pd.read_csv(file_path)

# Exclude the last column (typically the target column)
columns_to_analyze = df.columns[:-1]  # All except last

# Iterate over selected columns
for column in columns_to_analyze:
    print(f"\n--- Column: {column} ---")
    
    value_counts = df[column].value_counts(dropna=False)
    
    for value, count in value_counts.items():
        print(f"{repr(value)}: {count} times")

# Step 2: Split features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Step 3: Define categorical and numerical columns
categorical_features = ["Education", "Location", "Job_Title", "Gender"]
numerical_features = ["Experience", "Age"]

# Step 4: Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"  # keep numerical features as is
)

# Step 5: Create the model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Step 6: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train the model
model.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 10: Output the evaluation
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# Step 10: User Input for Prediction
print("\n--- Salary Prediction Based on User Input ---")
education = input("Enter Education (e.g., High School, Bachelor, Master, PhD): ")
experience = int(input("Enter Years of Experience: "))
location = input("Enter Location (e.g., Urban, Suburban, Rural): ")
job_title = input("Enter Job Title (e.g., Manager, Director, Analyst): ")
age = int(input("Enter Age: "))
gender = input("Enter Gender (Male/Female): ")

# Create a DataFrame from user input
user_data = pd.DataFrame([{
    "Education": education,
    "Experience": experience,
    "Location": location,
    "Job_Title": job_title,
    "Age": age,
    "Gender": gender
}])

# Predict salary
predicted_salary = model.predict(user_data)[0]
print(f"\n✅ Predicted Salary: ₹{predicted_salary:,.2f}")