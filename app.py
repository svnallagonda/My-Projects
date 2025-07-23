from flask import Flask, render_template, request, flash
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/plots"
app.secret_key = "your-secret-key-here"  # Required for flash messages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

def validate_input(education, experience, location, job_title, age, gender):
    """
    Validate user input and return error messages if any
    """
    errors = []
    
    # Validate numeric inputs
    try:
        experience = int(experience)
    except (ValueError, TypeError):
        errors.append("Experience must be a valid number")
        experience = None
    
    try:
        age = int(age)
    except (ValueError, TypeError):
        errors.append("Age must be a valid number")
        age = None
    
    # Check for negative or zero values
    if experience is not None and experience < 0:
        errors.append("Experience must be greater than or equal to 0")
    
    if age is not None and age <= 0:
        errors.append("Age must be greater than 0")
    
    # Check age-experience relationship
    if experience is not None and age is not None:
        # Calculate minimum possible age to start working
        min_working_age = 18  # Minimum age to start working
        expected_min_age = min_working_age + experience
        
        if age < expected_min_age:
            errors.append(f"Invalid age-experience combination. With {experience} years of experience, minimum age should be {expected_min_age} years (assuming work started at age {min_working_age})")
        
        # Check if experience is unreasonably high for the age
        max_possible_experience = age - min_working_age
        if experience > max_possible_experience:
            errors.append(f"Experience cannot exceed {max_possible_experience} years for age {age} (assuming work started at age {min_working_age})")
    
    
    # Check for reasonable limits
    if age is not None and age > 60:
        errors.append("Age must be less than 100")
    
    if experience is not None and experience > 70:
        errors.append("Experience must be less than 70 years")
    
    # Check if required fields are not empty
    if not education or education.strip() == "":
        errors.append("Education field is required")
    
    if not location or location.strip() == "":
        errors.append("Location field is required")
    
    if not job_title or job_title.strip() == "":
        errors.append("Job Title field is required")
    
    if not gender or gender.strip() == "":
        errors.append("Gender field is required")
    
    return errors, experience, age

def train_model():
    global model

    try:
        # Ensure plot directory exists
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        # Load data
        file_path = "salary_prediction_data.csv"
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Dataset file {file_path} not found")
            return None, None
        
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ["Education", "Location", "Job_Title", "Gender", "Experience", "Age", "Salary"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None, None
        
        # Check for empty dataset
        if df.empty:
            logger.error("Dataset is empty")
            return None, None

        X = df.drop("Salary", axis=1)
        y = df["Salary"]

        categorical_features = ["Education", "Location", "Job_Title", "Gender"]
        numerical_features = ["Experience", "Age"]

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
            remainder="passthrough"
        )

        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Check if we have enough data for training
        if len(X_train) < 10:
            logger.error("Not enough data for training")
            return None, None
        
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)

        # Evaluation Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
         # Create evaluation table with test data and predictions
        try:
            # Combine test data with predictions
            evaluation_data = X_test.copy()
            evaluation_data['Actual_Salary'] = y_test.values
            evaluation_data['Predicted_Salary'] = y_pred
            evaluation_data['Prediction_Error'] = y_test.values - y_pred
            evaluation_data['Error_Percentage'] = (abs(evaluation_data['Prediction_Error']) / evaluation_data['Actual_Salary']) * 100
            
            # Round numerical values for better display
            evaluation_data['Actual_Salary'] = evaluation_data['Actual_Salary'].round(0)
            evaluation_data['Predicted_Salary'] = evaluation_data['Predicted_Salary'].round(0)
            evaluation_data['Prediction_Error'] = evaluation_data['Prediction_Error'].round(0)
            evaluation_data['Error_Percentage'] = evaluation_data['Error_Percentage'].round(2)
            
            # Limit to first 20 samples for display
            evaluation_sample = evaluation_data.head(20)
            
            # Convert to list of dictionaries for easy template rendering
            evaluation_table = evaluation_sample.to_dict('records')
            
            # Store in global variable to access in route
            app.config['EVALUATION_TABLE'] = evaluation_table
            
        except Exception as e:
            logger.error(f"Error creating evaluation table: {e}")
            app.config['EVALUATION_TABLE'] = []

        # Plot 1: Actual vs Predicted (Line Plot)
        try:
            plt.figure(figsize=(10, 5))
            sample_size = min(50, len(y_test))
            plt.plot(y_test.values[:sample_size], label='Actual', marker='o')
            plt.plot(y_pred[:sample_size], label='Predicted', marker='x')
            plt.title(f"Actual vs Predicted Salary (Sample {sample_size})")
            plt.xlabel("Sample Index")
            plt.ylabel("Salary")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(app.config["UPLOAD_FOLDER"], "actual_vs_predicted.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating actual vs predicted plot: {e}")

        # Plot 2: Residuals Distribution
        try:
            residuals = y_test - y_pred
            plt.figure(figsize=(8, 5))
            sns.histplot(residuals, kde=True, bins=30)
            plt.title("Distribution of Residuals")
            plt.xlabel("Residuals")
            plt.tight_layout()
            plt.savefig(os.path.join(app.config["UPLOAD_FOLDER"], "residuals.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating residuals plot: {e}")

        # Plot 3: Feature Importance
        try:
            if hasattr(model_pipeline.named_steps["regressor"], "feature_importances_"):
                feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
                importances = model_pipeline.named_steps["regressor"].feature_importances_

                feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

                plt.figure(figsize=(10, 6))
                sns.barplot(x=feat_imp.values, y=feat_imp.index)
                plt.title("Feature Importance")
                plt.tight_layout()
                plt.savefig(os.path.join(app.config["UPLOAD_FOLDER"], "feature_importance.png"))
                plt.close()
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")

        # Plot 4: Scatter Plot of Actual vs Predicted
        try:
            plt.figure(figsize=(6, 6))
            sns.scatterplot(x=y_test, y=y_pred)
            plt.xlabel("Actual Salary")
            plt.ylabel("Predicted Salary")
            plt.title("Scatter Plot: Actual vs Predicted")
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line y=x
            plt.tight_layout()
            plt.savefig(os.path.join(app.config["UPLOAD_FOLDER"], "scatter_plot.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")

        model = model_pipeline
        return mse, r2, app.config.get('EVALUATION_TABLE', [])
    
    except FileNotFoundError:
        logger.error("Dataset file not found")
        return None, None
    except pd.errors.EmptyDataError:
        logger.error("Dataset file is empty")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        return None, None


@app.route("/", methods=["GET", "POST"])
def index():
    predicted_salary = None
    
    try:
        # Try to train the model
        mse, r2, evaluation_table = train_model()
        
        if mse is None or r2 is None:
            flash("Error: Unable to train the model. Please check the dataset.", "error")
            return render_template("index.html", error=True)
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        flash("Error: Unable to train the model. Please try again later.", "error")
        return render_template("index.html", error=True)

    if request.method == "POST":
        try:
            # Get form data
            education = request.form.get("education", "").strip()
            experience = request.form.get("experience", "").strip()
            location = request.form.get("location", "").strip()
            job_title = request.form.get("job_title", "").strip()
            age = request.form.get("age", "").strip()
            gender = request.form.get("gender", "").strip()
            
            # Validate inputs
            errors, validated_experience, validated_age = validate_input(
                education, experience, location, job_title, age, gender
            )
            
            if errors:
                for error in errors:
                    flash(error, "error")
                return render_template(
                    "index.html",
                    mse=round(mse, 2),
                    r2=round(r2, 2),
                    plot_paths=[
                        "static/plots/actual_vs_predicted.png",
                        "static/plots/residuals.png",
                        "static/plots/feature_importance.png",
                        "static/plots/scatter_plot.png"
                    ]
                )
            
            # Create user input dataframe
            user_input = pd.DataFrame([{
                "Education": education,
                "Experience": validated_experience,
                "Location": location,
                "Job_Title": job_title,
                "Age": validated_age,
                "Gender": gender
            }])
            
            # Make prediction
            if model is not None:
                predicted_salary = model.predict(user_input)[0]
                
                # Check for unreasonable predictions
                if predicted_salary < 0:
                    flash("Warning: Predicted salary is negative. Please check your inputs.", "warning")
                elif predicted_salary > 1000000:  # Assuming max reasonable salary
                    flash("Warning: Predicted salary seems unusually high. Please verify your inputs.", "warning")
                
            else:
                flash("Error: Model is not available. Please try again.", "error")
                
        except ValueError as e:
            logger.error(f"Value error during prediction: {e}")
            flash("Error: Invalid input values. Please check your entries.", "error")
        except KeyError as e:
            logger.error(f"Key error during prediction: {e}")
            flash("Error: Missing required form fields.", "error")
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            flash("Error: An unexpected error occurred. Please try again.", "error")

    return render_template(
        "index.html",
        predicted_salary=predicted_salary,
        mse=round(mse, 2) if mse is not None else None,
        r2=round(r2, 2) if r2 is not None else None,
        plot_paths=[
            "static/plots/actual_vs_predicted.png",
            "static/plots/residuals.png",
            "static/plots/feature_importance.png",
            "static/plots/scatter_plot.png"
        ],
        evaluation_table=evaluation_table
    )


if __name__ == "__main__":
    app.run(debug=True)