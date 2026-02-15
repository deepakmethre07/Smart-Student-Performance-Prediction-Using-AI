"""
Smart Student Performance Prediction Using AI
A beginner-friendly machine learning project to predict student performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("SMART STUDENT PERFORMANCE PREDICTION USING AI")
print("=" * 60)
print()

# ============================================================================
# STEP 1: CREATE STUDENT DATASET
# ============================================================================
print("Step 1: Creating Student Dataset...")
print("-" * 60)

# Generate realistic student data
num_students = 200

# Create features
attendance = np.random.randint(40, 100, num_students)  # Attendance percentage (40-100%)
study_hours = np.random.uniform(1, 10, num_students)   # Daily study hours (1-10 hours)
previous_marks = np.random.randint(30, 95, num_students)  # Previous exam marks (30-95)
assignment_scores = np.random.randint(40, 100, num_students)  # Assignment scores (40-100)

# Generate final marks based on weighted features with some randomness
# Formula: 30% attendance + 25% study_hours + 30% previous_marks + 15% assignments
final_marks = (
    0.30 * attendance +
    0.25 * (study_hours * 10) +  # Scale study hours to 0-100
    0.30 * previous_marks +
    0.15 * assignment_scores +
    np.random.normal(0, 5, num_students)  # Add some random noise
)

# Ensure marks are within 0-100 range
final_marks = np.clip(final_marks, 0, 100)

# Create DataFrame
data = pd.DataFrame({
    'Attendance (%)': attendance,
    'Study Hours': np.round(study_hours, 2),
    'Previous Marks': previous_marks,
    'Assignment Scores': assignment_scores,
    'Final Marks': np.round(final_marks, 2)
})

# Add student names
data.insert(0, 'Student ID', [f'STU{i+1:03d}' for i in range(num_students)])

print(f"✓ Dataset created with {num_students} students")
print(f"\nFirst 5 records:")
print(data.head())
print(f"\nDataset shape: {data.shape}")
print()

# ============================================================================
# STEP 2: DATA PREPROCESSING AND CLEANING
# ============================================================================
print("Step 2: Data Preprocessing and Cleaning...")
print("-" * 60)

# Check for missing values
print(f"Missing values per column:")
print(data.isnull().sum())

# Statistical summary
print(f"\nStatistical Summary:")
print(data.describe().round(2))

# Check for duplicates
duplicates = data.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

print("✓ Data is clean and ready for modeling")
print()

# ============================================================================
# STEP 3: PREPARE DATA FOR MACHINE LEARNING
# ============================================================================
print("Step 3: Preparing Data for Machine Learning...")
print("-" * 60)

# Separate features (X) and target variable (y)
X = data[['Attendance (%)', 'Study Hours', 'Previous Marks', 'Assignment Scores']]
y = data['Final Marks']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeature columns:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")
print()

# ============================================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================================
print("Step 4: Splitting Data into Training and Testing Sets...")
print("-" * 60)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)} students ({len(X_train)/len(X)*100:.0f}%)")
print(f"Testing set size: {len(X_test)} students ({len(X_test)/len(X)*100:.0f}%)")
print()

# ============================================================================
# STEP 5: TRAIN MACHINE LEARNING MODELS
# ============================================================================
print("Step 5: Training Machine Learning Models...")
print("-" * 60)

# Model 1: Linear Regression
print("\n[Model 1] Training Linear Regression Model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("✓ Linear Regression model trained successfully")

# Model 2: Decision Tree Regressor
print("\n[Model 2] Training Decision Tree Model...")
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
print("✓ Decision Tree model trained successfully")
print()

# ============================================================================
# STEP 6: EVALUATE MODEL ACCURACY
# ============================================================================
print("Step 6: Evaluating Model Performance...")
print("-" * 60)

# Make predictions
lr_predictions = lr_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)

# Calculate metrics for Linear Regression
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

# Calculate metrics for Decision Tree
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = np.sqrt(dt_mse)
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)

print("\n📊 LINEAR REGRESSION RESULTS:")
print(f"  • R² Score (Accuracy): {lr_r2*100:.2f}%")
print(f"  • Mean Absolute Error: {lr_mae:.2f} marks")
print(f"  • Root Mean Squared Error: {lr_rmse:.2f} marks")

print("\n📊 DECISION TREE RESULTS:")
print(f"  • R² Score (Accuracy): {dt_r2*100:.2f}%")
print(f"  • Mean Absolute Error: {dt_mae:.2f} marks")
print(f"  • Root Mean Squared Error: {dt_rmse:.2f} marks")

# Determine best model
if lr_r2 > dt_r2:
    best_model = lr_model
    best_model_name = "Linear Regression"
    best_accuracy = lr_r2
else:
    best_model = dt_model
    best_model_name = "Decision Tree"
    best_accuracy = dt_r2

print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy*100:.2f}% accuracy")
print()

# Display feature importance for Linear Regression
print("Feature Importance (Linear Regression Coefficients):")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"  • {feature}: {coef:.4f}")
print()

# ============================================================================
# STEP 7: PREDICT FOR NEW STUDENTS
# ============================================================================
print("Step 7: Making Predictions for New Students...")
print("-" * 60)

# Create sample new students
new_students = pd.DataFrame({
    'Attendance (%)': [95, 75, 60, 85],
    'Study Hours': [8.5, 5.0, 3.0, 6.5],
    'Previous Marks': [88, 70, 55, 80],
    'Assignment Scores': [92, 78, 65, 85]
})

# Make predictions using the best model
new_predictions = best_model.predict(new_students)

print(f"\nPredictions using {best_model_name}:\n")
print(f"{'Student':<10} {'Attendance':<12} {'Study Hrs':<12} {'Prev Marks':<12} {'Assign Score':<14} {'Predicted Marks':<15}")
print("-" * 80)

for i in range(len(new_students)):
    print(f"Student {i+1:<3} {new_students.iloc[i, 0]:<12} {new_students.iloc[i, 1]:<12.1f} "
          f"{new_students.iloc[i, 2]:<12} {new_students.iloc[i, 3]:<14} {new_predictions[i]:<15.2f}")

print()

# ============================================================================
# STEP 8: VISUALIZE RESULTS
# ============================================================================
print("Step 8: Creating Visualizations...")
print("-" * 60)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Smart Student Performance Prediction - Analysis Dashboard', 
             fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted (Linear Regression)
ax1 = plt.subplot(2, 3, 1)
ax1.scatter(y_test, lr_predictions, alpha=0.6, color='blue', edgecolors='black')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Marks', fontweight='bold')
ax1.set_ylabel('Predicted Marks', fontweight='bold')
ax1.set_title(f'Linear Regression\nAccuracy: {lr_r2*100:.2f}%', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted (Decision Tree)
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(y_test, dt_predictions, alpha=0.6, color='green', edgecolors='black')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Marks', fontweight='bold')
ax2.set_ylabel('Predicted Marks', fontweight='bold')
ax2.set_title(f'Decision Tree\nAccuracy: {dt_r2*100:.2f}%', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Model Comparison
ax3 = plt.subplot(2, 3, 3)
models = ['Linear\nRegression', 'Decision\nTree']
accuracies = [lr_r2 * 100, dt_r2 * 100]
colors = ['#3498db', '#2ecc71']
bars = ax3.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Accuracy (%)', fontweight='bold')
ax3.set_title('Model Comparison', fontweight='bold')
ax3.set_ylim([0, 100])
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

# Plot 4: Residual Plot (Linear Regression)
ax4 = plt.subplot(2, 3, 4)
residuals_lr = y_test - lr_predictions
ax4.scatter(lr_predictions, residuals_lr, alpha=0.6, color='purple', edgecolors='black')
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Marks', fontweight='bold')
ax4.set_ylabel('Residuals', fontweight='bold')
ax4.set_title('Residual Plot (Linear Regression)', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Feature Correlation Heatmap
ax5 = plt.subplot(2, 3, 5)
correlation_matrix = data[['Attendance (%)', 'Study Hours', 'Previous Marks', 
                            'Assignment Scores', 'Final Marks']].corr()
im = ax5.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax5.set_xticks(range(len(correlation_matrix.columns)))
ax5.set_yticks(range(len(correlation_matrix.columns)))
ax5.set_xticklabels(['Attend', 'Study', 'Prev', 'Assign', 'Final'], 
                     rotation=45, ha='right')
ax5.set_yticklabels(['Attend', 'Study', 'Prev', 'Assign', 'Final'])
ax5.set_title('Feature Correlation Matrix', fontweight='bold')
plt.colorbar(im, ax=ax5)

# Add correlation values
for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        text = ax5.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

# Plot 6: Prediction Distribution
ax6 = plt.subplot(2, 3, 6)
ax6.hist(y_test, bins=15, alpha=0.5, label='Actual', color='blue', edgecolor='black')
ax6.hist(lr_predictions, bins=15, alpha=0.5, label='Predicted (LR)', 
         color='orange', edgecolor='black')
ax6.set_xlabel('Marks', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.set_title('Actual vs Predicted Distribution', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/student_performance_ml/prediction_results.png', 
            dpi=300, bbox_inches='tight')
print("✓ Graphs saved as 'prediction_results.png'")

# Additional visualization: Feature Importance
fig2 = plt.figure(figsize=(10, 6))
features = X.columns
importance = np.abs(lr_model.coef_)
indices = np.argsort(importance)[::-1]

plt.bar(range(len(importance)), importance[indices], color='teal', 
        edgecolor='black', linewidth=1.5)
plt.xticks(range(len(importance)), [features[i] for i in indices], rotation=45, ha='right')
plt.xlabel('Features', fontweight='bold', fontsize=12)
plt.ylabel('Importance (Coefficient Magnitude)', fontweight='bold', fontsize=12)
plt.title('Feature Importance in Student Performance Prediction', 
          fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/home/claude/student_performance_ml/feature_importance.png', 
            dpi=300, bbox_inches='tight')
print("✓ Feature importance graph saved as 'feature_importance.png'")
print()

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================
print("Step 9: Saving Results...")
print("-" * 60)

# Save dataset
data.to_csv('/home/claude/student_performance_ml/student_dataset.csv', index=False)
print("✓ Dataset saved as 'student_dataset.csv'")

# Save predictions
results_df = pd.DataFrame({
    'Actual Marks': y_test.values,
    'Linear Regression Prediction': lr_predictions,
    'Decision Tree Prediction': dt_predictions,
    'LR Error': np.abs(y_test.values - lr_predictions),
    'DT Error': np.abs(y_test.values - dt_predictions)
})
results_df.to_csv('/home/claude/student_performance_ml/test_predictions.csv', index=False)
print("✓ Test predictions saved as 'test_predictions.csv'")

# Create summary report
with open('/home/claude/student_performance_ml/model_report.txt', 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("SMART STUDENT PERFORMANCE PREDICTION - MODEL REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("PROJECT OVERVIEW:\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Students: {len(data)}\n")
    f.write(f"Training Set: {len(X_train)} students (80%)\n")
    f.write(f"Testing Set: {len(X_test)} students (20%)\n")
    f.write(f"Features Used: {', '.join(X.columns)}\n\n")
    
    f.write("MODEL PERFORMANCE:\n")
    f.write("-" * 70 + "\n")
    f.write("\nLinear Regression:\n")
    f.write(f"  R² Score: {lr_r2*100:.2f}%\n")
    f.write(f"  Mean Absolute Error: {lr_mae:.2f} marks\n")
    f.write(f"  Root Mean Squared Error: {lr_rmse:.2f} marks\n")
    
    f.write("\nDecision Tree:\n")
    f.write(f"  R² Score: {dt_r2*100:.2f}%\n")
    f.write(f"  Mean Absolute Error: {dt_mae:.2f} marks\n")
    f.write(f"  Root Mean Squared Error: {dt_rmse:.2f} marks\n")
    
    f.write(f"\nBest Model: {best_model_name}\n")
    f.write(f"Best Accuracy: {best_accuracy*100:.2f}%\n\n")
    
    f.write("FEATURE IMPORTANCE:\n")
    f.write("-" * 70 + "\n")
    for feature, coef in zip(X.columns, lr_model.coef_):
        f.write(f"{feature}: {coef:.4f}\n")
    
    f.write("\n" + "=" * 70 + "\n")

print("✓ Model report saved as 'model_report.txt'")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY! 🎉")
print("=" * 60)
print("\n📁 Generated Files:")
print("  1. student_dataset.csv - Complete student dataset")
print("  2. test_predictions.csv - Model predictions on test data")
print("  3. prediction_results.png - Visualization dashboard")
print("  4. feature_importance.png - Feature importance chart")
print("  5. model_report.txt - Detailed model report")
print("\n💡 Key Insights:")
print(f"  • Best performing model: {best_model_name}")
print(f"  • Model accuracy: {best_accuracy*100:.2f}%")
print(f"  • Average prediction error: {lr_mae:.2f} marks")
print("\n✅ The model can now predict student performance based on:")
print("  • Attendance percentage")
print("  • Daily study hours")
print("  • Previous exam marks")
print("  • Assignment scores")
print("\n" + "=" * 60)
