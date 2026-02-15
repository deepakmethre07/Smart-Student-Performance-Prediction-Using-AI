"""
Simple script to predict performance for new students
Load the trained model and make predictions
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "=" * 60)
print("STUDENT PERFORMANCE PREDICTION TOOL")
print("=" * 60 + "\n")

# Load the dataset
data = pd.read_csv('student_dataset.csv')

# Prepare features and target
X = data[['Attendance (%)', 'Study Hours', 'Previous Marks', 'Assignment Scores']]
y = data['Final Marks']

# Train the model (in production, you would save/load the model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print("✓ Model loaded and ready for predictions!\n")

# ============================================================================
# INTERACTIVE PREDICTION FUNCTION
# ============================================================================
def predict_student_performance(attendance, study_hours, previous_marks, assignment_scores):
    """
    Predict student's final marks based on input features
    
    Parameters:
    - attendance: Attendance percentage (0-100)
    - study_hours: Daily study hours (0-10)
    - previous_marks: Marks from previous exam (0-100)
    - assignment_scores: Assignment scores (0-100)
    
    Returns:
    - Predicted final marks
    """
    # Create input array
    input_data = np.array([[attendance, study_hours, previous_marks, assignment_scores]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Ensure prediction is within valid range
    prediction = max(0, min(100, prediction))
    
    return prediction

# ============================================================================
# EXAMPLE PREDICTIONS
# ============================================================================
print("=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60 + "\n")

# Sample students
sample_students = [
    {"name": "Excellent Student", "attendance": 95, "study_hours": 8.5, 
     "previous_marks": 88, "assignment_scores": 92},
    {"name": "Good Student", "attendance": 85, "study_hours": 6.0, 
     "previous_marks": 75, "assignment_scores": 80},
    {"name": "Average Student", "attendance": 70, "study_hours": 4.0, 
     "previous_marks": 65, "assignment_scores": 70},
    {"name": "Needs Improvement", "attendance": 55, "study_hours": 2.5, 
     "previous_marks": 50, "assignment_scores": 60},
]

print(f"{'Student Type':<20} {'Attendance':<12} {'Study Hrs':<12} {'Prev Marks':<12} {'Assignments':<12} {'Predicted':<12}")
print("-" * 90)

for student in sample_students:
    predicted_marks = predict_student_performance(
        student['attendance'],
        student['study_hours'],
        student['previous_marks'],
        student['assignment_scores']
    )
    
    print(f"{student['name']:<20} {student['attendance']:<12} {student['study_hours']:<12.1f} "
          f"{student['previous_marks']:<12} {student['assignment_scores']:<12} {predicted_marks:<12.2f}")

print("\n" + "=" * 60)

# ============================================================================
# PERFORMANCE GRADE FUNCTION
# ============================================================================
def get_grade(marks):
    """Convert marks to letter grade"""
    if marks >= 90:
        return "A+ (Excellent)"
    elif marks >= 80:
        return "A (Very Good)"
    elif marks >= 70:
        return "B (Good)"
    elif marks >= 60:
        return "C (Average)"
    elif marks >= 50:
        return "D (Below Average)"
    else:
        return "F (Fail)"

# ============================================================================
# INTERACTIVE MODE (OPTIONAL)
# ============================================================================
print("\nWOULD YOU LIKE TO PREDICT FOR A CUSTOM STUDENT?")
print("-" * 60)

try:
    response = input("Enter 'yes' to continue or press Enter to skip: ").strip().lower()
    
    if response in ['yes', 'y']:
        print("\nEnter student details:")
        attendance = float(input("Attendance (%) [0-100]: "))
        study_hours = float(input("Daily study hours [0-10]: "))
        previous_marks = float(input("Previous exam marks [0-100]: "))
        assignment_scores = float(input("Assignment scores [0-100]: "))
        
        predicted_marks = predict_student_performance(
            attendance, study_hours, previous_marks, assignment_scores
        )
        
        grade = get_grade(predicted_marks)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"\n📊 Predicted Final Marks: {predicted_marks:.2f}/100")
        print(f"🎓 Expected Grade: {grade}")
        print("\n" + "=" * 60)
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if predicted_marks < 60:
            print("  ⚠️  Student needs significant improvement!")
            if attendance < 75:
                print("  • Improve attendance (currently {:.0f}%)".format(attendance))
            if study_hours < 4:
                print("  • Increase study hours (currently {:.1f} hrs/day)".format(study_hours))
            print("  • Consider extra tutoring or study groups")
        elif predicted_marks < 75:
            print("  ⚡ Student is performing adequately but can improve!")
            print("  • Maintain consistent study habits")
            print("  • Focus on weak subjects")
        else:
            print("  ✅ Student is performing well!")
            print("  • Keep up the good work")
            print("  • Challenge yourself with advanced topics")
        
        print("\n" + "=" * 60 + "\n")
    else:
        print("\nPrediction skipped. Run the script again to make predictions.\n")
        
except (ValueError, KeyboardInterrupt):
    print("\n\nPrediction cancelled or invalid input.\n")

print("Thank you for using the Student Performance Prediction Tool! 🎓\n")
