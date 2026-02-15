"""
Smart Student Performance Prediction - Live Web Application
Interactive ML-powered web app built with Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2ca02c;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'lr_model' not in st.session_state:
    st.session_state.lr_model = None
if 'dt_model' not in st.session_state:
    st.session_state.dt_model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ============================================================================
# FUNCTIONS
# ============================================================================

@st.cache_data
def generate_dataset(num_students=200):
    """Generate synthetic student dataset"""
    np.random.seed(42)
    
    attendance = np.random.randint(40, 100, num_students)
    study_hours = np.random.uniform(1, 10, num_students)
    previous_marks = np.random.randint(30, 95, num_students)
    assignment_scores = np.random.randint(40, 100, num_students)
    
    final_marks = (
        0.30 * attendance +
        0.25 * (study_hours * 10) +
        0.30 * previous_marks +
        0.15 * assignment_scores +
        np.random.normal(0, 5, num_students)
    )
    
    final_marks = np.clip(final_marks, 0, 100)
    
    data = pd.DataFrame({
        'Student ID': [f'STU{i+1:03d}' for i in range(num_students)],
        'Attendance (%)': attendance,
        'Study Hours': np.round(study_hours, 2),
        'Previous Marks': previous_marks,
        'Assignment Scores': assignment_scores,
        'Final Marks': np.round(final_marks, 2)
    })
    
    return data

def train_models(data):
    """Train both ML models"""
    X = data[['Attendance (%)', 'Study Hours', 'Previous Marks', 'Assignment Scores']]
    y = data['Final Marks']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    
    # Decision Tree
    dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_r2 = r2_score(y_test, dt_pred)
    dt_mae = mean_absolute_error(y_test, dt_pred)
    
    return {
        'lr_model': lr_model,
        'dt_model': dt_model,
        'X_test': X_test,
        'y_test': y_test,
        'lr_pred': lr_pred,
        'dt_pred': dt_pred,
        'lr_r2': lr_r2,
        'lr_mae': lr_mae,
        'dt_r2': dt_r2,
        'dt_mae': dt_mae
    }

def get_grade(marks):
    """Convert marks to letter grade"""
    if marks >= 90:
        return "A+", "🌟 Excellent"
    elif marks >= 80:
        return "A", "✨ Very Good"
    elif marks >= 70:
        return "B", "👍 Good"
    elif marks >= 60:
        return "C", "📚 Average"
    elif marks >= 50:
        return "D", "⚠️ Below Average"
    else:
        return "F", "❌ Fail"

def get_recommendations(attendance, study_hours, predicted_marks):
    """Generate personalized recommendations"""
    recommendations = []
    
    if predicted_marks < 60:
        recommendations.append("⚠️ **Immediate Action Required!** Student needs significant improvement.")
    
    if attendance < 75:
        recommendations.append(f"📅 **Improve Attendance**: Currently at {attendance}%. Aim for 85%+.")
    elif attendance >= 90:
        recommendations.append(f"✅ **Excellent Attendance**: Keep it up! ({attendance}%)")
    
    if study_hours < 4:
        recommendations.append(f"📖 **Increase Study Time**: Currently {study_hours:.1f} hrs/day. Aim for 5-6 hours.")
    elif study_hours >= 7:
        recommendations.append(f"✅ **Great Study Habits**: {study_hours:.1f} hrs/day is excellent!")
    
    if predicted_marks >= 75:
        recommendations.append("🎯 **Challenge Yourself**: Try advanced topics and competitions.")
    elif predicted_marks >= 60:
        recommendations.append("📈 **Keep Improving**: Focus on consistency and weak subjects.")
    else:
        recommendations.append("💪 **Extra Support Needed**: Consider tutoring or study groups.")
    
    return recommendations

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.title("🎓 Smart Student Performance Prediction System")
st.markdown("### AI-Powered ML Application for Educational Analytics")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Dataset size selector
    num_students = st.slider(
        "Dataset Size",
        min_value=100,
        max_value=500,
        value=200,
        step=50,
        help="Number of students in the training dataset"
    )
    
    # Model selector
    model_choice = st.radio(
        "Select Model",
        ["Linear Regression", "Decision Tree", "Both"],
        help="Choose which ML model to use for predictions"
    )
    
    st.markdown("---")
    
    # Train model button
    if st.button("🚀 Train/Retrain Models", type="primary", use_container_width=True):
        with st.spinner("Training models..."):
            st.session_state.data = generate_dataset(num_students)
            results = train_models(st.session_state.data)
            
            st.session_state.lr_model = results['lr_model']
            st.session_state.dt_model = results['dt_model']
            st.session_state.model_results = results
            st.session_state.model_trained = True
            
            st.success("✅ Models trained successfully!")
            st.balloons()
    
    st.markdown("---")
    
    # Info section
    st.info("""
    **How to use:**
    1. Train the models
    2. Enter student details
    3. Get instant predictions
    4. View analytics
    """)
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Make Prediction", 
    "📊 Model Performance", 
    "📈 Data Analytics",
    "📚 Prediction History"
])

# ============================================================================
# TAB 1: MAKE PREDICTION
# ============================================================================
with tab1:
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first using the sidebar!")
    else:
        st.header("Enter Student Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Academic Factors")
            attendance = st.slider(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                value=75,
                help="Student's class attendance percentage"
            )
            
            previous_marks = st.slider(
                "Previous Exam Marks",
                min_value=0,
                max_value=100,
                value=70,
                help="Marks from the previous examination"
            )
        
        with col2:
            st.subheader("Study Metrics")
            study_hours = st.slider(
                "Daily Study Hours",
                min_value=0.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                help="Average daily study hours"
            )
            
            assignment_scores = st.slider(
                "Assignment Scores",
                min_value=0,
                max_value=100,
                value=75,
                help="Average assignment scores"
            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
            # Prepare input
            input_data = pd.DataFrame({
                'Attendance (%)': [attendance],
                'Study Hours': [study_hours],
                'Previous Marks': [previous_marks],
                'Assignment Scores': [assignment_scores]
            })
            
            # Make predictions
            lr_prediction = st.session_state.lr_model.predict(input_data)[0]
            dt_prediction = st.session_state.dt_model.predict(input_data)[0]
            
            # Clip predictions
            lr_prediction = max(0, min(100, lr_prediction))
            dt_prediction = max(0, min(100, dt_prediction))
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            st.markdown("### 🎯 Prediction Results")
            
            # Create columns for results
            if model_choice == "Both":
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Linear Regression")
                    grade, status = get_grade(lr_prediction)
                    st.metric("Predicted Marks", f"{lr_prediction:.2f}/100")
                    st.metric("Grade", f"{grade} - {status}")
                
                with col2:
                    st.markdown("#### Decision Tree")
                    grade, status = get_grade(dt_prediction)
                    st.metric("Predicted Marks", f"{dt_prediction:.2f}/100")
                    st.metric("Grade", f"{grade} - {status}")
                
                # Use average for recommendations
                avg_prediction = (lr_prediction + dt_prediction) / 2
            else:
                prediction = lr_prediction if model_choice == "Linear Regression" else dt_prediction
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Final Marks", f"{prediction:.2f}/100")
                
                with col2:
                    grade, status = get_grade(prediction)
                    st.metric("Expected Grade", grade)
                
                with col3:
                    st.metric("Status", status)
                
                avg_prediction = prediction
            
            # Visualization
            st.markdown("### 📊 Performance Visualization")
            
            fig = go.Figure()
            
            # Add gauge chart
            fig.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = avg_prediction,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Predicted Performance", 'font': {'size': 24}},
                delta = {'reference': 60, 'increasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#ff6b6b'},
                        {'range': [50, 60], 'color': '#ffd93d'},
                        {'range': [60, 70], 'color': '#95e1d3'},
                        {'range': [70, 80], 'color': '#6bcf7f'},
                        {'range': [80, 100], 'color': '#4caf50'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("### 💡 Personalized Recommendations")
            recommendations = get_recommendations(attendance, study_hours, avg_prediction)
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Add to history
            st.session_state.prediction_history.append({
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Attendance': attendance,
                'Study Hours': study_hours,
                'Previous Marks': previous_marks,
                'Assignments': assignment_scores,
                'Predicted Marks': round(avg_prediction, 2),
                'Grade': get_grade(avg_prediction)[0]
            })

# ============================================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================================
with tab2:
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first!")
    else:
        st.header("Model Performance Metrics")
        
        results = st.session_state.model_results
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Linear Regression")
            st.metric("R² Score (Accuracy)", f"{results['lr_r2']*100:.2f}%")
            st.metric("Mean Absolute Error", f"{results['lr_mae']:.2f} marks")
            
        with col2:
            st.markdown("### 🌳 Decision Tree")
            st.metric("R² Score (Accuracy)", f"{results['dt_r2']*100:.2f}%")
            st.metric("Mean Absolute Error", f"{results['dt_mae']:.2f} marks")
        
        st.markdown("---")
        
        # Comparison chart
        st.subheader("📊 Model Comparison")
        
        comparison_data = pd.DataFrame({
            'Model': ['Linear Regression', 'Decision Tree'],
            'Accuracy (%)': [results['lr_r2']*100, results['dt_r2']*100],
            'Error (marks)': [results['lr_mae'], results['dt_mae']]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(
                comparison_data,
                x='Model',
                y='Accuracy (%)',
                title='Model Accuracy Comparison',
                color='Model',
                text_auto='.2f'
            )
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(
                comparison_data,
                x='Model',
                y='Error (marks)',
                title='Mean Absolute Error',
                color='Model',
                text_auto='.2f'
            )
            fig2.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # Actual vs Predicted scatter plots
        st.subheader("🎯 Actual vs Predicted Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=results['y_test'],
                y=results['lr_pred'],
                mode='markers',
                name='Predictions',
                marker=dict(size=8, color='blue', opacity=0.6)
            ))
            fig3.add_trace(go.Scatter(
                x=[results['y_test'].min(), results['y_test'].max()],
                y=[results['y_test'].min(), results['y_test'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig3.update_layout(
                title='Linear Regression',
                xaxis_title='Actual Marks',
                yaxis_title='Predicted Marks',
                height=400
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=results['y_test'],
                y=results['dt_pred'],
                mode='markers',
                name='Predictions',
                marker=dict(size=8, color='green', opacity=0.6)
            ))
            fig4.add_trace(go.Scatter(
                x=[results['y_test'].min(), results['y_test'].max()],
                y=[results['y_test'].min(), results['y_test'].max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig4.update_layout(
                title='Decision Tree',
                xaxis_title='Actual Marks',
                yaxis_title='Predicted Marks',
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # Feature importance
        st.markdown("---")
        st.subheader("🔑 Feature Importance (Linear Regression)")
        
        feature_names = ['Attendance (%)', 'Study Hours', 'Previous Marks', 'Assignment Scores']
        coefficients = st.session_state.lr_model.coef_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coefficients)
        }).sort_values('Importance', ascending=True)
        
        fig5 = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance (Coefficient Magnitude)',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig5.update_layout(height=400)
        st.plotly_chart(fig5, use_container_width=True)

# ============================================================================
# TAB 3: DATA ANALYTICS
# ============================================================================
with tab3:
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the models first!")
    else:
        st.header("Dataset Analytics")
        
        data = st.session_state.data
        
        # Basic statistics
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(data))
        with col2:
            st.metric("Avg Final Marks", f"{data['Final Marks'].mean():.2f}")
        with col3:
            st.metric("Avg Attendance", f"{data['Attendance (%)'].mean():.1f}%")
        with col4:
            st.metric("Avg Study Hours", f"{data['Study Hours'].mean():.2f}")
        
        st.markdown("---")
        
        # Distribution plots
        st.subheader("📈 Data Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig6 = px.histogram(
                data,
                x='Final Marks',
                nbins=20,
                title='Final Marks Distribution',
                labels={'Final Marks': 'Marks', 'count': 'Frequency'},
                color_discrete_sequence=['#1f77b4']
            )
            fig6.update_layout(height=400)
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            fig7 = px.box(
                data,
                y=['Attendance (%)', 'Previous Marks', 'Assignment Scores'],
                title='Feature Distributions',
                labels={'value': 'Score', 'variable': 'Feature'}
            )
            fig7.update_layout(height=400)
            st.plotly_chart(fig7, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Feature Correlations")
        
        corr_data = data[['Attendance (%)', 'Study Hours', 'Previous Marks', 
                          'Assignment Scores', 'Final Marks']].corr()
        
        fig8 = px.imshow(
            corr_data,
            text_auto='.2f',
            title='Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig8.update_layout(height=500)
        st.plotly_chart(fig8, use_container_width=True)
        
        # Data table
        st.subheader("📋 Sample Data")
        st.dataframe(data.head(20), use_container_width=True)
        
        # Download button
        csv = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset",
            data=csv,
            file_name="student_dataset.csv",
            mime="text/csv"
        )

# ============================================================================
# TAB 4: PREDICTION HISTORY
# ============================================================================
with tab4:
    st.header("Prediction History")
    
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions made yet. Use the 'Make Prediction' tab to get started!")
    else:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        st.subheader(f"📊 Total Predictions: {len(history_df)}")
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Predicted Marks", f"{history_df['Predicted Marks'].mean():.2f}")
        with col2:
            st.metric("Highest Prediction", f"{history_df['Predicted Marks'].max():.2f}")
        with col3:
            st.metric("Lowest Prediction", f"{history_df['Predicted Marks'].min():.2f}")
        
        st.markdown("---")
        
        # Trend chart
        fig9 = px.line(
            history_df,
            x=history_df.index,
            y='Predicted Marks',
            title='Prediction Trend',
            markers=True,
            labels={'index': 'Prediction Number', 'Predicted Marks': 'Marks'}
        )
        fig9.update_layout(height=400)
        st.plotly_chart(fig9, use_container_width=True)
        
        # History table
        st.subheader("📋 Detailed History")
        st.dataframe(history_df, use_container_width=True)
        
        # Download history
        csv_history = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prediction History",
            data=csv_history,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>🎓 Smart Student Performance Prediction System | Powered by Machine Learning</p>
    <p>Built with Streamlit, scikit-learn, and Plotly</p>
</div>
""", unsafe_allow_html=True)
