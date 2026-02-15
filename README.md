# Smart Student Performance Prediction Using AI 🎓

## 🌟 **NEW: Live Web Application!**

This project now includes an **interactive web application** built with Streamlit! You can run it locally in your browser with a beautiful, user-friendly interface.

### 🚀 Quick Start - Web App

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web application
streamlit run app.py
```
or 
## 🌐 Live Demo

🚀 Try the live web app here:  
👉 https://deepak-performance-predictor.streamlit.app


**Or use the launcher scripts:**
- **Linux/Mac**: `bash run_app.sh`
- **Windows**: Double-click `run_app.bat`

The app will automatically open in your browser at `http://localhost:8501`

### ✨ Web App Features

- 🎯 **Interactive Prediction Interface** - Real-time student performance predictions
- 📊 **Live Model Training** - Train models with custom dataset sizes
- 📈 **Beautiful Visualizations** - Interactive charts with Plotly
- 📚 **Prediction History** - Track all predictions with timestamps
- 💡 **Smart Recommendations** - Personalized advice for each student
- 📥 **Data Export** - Download datasets and prediction history
- 🎨 **Modern UI** - Clean, responsive design with tabs and metrics
- 🔄 **Real-time Updates** - Instant predictions and visualizations

---

## Project Overview
This machine learning project predicts student performance based on key academic indicators using Python and scikit-learn. The system achieves **73.81% accuracy** using Linear Regression.

## Features
✅ **Web Application** with interactive UI  
✅ Automated dataset generation with 200+ student records  
✅ Data preprocessing and cleaning  
✅ Two ML models: Linear Regression & Decision Tree  
✅ 80-20 train-test split  
✅ Comprehensive model evaluation  
✅ New student prediction capability  
✅ Beautiful visualizations and dashboards  
✅ Beginner-friendly code with detailed comments  
✅ Prediction history tracking  
✅ Export functionality  

## Prediction Factors
The model predicts **Final Marks** based on:
1. **Attendance (%)** - Class attendance percentage
2. **Study Hours** - Daily study hours (1-10 hours)
3. **Previous Marks** - Performance in previous exams
4. **Assignment Scores** - Assignment completion scores

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Required Libraries
```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy - Data manipulation
- matplotlib, seaborn, plotly - Visualizations
- scikit-learn - Machine learning
- streamlit - Web application framework

## Usage

### 🌐 Option 1: Web Application (Recommended)

```bash
streamlit run app.py
```

**Web App Includes:**
- Interactive prediction form with sliders
- Real-time model training
- Multiple visualization tabs
- Performance metrics dashboard
- Prediction history with export
- Data analytics and insights

### 💻 Option 2: Command Line Scripts

**Run the main ML pipeline:**
```bash
python student_performance_predictor.py
```

**Predict for new students:**
```bash
python predict_new_student.py
```

## Project Structure
```
student_performance_ml/
│
├── app.py                            # 🌟 NEW: Streamlit web application
├── run_app.sh                        # 🌟 NEW: Launch script (Linux/Mac)
├── run_app.bat                       # 🌟 NEW: Launch script (Windows)
│
├── student_performance_predictor.py  # Main ML project script
├── predict_new_student.py            # CLI prediction tool
│
├── student_dataset.csv               # Generated dataset
├── test_predictions.csv              # Model predictions
├── prediction_results.png            # Visualization dashboard
├── feature_importance.png            # Feature importance chart
├── model_report.txt                  # Detailed model report
│
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

## Model Performance

### Linear Regression (Best Model)
- **R² Score (Accuracy)**: 73.81%
- **Mean Absolute Error**: 4.64 marks
- **Root Mean Squared Error**: 5.68 marks

### Decision Tree
- **R² Score (Accuracy)**: 24.77%
- **Mean Absolute Error**: 7.13 marks
- **Root Mean Squared Error**: 9.62 marks

## Feature Importance
Based on Linear Regression coefficients:
1. **Study Hours**: 2.4803 (Most Important)
2. **Attendance (%)**: 0.3273
3. **Previous Marks**: 0.3050
4. **Assignment Scores**: 0.1395

## Sample Predictions
| Student | Attendance | Study Hours | Previous Marks | Assignments | Predicted Marks |
|---------|------------|-------------|----------------|-------------|-----------------|
| 1       | 95%        | 8.5 hrs     | 88             | 92          | 90.73           |
| 2       | 75%        | 5.0 hrs     | 70             | 78          | 68.06           |
| 3       | 60%        | 3.0 hrs     | 55             | 65          | 51.81           |
| 4       | 85%        | 6.5 hrs     | 80             | 85          | 79.08           |

## Web App Screenshots

### Main Prediction Interface
- Interactive sliders for all student metrics
- Real-time prediction with grade display
- Gauge chart visualization
- Personalized recommendations

### Model Performance Dashboard
- Side-by-side model comparison
- Actual vs Predicted scatter plots
- Feature importance visualization
- Detailed accuracy metrics

### Data Analytics
- Dataset statistics overview
- Distribution histograms
- Correlation heatmap
- Downloadable datasets

### Prediction History
- Complete prediction log
- Trend visualization
- Export to CSV
- Clear history option

## Key Insights
- **Study hours** have the strongest impact on final marks
- The model achieves low prediction errors (~4.64 marks)
- Linear Regression outperforms Decision Tree for this dataset
- Students with 8+ study hours and 90%+ attendance score above 85

## Educational Value
Perfect for:
- Machine Learning beginners
- Data Science students
- Academic project demonstrations
- Understanding regression models
- Learning scikit-learn basics
- Web app development with Streamlit
- Interactive data visualization

## Technology Stack
- **Backend**: Python 3.8+
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **Data Processing**: pandas, numpy

## Future Improvements
- ✅ Add interactive web interface (COMPLETED!)
- 🔄 User authentication and data persistence
- 🔄 Database integration (SQLite/PostgreSQL)
- 🔄 More advanced models (Random Forest, XGBoost)
- 🔄 Cross-validation and hyperparameter tuning
- 🔄 API endpoints for external integration
- 🔄 Mobile-responsive design
- 🔄 Multi-language support
- 🔄 Export reports to PDF

## Troubleshooting

### Web App Won't Start
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Try running directly
streamlit run app.py
```

### Port Already in Use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Module Not Found Errors
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

## Contributing
Feel free to fork this project and add your own improvements!

## License
This project is open-source and available for educational purposes.

## Author
Created as a beginner-friendly ML project with live web interface for educational demonstrations.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/)
- ML models using [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Plotly](https://plotly.com/)

---
**Note**: Dataset is synthetically generated for demonstration purposes.

## Contact & Support
For issues or questions, please open an issue on the project repository.

---
**Happy Learning! 🎓📊🚀**
