# PlaceSight - A Placement and Salary Prediction System using Machine Learning

## Project Overview

The Placement and Salary Prediction System is a machine learning-based web application that predicts:
1. **Placement Status** – Whether a student is likely to be placed or not.
2. **Expected Salary Package** – Estimated salary based on academic and skill-related features.

The system uses two machine learning models:
- **Random Forest Classifier** for placement prediction.
- **Random Forest Regressor** for salary prediction.

A web interface is developed using Flask to collect student details and display predictions in real time. The prediction data is stored in a CSV file and visualized using Power BI dashboards for analytical insights.

## Objectives
- Predict the placement probability of students.
- Estimate expected salary packages.
- Provide a user-friendly web application for predictions.
- Store prediction results for further analysis.
- Create interactive Power BI dashboards.

## Technologies Used

### Programming Language
- Python 3.x

### Machine Learning Libraries
- Pandas
- NumPy
- Scikit-learn
- Pickle

### Web Development
- Flask
- HTML
- CSS
- JavaScript 

### Data Visualization
- Power BI

## Machine Learning Models

### Placement Prediction Model
- Algorithm: Random Forest Classifier
- Output: Placed / Not Placed

### Salary Prediction Model
- Algorithm: Random Forest Regressor
- Output: Predicted Salary

## Project Structure

placement-salary-prediction/
│── app.py
│── placement_prediction.py
│── salary_prediction.py
│── placement_model.pkl
│── salary_model.pkl
│── requirements.txt
│── predictions.csv
│
├── templates/
│   ├── index.html
│   ├── result.html
│
├── static/
│   ├── style.css
│
├── dataset/
│   ├── student_data.csv
│
└── powerbi/
    └── dashboard.pbix
