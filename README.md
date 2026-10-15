# 🎓 Student Academic Performance Prediction Using Machine Learning

## 📌 Project Overview

This project uses Machine Learning to predict a student's final academic performance based on their age, weekly study time, past class failures, absences, semester pointers, and participation in extracurricular activities.

The project includes a Streamlit web application where users can enter student details and receive a predicted final percentage.

## 🎯 Objectives

- Predict student academic performance using Machine Learning.
- Preprocess numerical and categorical data.
- Train and compare multiple regression algorithms.
- Evaluate model performance using MAE, RMSE, and R² score.
- Develop an interactive web application using Streamlit.

## 🤖 Machine Learning Algorithms

The project trains and compares three algorithms:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The model with the lowest Root Mean Squared Error (RMSE) is selected for prediction.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## 📊 Dataset

This project uses the **UCI Student Performance dataset**, specifically `student-mat.csv`, which contains information about Portuguese secondary-school students and their mathematics performance.

The original grades are converted to a 0–10 scale for the two semester-pointer inputs and a 0–100 scale for the final percentage.

**Note:** This is a prototype using secondary-school data. Its predictions may not generalize to Indian college students.

## ✨ Features

- Student detail input form
- Semester 1 and Semester 2 pointer entry
- Predicted final percentage out of 100
- Comparison of three Machine Learning models
- Evaluation metrics: MAE, RMSE, and R² score
- Interactive Streamlit interface

## 📁 Project Structure

```text
Student_Performance_Prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── student_model.joblib
├── model_results.csv
├── model_comparison.png
└── README.md
```

The dataset file `student-mat.csv` is needed when retraining the models, but not when running the application with the existing saved model.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Student_Performance_Prediction
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your repository's actual URL.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the application

```bash
python -m streamlit run app.py
```

### 4. Retrain the models (optional)

Place `student-mat.csv` in the project folder, then run:

```bash
python train_model.py
```

This trains the models, compares their evaluation scores, and saves the best model.

## 📈 Model Evaluation

The models are evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Measures how well the model explains variation in the target

The evaluation results are saved in `model_results.csv`, and a model comparison graph is saved as `model_comparison.png`.

## ☁️ Deployment

The application can be deployed using [Streamlit Community Cloud](https://share.streamlit.io/).

Select the GitHub repository, choose the appropriate branch, and set `app.py` as the main file.

## ⚠️ Limitations

- The dataset contains Portuguese secondary-school mathematics data, not Indian college records.
- Predicted percentages are estimates, not guaranteed results.
- The model's performance depends on the available dataset and selected features.

## 👨‍💻 Author

Developed as a Machine Learning mini-project for student academic performance prediction.