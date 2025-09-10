# Data Co-Pilot: An Automated Machine Learning Web Application

## Project Overview

Data Co-Pilot is a comprehensive, web-based AutoML (Automated Machine Learning) application built with Streamlit. It provides a complete end-to-end pipeline for data analysis and machine learning, enabling users to go from raw data to a trained, evaluated, and interpretable model through an interactive user interface.

The application supports both classification and regression tasks and includes a wide array of preprocessing, modeling, and evaluation functionalities.

## Key Features

* **Flexible Data Ingestion:** Supports multiple data sources, including local file uploads (CSV, XLSX, Parquet, JSON) and direct connections to relational databases (PostgreSQL, MySQL, SQLite) via SQLAlchemy.
* **Comprehensive Exploratory Data Analysis (EDA):**
    * Generates interactive visualizations for numerical and categorical data using Plotly.
    * Provides a detailed, one-click EDA report using `ydata-profiling`.
    * Includes correlation heatmaps and missing value analysis.
* **Automated Preprocessing Pipeline:**
    * Constructs a robust `scikit-learn` pipeline to handle missing data (KNNImputer), feature scaling, and one-hot encoding.
    * Includes advanced options for feature engineering (datetime extraction, polynomial features), outlier removal (Isolation Forest), and feature selection (SelectKBest, PCA).
* **Model Training & Hyperparameter Tuning:**
    * Supports a wide range of industry-standard models, including RandomForest, XGBoost, LightGBM, SVMs, and Linear Models.
    * Features an optional GridSearchCV integration for hyperparameter optimization.
* **In-Depth Model Evaluation:**
    * Calculates and displays key performance metrics (Accuracy, F1, AUC-ROC for classification; R², MSE, MAE for regression).
    * Generates visualizations such as confusion matrices, ROC curves, and feature importance plots.
* **Model Interpretability:** Integrates **SHAP** (SHapley Additive exPlanations) to provide both global (summary plots) and local (dependence plots) explanations for model predictions.
* **Model Export & Live Prediction:**
    * Allows users to export the fully trained `scikit-learn` pipeline (preprocessor + model) for deployment.
    * Includes a real-time prediction interface to test the model with new, user-provided data points.

## Technology Stack

* **Application Framework:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **Database Connectivity:** SQLAlchemy, Psycopg2, My-SQL-Connector
* **Machine Learning & Preprocessing:** Scikit-learn, XGBoost, LightGBM
* **Model Interpretability:** SHAP
* **Data Profiling & Visualization:** YData-Profiling, Plotly, Matplotlib, Seaborn
