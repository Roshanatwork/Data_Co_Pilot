import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import datetime # For date/time feature extraction
import re # For regular expressions to sanitize feature names
import sqlite3 # For SQLite database connection

# NEW IMPORTS FOR PHASE 3 DATABASE CONNECTORS
import sqlalchemy # Required for pandas.read_sql_query with many DBs
# You will need to install specific drivers:
# !pip install psycopg2-binary # For PostgreSQL
# !pip install mysql-connector-python # For MySQL

# Scikit-learn Pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures

# Feature Selection and Dimensionality Reduction
from sklearn.feature_selection import SelectKBest, f_classif, f_regression # Added f_regression for regression tasks
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.utils import resample

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve,
                             classification_report, mean_squared_error, r2_score, mean_absolute_error)
import shap # For SHAP interpretability

# EDA Reporting
from ydata_profiling import ProfileReport # Using ydata-profiling for comprehensive reports
import plotly.express as px # For interactive plots

# --- Helper function to extract datetime features ---
def apply_datetime_features(df_input, datetime_columns):
    """
    Extracts various time-based features (year, month, day, hour, etc.) from datetime columns
    and drops the original datetime columns.
    """
    df_processed = df_input.copy()
    for col in datetime_columns:
        if pd.api.types.is_datetime64_any_dtype(df_processed[col]):
            df_processed[f"{col}_year"] = df_processed[col].dt.year
            df_processed[f"{col}_month"] = df_processed[col].dt.month
            df_processed[f"{col}_day"] = df_processed[col].dt.day
            df_processed[f"{col}_dayofweek"] = df_processed[col].dt.dayofweek
            df_processed[f"{col}_dayofyear"] = df_processed[col].dt.dayofyear
            df_processed[f"{col}_weekofyear"] = df_processed[col].dt.isocalendar().week.astype(int)
            df_processed[f"{col}_quarter"] = df_processed[col].dt.quarter
            df_processed[f"{col}_is_month_start"] = df_processed[col].dt.is_month_start.astype(int)
            df_processed[f"{col}_is_month_end"] = df_processed[col].dt.is_month_end.astype(int)
            df_processed[f"{col}_hour"] = df_processed[col].dt.hour
            df_processed[f"{col}_minute"] = df_processed[col].dt.minute
            # Drop the original datetime column after extraction
            df_processed = df_processed.drop(columns=[col])
    return df_processed

# --- Helper function to sanitize feature names for models ---
def sanitize_feature_names(feature_names):
    """
    Cleans feature names by removing special characters and handling names starting with numbers,
    which can cause issues with some ML libraries (e.g., XGBoost, LightGBM).
    """
    sanitized_names = []
    for name in feature_names:
        # Remove characters that are not alphanumeric or underscore
        sanitized_name = re.sub(r'[^A-Za-z0-9_]+', '', str(name))
        # Prepend 'feature_' if the name starts with a number (invalid in some libraries like XGBoost)
        if re.match(r'^\d', sanitized_name):
            sanitized_name = 'feature_' + sanitized_name
        sanitized_names.append(sanitized_name)
    return sanitized_names

# --- Helper function for synthetic feature (now correctly uses existing features to create a new one) ---
def add_synthetic_feature_to_df(input_df, original_df_cleaned, feature_names_from_training):
    """
    Adds a synthetic, highly correlated feature for demonstration purposes during training.
    During prediction, it attempts to recreate this feature based on existing numerical features,
    or adds a default if no numerical features are available.
    """
    # This function should only add the feature if it was actually added during training
    # This check is crucial for consistent prediction
    if 'synthetic_highly_correlated_feature' not in feature_names_from_training:
        return input_df # If the feature wasn't part of training, don't add it now

    df_with_synthetic = input_df.copy()

    numerical_features_in_input = input_df.select_dtypes(include=np.number).columns.tolist()

    if numerical_features_in_input:
        # Use the mean of numerical features as a proxy for the 'target' behavior for synthetic feature creation
        # This is a simplified approach for demonstration purposes, assuming some correlation
        # between numerical features and the original target behavior.
        proxy_target_behavior = df_with_synthetic[numerical_features_in_input].mean(axis=1)
        noise = np.random.normal(0, 0.01, size=len(df_with_synthetic))
        df_with_synthetic['synthetic_highly_correlated_feature'] = proxy_target_behavior + noise
    else:
        # If no numerical features, create a placeholder synthetic feature or handle appropriately
        df_with_synthetic['synthetic_highly_correlated_feature'] = 0.5 + np.random.normal(0, 0.01, size=len(df_with_synthetic))
        st.warning("No numerical features to base synthetic feature on. Adding a default synthetic feature.")

    return df_with_synthetic


# --- Caching data loading to prevent reloading the CSV every time the app reruns ---
@st.cache_data(show_spinner="Loading data...")
def load_data(data_source_type, uploaded_file=None, db_config=None):
    """
    Loads data into a pandas DataFrame from either an uploaded file or a database.
    Performs initial data quality checks.
    """
    df = None

    if data_source_type == "upload_file":
        if uploaded_file is None:
            st.error("No file uploaded. Please upload a file to proceed.")
            st.stop()

        file_extension = uploaded_file.name.split('.')[-1].lower()

        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
            elif file_extension == "parquet":
                df = pd.read_parquet(uploaded_file)
            elif file_extension == "json":
                df = pd.read_json(uploaded_file)
            elif file_extension == "xlsx":
                # For xlsx, use openpyxl engine
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:
                st.error(f"Unsupported file type: .{file_extension}. Please upload a CSV, Parquet, JSON, or XLSX file.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

    elif data_source_type == "database":
        if db_config is None:
            st.error("Database configuration missing.")
            st.stop()

        db_type = db_config.get("type")
        db_path = db_config.get("path") # Used for SQLite
        sql_query = db_config.get("query")
        # New DB connection parameters
        db_host = db_config.get("host")
        db_port = db_config.get("port")
        db_name = db_config.get("database_name")
        db_user = db_config.get("username")
        db_password = db_config.get("password")

        conn_str = None

        if db_type == "sqlite":
            if not db_path:
                st.error("SQLite database file path is required.")
                st.stop()
            if not sql_query:
                st.error("SQL Query is required for SQLite.")
                st.stop()

            try:
                # Use a context manager for the connection
                with sqlite3.connect(db_path) as conn:
                    df = pd.read_sql_query(sql_query, conn)
                st.success(f"Successfully loaded data from SQLite database: {db_path}")
            except FileNotFoundError:
                st.error(f"SQLite database file not found at: {db_path}")
                st.stop()
            except pd.io.sql.DatabaseError as e:
                st.error(f"Error executing SQL query or connecting to database: {e}. Check your query or database file.")
                st.stop()
            except Exception as e:
                st.error(f"An unexpected error occurred while connecting to SQLite: {e}")
                st.stop()

        elif db_type == "postgresql":
            if not all([db_host, db_port, db_name, db_user, db_password, sql_query]):
                st.error("All PostgreSQL connection details (host, port, database, user, password, query) are required.")
                st.stop()
            try:
                conn_str = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                # Using SQLAlchemy engine for pandas.read_sql_query
                engine = sqlalchemy.create_engine(conn_str)
                with engine.connect() as conn:
                    df = pd.read_sql_query(sql_query, conn)
                st.success(f"Successfully loaded data from PostgreSQL database: {db_name}@{db_host}")
            except ImportError:
                st.error("PostgreSQL driver (psycopg2-binary) not found. Please install it: `pip install psycopg2-binary`")
                st.stop()
            except sqlalchemy.exc.SQLAlchemyError as e:
                st.error(f"Error connecting to PostgreSQL or executing query: {e}. Check your credentials and query.")
                st.stop()
            except Exception as e:
                st.error(f"An unexpected error occurred while connecting to PostgreSQL: {e}")
                st.stop()

        elif db_type == "mysql":
            if not all([db_host, db_port, db_name, db_user, db_password, sql_query]):
                st.error("All MySQL connection details (host, port, database, user, password, query) are required.")
                st.stop()
            try:
                conn_str = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                # Using SQLAlchemy engine for pandas.read_sql_query
                engine = sqlalchemy.create_engine(conn_str)
                with engine.connect() as conn:
                    df = pd.read_sql_query(sql_query, conn)
                st.success(f"Successfully loaded data from MySQL database: {db_name}@{db_host}")
            except ImportError:
                st.error("MySQL driver (mysql-connector-python) not found. Please install it: `pip install mysql-connector-python`")
                st.stop()
            except sqlalchemy.exc.SQLAlchemyError as e:
                st.error(f"Error connecting to MySQL or executing query: {e}. Check your credentials and query.")
                st.stop()
            except Exception as e:
                st.error(f"An unexpected error occurred while connecting to MySQL: {e}")
                st.stop()

        else:
            st.error(f"Unsupported database type: {db_type}")
            st.stop()

    if df is not None:
        # --- Data Quality Check: Drop Duplicates (Step 5) ---
        original_rows = len(df)
        df.drop_duplicates(inplace=True)
        if len(df) < original_rows:
            st.info(f"🗑️ Removed {original_rows - len(df)} duplicate rows from the dataset.")

        # Debugging line: Print the number of rows loaded
        st.info(f"Debug: DataFrame loaded with {len(df)} rows after initial cleaning.")

    return df

# --- Defines the preprocessing pipeline for the ColumnTransformer ---
def create_preprocessing_pipeline_for_transformer(numerical_cols, categorical_cols, impute, scale, ohe_encode, poly_features, select, reduce, task_type):
    """
    Creates a scikit-learn preprocessing pipeline using ColumnTransformer.
    This pipeline includes steps for imputation, polynomial features, scaling,
    feature selection (SelectKBest), dimensionality reduction (PCA), and one-hot encoding.
    """
    numerical_transformer_steps = []
    if impute:
        numerical_transformer_steps.append(('imputer', KNNImputer(n_neighbors=3)))
    if poly_features:
        numerical_transformer_steps.append(('poly', PolynomialFeatures(degree=2, include_bias=False)))
    if scale:
        numerical_transformer_steps.append(('scaler', StandardScaler()))

    # SelectKBest and PCA operate on the numerical features after imputation and scaling
    if select and len(numerical_cols) > 0:
        if task_type == "Classification":
            numerical_transformer_steps.append(('selector', SelectKBest(score_func=f_classif, k=min(len(numerical_cols), 10))))
        else: # Regression - use f_regression for numerical targets
            numerical_transformer_steps.append(('selector', SelectKBest(score_func=f_regression, k=min(len(numerical_cols), 10))))

    if reduce and len(numerical_cols) > 1:
        numerical_transformer_steps.append(('pca', PCA(n_components=min(len(numerical_cols)-1, 5), random_state=42)))

    numerical_transformer = Pipeline(numerical_transformer_steps) if numerical_transformer_steps else 'passthrough'


    categorical_transformer_steps = []
    if ohe_encode and categorical_cols:
        # OneHotEncoder handles unknown categories and missing values (NaN) by treating them as a new category
        categorical_transformer_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))

    categorical_transformer = Pipeline(categorical_transformer_steps) if categorical_transformer_steps else 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough' # Keep other columns not explicitly transformed
    )
    return preprocessor


# --- Initialize session state variables ---
# IMPORTANT: All st.session_state variables must be initialized here to prevent AttributeError
# upon initial load or when certain paths are not executed.
if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False
    st.session_state.pipeline = None
    st.session_state.label_encoder = None
    st.session_state.task_type = "Classification" # Default task type
    st.session_state.comparison_results = pd.DataFrame()
    st.session_state.X_transformed_cols = []
    st.session_state.X_train_shape = None
    st.session_state.full_ml_pipeline = None
    st.session_state.original_X_cols_before_custom_fe = []
    st.session_state.id_cols_used_in_training = []
    st.session_state.target_col_used_in_training = None
    st.session_state.datetime_features_enabled = False # Flag for datetime feature extraction
    st.session_state.datetime_cols_extracted = [] # Stores list of datetime columns identified during training FE
    st.session_state.add_synthetic_feature_enabled = False # Flag for synthetic feature
    st.session_state.processed_training_features_at_pipeline_entry = [] # Crucial: X.columns BEFORE ColumnTransformer fit
    st.session_state.target_encoder = None # Store target encoder, initialized here for target encoding (if applicable)

# NEW: Initialize session state for data source persistence
if 'data_source_choice_state' not in st.session_state:
    st.session_state.data_source_choice_state = "Upload File"
if 'uploaded_file_state' not in st.session_state:
    st.session_state.uploaded_file_state = None
if 'db_config_state' not in st.session_state:
    st.session_state.db_config_state = None


# --- Streamlit UI Elements and Main Logic ---

st.set_page_config(page_title="Data Co-Pilot", layout="wide")
st.title("🧠 Data Co-Pilot")

# Choose data source type (file upload or database)
st.session_state.data_source_choice_state = st.radio(
    "Select Data Source",
    ("Upload File", "Connect to Database"),
    key="data_source_choice",
    index=0 if st.session_state.data_source_choice_state == "Upload File" else 1 # Maintain state
)

df = None # Initialize df outside the conditional blocks

if st.session_state.data_source_choice_state == "Upload File":
    uploaded_file_temp = st.file_uploader("📤 Upload your data file", type=["csv", "parquet", "json", "xlsx"])
    if uploaded_file_temp: # If a new file is uploaded or existing is there
        st.session_state.uploaded_file_state = uploaded_file_temp

    # Only attempt to load data if a file is present in session state
    if st.session_state.uploaded_file_state is not None:
        df = load_data(data_source_type="upload_file", uploaded_file=st.session_state.uploaded_file_state)
    else:
        st.info("Please upload a data file to get started or select 'Connect to Database'.")
        st.stop() # Explicitly stop if no file is available to prevent error

elif st.session_state.data_source_choice_state == "Connect to Database":
    st.subheader("Database Connection Settings")
    # Updated: Add PostgreSQL and MySQL to the selectbox
    db_type = st.selectbox("Database Type", ["SQLite", "PostgreSQL", "MySQL"], key="db_type_select")

    db_config_current = {}

    if db_type == "SQLite":
        sqlite_file_path = st.text_input("SQLite Database File Path", value="example.db", key="sqlite_path_input")
        sqlite_sql_query = st.text_area("SQL Query", value="SELECT * FROM my_table;", height=100, key="sqlite_query_input")
        db_config_current = {
            "type": "sqlite",
            "path": sqlite_file_path,
            "query": sqlite_sql_query
        }
    elif db_type == "PostgreSQL":
        pg_host = st.text_input("PostgreSQL Host", value="localhost", key="pg_host_input")
        pg_port = st.number_input("PostgreSQL Port", value=5432, key="pg_port_input")
        pg_db_name = st.text_input("PostgreSQL Database Name", value="mydatabase", key="pg_db_name_input")
        pg_user = st.text_input("PostgreSQL Username", value="myuser", key="pg_user_input")
        pg_password = st.text_input("PostgreSQL Password", type="password", key="pg_password_input")
        pg_sql_query = st.text_area("SQL Query", value="SELECT * FROM customer_data;", height=100, key="pg_query_input")
        db_config_current = {
            "type": "postgresql",
            "host": pg_host,
            "port": pg_port,
            "database_name": pg_db_name,
            "username": pg_user,
            "password": pg_password,
            "query": pg_sql_query
        }
    elif db_type == "MySQL":
        mysql_host = st.text_input("MySQL Host", value="localhost", key="mysql_host_input")
        mysql_port = st.number_input("MySQL Port", value=3306, key="mysql_port_input")
        mysql_db_name = st.text_input("MySQL Database Name", value="mydatabase", key="mysql_db_name_input")
        mysql_user = st.text_input("MySQL Username", value="myuser", key="mysql_user_input")
        mysql_password = st.text_input("MySQL Password", type="password", key="mysql_password_input")
        mysql_sql_query = st.text_area("SQL Query", value="SELECT * FROM my_table;", height=100, key="mysql_query_input")
        db_config_current = {
            "type": "mysql",
            "host": mysql_host,
            "port": mysql_port,
            "database_name": mysql_db_name,
            "username": mysql_user,
            "password": mysql_password,
            "query": mysql_sql_query
        }

    # Add the Clear Cache button for database connections
    if st.button("Clear Data Cache"):
        st.cache_data.clear()
        st.success("Data cache cleared. Please click 'Load Data from Database' again.")
        # Ensure that after clearing cache, if nothing else happens, the app state reflects no data
        st.session_state.db_config_state = None
        st.rerun() # Rerun to reflect cache clear state

    # Separate button to trigger data load from database
    if st.button("Load Data from Database"):
        if db_config_current:
            st.session_state.db_config_state = db_config_current # Store updated config
            # Immediately attempt to load data after setting config
            df = load_data(data_source_type="database", db_config=st.session_state.db_config_state)
        else:
            st.warning("Please provide complete database connection details.")

    # If df is still None after attempting to load (e.g., first run or invalid config), stop
    if df is None and st.session_state.db_config_state is not None: # Only try to load if a config exists
        # This branch ensures data is loaded if config exists from previous run but df isn't populated yet
        df = load_data(data_source_type="database", db_config=st.session_state.db_config_state)
    elif df is None: # If no config and no df, just stop gracefully
        st.info("Please enter database details and click 'Load Data from Database'.")
        st.stop()


# Data validation: Ensure the dataset has enough rows and is not None
if df is None or len(df) < 10:
    if df is None:
        st.error("No data loaded. Please upload a file or connect to a database.")
    else: # df is not None, but len(df) < 10
        st.error("Dataset too small (minimum 10 rows required for analysis)")
    st.stop() # This st.stop() will cause the app to rerun from the top with the error message

# ALL subsequent code that depends on `df` must be inside this block:
if df is not None:
    st.subheader("📄 Raw Data Preview (after initial cleaning)")
    st.dataframe(df.head())

    mode = st.radio("Choose Mode", ["EDA", "ML Mode"])

    # --- EDA Mode ---
    if mode == "EDA":
        st.subheader("📋 Data Summary")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

        st.subheader("🧩 Missing Values")
        st.dataframe(df.isnull().sum().to_frame("Missing Values"))

        st.subheader("📈 Numeric Distributions (Interactive with Plotly)")
        for col in df.select_dtypes(include=np.number).columns:
            fig = px.histogram(df, x=col, marginal="box", title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📦 Boxplots (Interactive with Plotly)")
        for col in df.select_dtypes(include=np.number).columns:
            fig = px.box(df, y=col, title=f'Boxplot of {col}')
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔥 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("📊 Comprehensive EDA Report")
        if st.button("Generate Detailed Report (via ydata-profiling)"):
            with st.spinner("Generating comprehensive EDA report... This might take a while for large datasets."):
                profile = ProfileReport(df, title="Pandas Profiling Report")
                st.components.v1.html(profile.to_html(), height=800, scrolling=True)
                st.success("Detailed EDA report generated above.")

    # --- ML Mode ---
    elif mode == "ML Mode":
        # --- Start of enhanced error handling for ML Mode ---
        try:
            # Initialize progress tracking for preprocessing
            progress_bar = st.progress(0)
            status_text = st.empty()
            progress_state = {"steps": 0}

            # --- Target and Task Type Selection ---
            # Ensure there are columns to select from
            if df.columns.empty:
                st.error("The loaded dataset has no columns. Cannot proceed to ML Mode.")
                st.stop()

            target = st.selectbox("🎯 Select Target Column", df.columns[::-1])

            # Updated: More robust task type determination
            # Check if the target is numeric
            if pd.api.types.is_numeric_dtype(df[target]):
                # If numeric and many unique values (heuristic for continuous)
                if df[target].nunique() > len(df) * 0.2: # More than 20% unique values often implies regression
                    st.session_state.task_type = "Regression"
                    st.info(f"Target '{target}' appears continuous (numeric with many unique values). Setting task type to **Regression**.")
                # If numeric and few unique values (heuristic for discrete/classification)
                elif df[target].nunique() <= 10: # A smaller threshold for classification
                    st.session_state.task_type = "Classification"
                    st.info(f"Target '{target}' appears discrete (numeric with few unique values). Setting task type to **Classification**.")
                else: # Fallback for ambiguous numeric cases
                    st.warning("Could not definitively determine task type for numeric target. Defaulting to Regression.")
                    st.session_state.task_type = "Regression"
            else: # Non-numeric target always implies classification
                st.session_state.task_type = "Classification"
                st.info(f"Target '{target}' is non-numeric. Setting task type to **Classification**.")

            # Separate features (X) and target (y)
            X = df.drop(columns=[target])
            y = df[target]

            # --- Handle missing values in target 'y' early (Step 4 - Imputation) ---
            if isinstance(y, pd.Series):
                if y.isnull().any():
                    if st.session_state.task_type == "Classification":
                        y = y.fillna(y.mode()[0])
                        st.write("🎚️ Imputed missing values in target column 'y' with mode (for classification).")
                    else: # Regression
                        y = y.fillna(y.median())
                        st.write("🎚️ Imputed missing values in target column 'y' with median (for regression).")
            else: # For numpy arrays (if y was already converted to numpy array)
                if np.isnan(y).any():
                    # For numpy arrays, need to ensure y is mutable if it's not already
                    y = np.array(y) # Ensure it's a modifiable numpy array
                    y = np.nan_to_num(y, nan=np.nanmedian(y))
                    st.write("🎚️ Imputed missing values in target array 'y' with median/mode.")


            # --- Handle binary/multi-class target encoding (only for classification) ---
            if st.session_state.task_type == "Classification":
                if y.dtype == 'object' or pd.api.types.is_categorical_dtype(y):
                    le = LabelEncoder()
                    y = le.fit_transform(y.astype(str)) # Ensure string for LabelEncoder
                    st.session_state.label_encoder = le # Store the label encoder in session state
                    st.info(f"Mapped target classes: {list(le.classes_)} to numerical values for classification.")
                    y = y.astype(int) # Explicitly cast to integer numpy array after encoding for robustness

            # --- Drop ID-like columns from features (Step 2 - Data Cleaning) ---
            id_cols_to_drop = [col for col in X.columns if 'id' in col.lower() and X[col].nunique() == len(X)]
            if id_cols_to_drop:
                X.drop(columns=id_cols_to_drop, inplace=True)
                st.info(f"Dropped potential ID columns: {', '.join(id_cols_to_drop)}")

            # Store original feature column names *before* custom feature engineering
            # These are the columns the input widgets will be built from.
            st.session_state.id_cols_used_in_training = id_cols_to_drop
            st.session_state.target_col_used_in_training = target
            st.session_state.original_X_cols_before_custom_fe = [col for col in df.columns if col not in [target] + id_cols_to_drop]

            # --- NEW ROBUSTNESS CHECK: Ensure X_for_pipeline is not empty ---
            # Perform this check before further processing X_for_pipeline
            if X.empty or X.shape[1] == 0:
                st.error("❌ No features available for ML training after dropping target and potential ID columns. Please ensure your dataset has at least two columns and one is suitable as a target, leaving other features for training.")
                st.stop()

            # --- Preprocessing Toggles ---
            st.sidebar.subheader("⚙️ Preprocessing")
            auto = st.sidebar.checkbox("Auto Mode", value=True, key="auto_mode_checkbox")

            if not auto:
                impute = st.sidebar.checkbox("KNN Impute Missing (Numerical)", key="impute_checkbox")
                scale = st.sidebar.checkbox("Scale Features (Numerical)", key="scale_checkbox")
                ohe_encode = st.sidebar.checkbox("One-Hot Encode Categoricals", value=True, key="ohe_encode_checkbox") # Default to True

                # Direct assignment to session state variable
                st.session_state.datetime_features_enabled = st.sidebar.checkbox("Extract Date/Time Features",
                                                                                 value=st.session_state.datetime_features_enabled,
                                                                                 key="datetime_features_checkbox")

                poly_features = st.sidebar.checkbox("Add Polynomial Features (Numerical)", key="poly_features_checkbox") # New Poly Feature
                outliers = st.sidebar.checkbox("Outlier Handling (Isolation Forest)", key="outliers_checkbox")
                select = st.sidebar.checkbox("Feature Selection (SelectKBest)", key="select_checkbox")
                reduce = st.sidebar.checkbox("Apply PCA (Dimensionality Reduction)", key="reduce_checkbox")
                imbalance = st.sidebar.checkbox("Handle Imbalance (Upsampling Binary Classification)", key="imbalance_checkbox")
            else:
                impute = scale = ohe_encode = True
                st.session_state.datetime_features_enabled = True # In auto mode, it's always true
                poly_features = False # Not default in auto for complexity
                reduce = False # Not default in auto for interpretability
                outliers = True
                select = True
                imbalance = True

            # Capture the state of add_synthetic_feature for session_state
            st.sidebar.subheader("🧪 Synthetic Feature (Demonstration)")
            # Direct assignment to session state variable
            st.session_state.add_synthetic_feature_enabled = st.sidebar.checkbox("Add highly correlated synthetic feature (for demo)",
                                                                                 value=st.session_state.add_synthetic_feature_enabled,
                                                                                 key="add_synthetic_feature_checkbox")

            if st.session_state.add_synthetic_feature_enabled:
                st.markdown("""
                <div style="background-color:#fff3cd; color:#856404; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    ⚠️ **Warning on 100% Accuracy:** Achieving near 100% accuracy in real-world machine learning problems is extremely rare and often indicates issues like **data leakage** (where the model accidentally sees the answer during training) or an overly simplistic dataset.
                    <br><br>
                    This "synthetic highly correlated feature" is added **for demonstration purposes only** to show how models can perform with a very strong, almost perfect signal. It is **not a typical practice** in building robust real-world ML systems unless such a feature genuinely exists and is ethically sourced.
                </div>
                """, unsafe_allow_html=True)


            # Calculate total steps for the progress bar *before* defining update_progress or making calls
            total_steps = 0
            if st.session_state.datetime_features_enabled: total_steps += 1
            if outliers: total_steps += 1
            if imbalance: total_steps += 1
            total_steps += 1 # for preprocessing_pipeline.fit_transform()

            if total_steps == 0: # Ensure it's never zero to prevent division by zero
                total_steps = 1

            # Helper function to update the progress bar
            def update_progress():
                progress_state["steps"] += 1
                current_progress = progress_state["steps"] / total_steps
                progress_bar.progress(current_progress)
                status_text.text(f"Preprocessing: Step {progress_state['steps']}/{total_steps}")


            # --- Feature Engineering (applied to X before ColumnTransformer) ---
            X_for_pipeline = X.copy() # This X_for_pipeline will be the input to the ColumnTransformer

            # 1. Apply datetime feature extraction (Step 12 - Transformation)
            if st.session_state.datetime_features_enabled:
                datetime_cols_in_X = [col for col in X_for_pipeline.columns if pd.api.types.is_datetime64_any_dtype(X_for_pipeline[col])]
                X_for_pipeline = apply_datetime_features(X_for_pipeline, datetime_cols_in_X)
                st.session_state.datetime_cols_extracted = datetime_cols_in_X # Store for prediction
                if datetime_cols_in_X:
                    st.write(f"✅ Extracted date/time features and dropped original columns for: {', '.join(datetime_cols_in_X)}")
            else:
                st.session_state.datetime_cols_extracted = []
            update_progress()

            # 2. Add synthetic feature if enabled (Step 12 - Transformation)
            if st.session_state.add_synthetic_feature_enabled:
                # For training, we can use the actual target to create the synthetic feature
                temp_target_for_synth_creation = df[target].copy()
                if st.session_state.task_type == "Classification" and st.session_state.label_encoder: # Check if encoder exists
                    temp_target_for_synth_creation = st.session_state.label_encoder.transform(temp_target_for_synth_creation.astype(str))
                elif temp_target_for_synth_creation.dtype == 'object' or pd.api.types.is_categorical_dtype(temp_target_for_synth_creation):
                     # If target wasn't encoded for some reason, and it's categorical, encode it here locally for synth feature creation
                     temp_le_for_synth = LabelEncoder()
                     temp_target_for_synth_creation = temp_le_for_synth.fit_transform(temp_target_for_synth_creation.astype(str))

                # Ensure the length matches, important after any previous filtering (e.g., outlier removal in X)
                if len(X_for_pipeline) == len(temp_target_for_synth_creation):
                    noise = np.random.normal(0, 0.01, size=len(temp_target_for_synth_creation))
                    X_for_pipeline['synthetic_highly_correlated_feature'] = temp_target_for_synth_creation + noise
                    st.info("Added a 'synthetic_highly_correlated_feature' for demonstration during training.")
                else:
                    st.warning("Skipping synthetic feature creation during training due to length mismatch after other preprocessing steps. This may affect consistency.")


            # Store the columns of X_for_pipeline BEFORE train_test_split and ColumnTransformer
            # This is the exact set of columns that the preprocessing pipeline (ColumnTransformer) will expect.
            st.session_state.processed_training_features_at_pipeline_entry = X_for_pipeline.columns.tolist()


            # Identify numerical and categorical columns *after* custom feature engineering but *before* ColumnTransformer
            numerical_cols = X_for_pipeline.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = X_for_pipeline.select_dtypes(include=['object', 'category']).columns.tolist()

            # --- Handle Outliers and Imbalance (applied to X_for_pipeline and y) ---
            if outliers:
                if numerical_cols:
                    iso_forest = IsolationForest(contamination=0.05, n_jobs=-1, random_state=42)
                    mask = iso_forest.fit_predict(X_for_pipeline[numerical_cols]) != -1
                    X_for_pipeline, y = X_for_pipeline[mask], y[mask]
                    st.write(f"📈 Removed {len(mask) - sum(mask)} outliers using IsolationForest.")
                else:
                    st.info("Outlier handling skipped: No numerical features to process for outlier detection.")
                update_progress()

            if imbalance:
                if st.session_state.task_type == "Classification" and len(np.unique(y)) == 2:
                    y_temp_series = pd.Series(y)
                    if (y_temp_series.value_counts().min() / y_temp_series.value_counts().max() < 0.5):
                        # Ensure alignment of indices before concat
                        X_reset = X_for_pipeline.reset_index(drop=True)
                        y_reset = y_temp_series.reset_index(drop=True).rename(target)
                        df_bal = pd.concat([X_reset, y_reset], axis=1)

                        major_class = y_temp_series.value_counts().idxmax()
                        df_major = df_bal[df_bal[target] == major_class]
                        df_minor = df_bal[df_bal[target] != major_class]
                        df_minor_up = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
                        df_balanced = pd.concat([df_major, df_minor_up])
                        X_for_pipeline = df_balanced.drop(columns=[target])
                        y = df_balanced[target].values if not isinstance(y, pd.Series) else df_balanced[target]
                        st.write("⚖️ Handled class imbalance by upsampling the minority class.")
                    else:
                        st.info("Imbalance handling skipped: Already sufficiently balanced or not a significant imbalance.")
                else:
                    st.info("Imbalance handling skipped: Not binary classification.")
                update_progress()


            # Define the preprocessing pipeline for the ColumnTransformer

            # Calculate total preprocessing steps for the progress bar (excluding outlier/imbalance as they are separate)
            total_pipeline_steps = sum([impute, scale, ohe_encode, poly_features, select, reduce]) # These are in the pipeline
            if total_pipeline_steps == 0:
                total_pipeline_steps = 1 # Prevent division by zero if no steps are active

            # Split the data into training and testing sets *before* pipeline fitting
            X_train, X_test, y_train, y_test = train_test_split(
                X_for_pipeline, y, test_size=0.3, random_state=42,
                stratify=y if st.session_state.task_type == "Classification" and len(np.unique(y)) == 2 else None
            )
            # FIX: Re-encode y_train and y_test AFTER all filtering and splitting
            if st.session_state.task_type == "Classification":
                final_encoder = LabelEncoder()
                y_train = final_encoder.fit_transform(y_train)
                try:
                    y_test = final_encoder.transform(y_test)
                except ValueError:
                    st.error("Test set contains classes not present in the training set after filtering. Cannot proceed.")
                    st.stop()
                st.session_state.final_encoder = final_encoder

            st.success("Data split into training and testing sets.")

            # Create and fit the preprocessing pipeline (ColumnTransformer)
            with st.spinner("Building and fitting preprocessing pipeline..."):
                # Ensure y_train is an array for SelectKBest, if it's used within the pipeline
                y_train_for_selector = y_train.values if isinstance(y_train, pd.Series) else y_train
                preprocessing_pipeline = create_preprocessing_pipeline_for_transformer(
                    numerical_cols, categorical_cols, impute, scale, ohe_encode, poly_features, select, reduce, st.session_state.task_type
                )

                try:
                    X_train_transformed = preprocessing_pipeline.fit_transform(X_train, y_train_for_selector) # y_train is needed for SelectKBest fit
                    X_test_transformed = preprocessing_pipeline.transform(X_test)

                    # Get feature names after transformation by ColumnTransformer
                    try:
                        transformed_feature_names = preprocessing_pipeline.get_feature_names_out()
                    except Exception:
                        st.warning("Could not get feature names from ColumnTransformer directly. Naming features generically (e.g., 'feature_0', 'feature_1').")
                        transformed_feature_names = [f'feature_{i}' for i in range(X_train_transformed.shape[1])]

                    # --- NEW: Sanitize feature names here (Step 5 - Data Quality) ---
                    sanitized_transformed_feature_names = sanitize_feature_names(transformed_feature_names)

                    X_train = pd.DataFrame(X_train_transformed, columns=sanitized_transformed_feature_names, index=X_train.index)
                    X_test = pd.DataFrame(X_test_transformed, columns=sanitized_transformed_feature_names, index=X_test.index)

                    st.session_state.X_transformed_cols = X_train.columns.tolist() # These are the final feature names after ColumnTransformer
                    st.session_state.X_train_shape = X_train.shape
                    st.success("✅ Preprocessing pipeline built and applied.")
                    update_progress() # Update progress for the whole pipeline step

                except Exception as pp_e:
                    st.error(f"❌ Error during preprocessing pipeline application: {str(pp_e)}")
                    st.exception(pp_e)
                    st.stop()


            # Final validation after all preprocessing
            if X_train.isnull().any().any():
                st.error("🚨 NaN values still present in X_train after preprocessing. Please check your data.")
                st.dataframe(X_train[X_train.isnull().any(axis=1)])
                st.stop()
            if X_test.isnull().any().any():
                st.error("🚨 NaN values still present in X_test after preprocessing. Please check your data.")
                st.dataframe(X_test[X_test.isnull().any(axis=1)])
                st.stop()
            if y.isnull().any() if isinstance(y, pd.Series) else np.isnan(y).any():
                st.error("🚨 NaN values still present in target after preprocessing. Please check your data.")
                st.stop()

            st.session_state.feature_names = X_train.columns.tolist() # Update feature names (final transformed names)

            # --- Model Selection & Hyperparameter Optimization ---
            st.sidebar.title("🎛️ Hyperparameter Tuning")
            n_estimators = st.sidebar.slider("n_estimators (RF, XGB, LGB)", 50, 500, 100, 50, key="n_estimators_slider")
            max_depth = st.sidebar.slider("max_depth (RF, XGB, LGB)", 2, 20, 8, key="max_depth_slider")
            learning_rate = st.sidebar.slider("learning_rate (XGB, LGB)", 0.01, 0.5, 0.1, 0.01, key="learning_rate_slider")
            C_param = st.sidebar.slider("C (for SVR/SVC/LogisticRegression)", 0.1, 10.0, 1.0, 0.1, key="C_param_slider") # Updated slider label
            alpha_param = st.sidebar.slider("alpha (for Ridge/Lasso/ElasticNet)", 0.1, 10.0, 1.0, 0.1, key="alpha_param_slider") # Updated slider label
            l1_ratio_param = st.sidebar.slider("l1_ratio (for ElasticNet)", 0.0, 1.0, 0.5, 0.05, key="l1_ratio_param_slider") # New slider for l1_ratio
            n_neighbors_param = st.sidebar.slider("n_neighbors (for KNN)", 3, 15, 5, key="n_neighbors_slider")


            st.subheader("🤖 Model Training")
            model_container = st.container() # Use a container for dynamic content
            with model_container:
                # Model choice depends on task type
                if st.session_state.task_type == "Classification":
                    model_choice = st.selectbox("Choose ML Model", ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression", "SVC", "K-Nearest Neighbors", "Gaussian Naive Bayes"])
                else: # Regression
                    model_choice = st.selectbox("Choose ML Model", ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "SVR", "K-Nearest Neighbors", "Random Forest", "XGBoost", "LightGBM"]) # Added ElasticNet

                enable_hpo = st.checkbox("⚙️ Enable Hyperparameter Optimization (GridSearchCV)", value=False, key="enable_hpo_checkbox")

                param_grid = {}
                if st.session_state.task_type == "Classification":
                    if model_choice == "Random Forest":
                        estimator = RandomForestClassifier(random_state=42)
                        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
                    elif model_choice == "XGBoost":
                        estimator = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2]}
                    elif model_choice == "LightGBM":
                        estimator = LGBMClassifier(random_state=42)
                        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, -1], 'learning_rate': [0.01, 0.1, 0.2]}
                    elif model_choice == "Logistic Regression":
                        estimator = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000) # Added solver and max_iter
                        param_grid = {'C': [0.1, 1.0, 10.0]}
                    elif model_choice == "SVC":
                        estimator = SVC(random_state=42, probability=True) # probability=True for ROC AUC
                        param_grid = {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']}
                    elif model_choice == "K-Nearest Neighbors":
                        estimator = KNeighborsClassifier()
                        param_grid = {'n_neighbors': [3, 5, 7, 9]}
                    elif model_choice == "Gaussian Naive Bayes":
                        estimator = GaussianNB()
                        param_grid = {} # No hyperparameters to tune usually
                    scoring_metric = 'accuracy'
                else: # Regression
                    if model_choice == "Linear Regression":
                        estimator = LinearRegression()
                        param_grid = {}
                    elif model_choice == "Ridge":
                        estimator = Ridge(random_state=42)
                        param_grid = {'alpha': [0.1, 1.0, 10.0]}
                    elif model_choice == "Lasso":
                        estimator = Lasso(random_state=42)
                        param_grid = {'alpha': [0.1, 1.0, 10.0]}
                    elif model_choice == "ElasticNet": # Added ElasticNet to HPO options
                        estimator = ElasticNet(random_state=42)
                        param_grid = {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]}
                    elif model_choice == "SVR":
                        estimator = SVR()
                        param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
                    elif model_choice == "K-Nearest Neighbors":
                        estimator = KNeighborsRegressor()
                        param_grid = {'n_neighbors': [3, 5, 7, 9]}
                    elif model_choice == "Random Forest":
                        estimator = RandomForestRegressor(random_state=42)
                        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
                    elif model_choice == "XGBoost":
                        estimator = XGBRegressor(objective='reg:squarederror', random_state=42)
                        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.2]}
                    elif model_choice == "LightGBM":
                        estimator = LGBMRegressor(objective='regression', random_state=42)
                        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, -1], 'learning_rate': [0.01, 0.1, 0.2]}
                    scoring_metric = 'r2'

                # --- Re-encode y_train and y_test for models strict about 0-indexed contiguous labels (Step 15) ---
                # This is crucial for models like XGBoost, LightGBM, and SVC which expect labels starting from 0.
                # If classes were removed during outlier handling or imbalance, LabelEncoder might leave gaps.
                # This re-encoding ensures contiguous labels.
                y_train_processed = y_train
                y_test_processed = y_test
                temp_target_encoder_for_model = None

                if st.session_state.task_type == "Classification" and model_choice in ["XGBoost", "LightGBM", "SVC"]:
                    st.info(f"Re-encoding target labels for {model_choice} to ensure 0-indexed contiguous classes (due to potential class removal by earlier steps).")

                    temp_target_encoder_for_model = LabelEncoder()
                    y_train_processed = temp_target_encoder_for_model.fit_transform(y_train)

                    # Use a try-except block for transforming y_test_processed
                    # If y_test_processed contains classes not seen in y_train_processed, this will raise a ValueError.
                    # For simplicity here, we'll stop execution if unmappable classes are found.
                    try:
                        y_test_processed = temp_target_encoder_for_model.transform(y_test)
                    except ValueError as ve:
                        st.error(f"❌ Error: Test set contains target classes not seen in the training set after re-encoding for {model_choice}. Cannot proceed. Please ensure your data split and class distribution are consistent. Error: {str(ve)}")
                        st.stop()

                    st.write(f"  Unique labels in y_train after re-encoding for model: {np.unique(y_train_processed)}")
                    st.write(f"  Unique labels in y_test after re-encoding for model: {np.unique(y_test_processed)}")
                    st.session_state.target_encoder = temp_target_encoder_for_model # Store this encoder for inverse_transform in predictions

                # Add a flag to control whether HPO is performed
                perform_hpo = enable_hpo and bool(param_grid) # Check if HPO is enabled AND there are params to tune

                if perform_hpo:
                    if st.session_state.task_type == "Classification":
                        min_class_count_in_train = pd.Series(y_train_processed).value_counts().min()
                        if min_class_count_in_train < 2:
                            st.warning(f"Smallest class in training data has only {min_class_count_in_train} sample(s). Cannot perform GridSearchCV (requires n_splits >= 2). Hyperparameter Optimization will be skipped.")
                            perform_hpo = False # Force skip HPO
                        elif min_class_count_in_train < 5:
                            st.info(f"Smallest class in training data has only {min_class_count_in_train} samples. Adjusting GridSearchCV n_splits to {min_class_count_in_train}.")
                            grid_search_cv_param = min_class_count_in_train
                        else:
                            grid_search_cv_param = 5
                            st.info(f"Using cv={grid_search_cv_param} for GridSearchCV based on training data class distribution.")
                    else: # Regression task
                        grid_search_cv_param = 5 # Default for regression

                    if perform_hpo: # Re-check after potential skipping for classification
                        with st.spinner(f'🚀 Running GridSearchCV for {model_choice}...'):
                            grid_search = GridSearchCV(estimator, param_grid, cv=grid_search_cv_param, scoring=scoring_metric, n_jobs=-1, verbose=1)
                            grid_search.fit(X_train, y_train_processed) # Use processed y_train
                            model = grid_search.best_estimator_
                            st.success(f"✅ HPO complete. Best parameters for {model_choice}: {grid_search.best_params_}")
                            st.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
                    else: # If HPO was skipped for classification, train manually
                        # Manual model training using sidebar sliders
                        if st.session_state.task_type == "Classification":
                            if model_choice == "Random Forest":
                                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                            elif model_choice == "XGBoost":
                                st.write(f"Debug: X_train columns before XGBoost training: {X_train.columns.tolist()}") # DEBUG
                                model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss', random_state=42)
                            elif model_choice == "LightGBM":
                                model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
                            elif model_choice == "Logistic Regression":
                                model = LogisticRegression(C=C_param, random_state=42, solver='liblinear', max_iter=1000)
                            elif model_choice == "SVC":
                                model = SVC(C=C_param, random_state=42, probability=True)
                            elif model_choice == "K-Nearest Neighbors":
                                model = KNeighborsClassifier(n_neighbors=n_neighbors_param)
                            elif model_choice == "Gaussian Naive Bayes":
                                model = GaussianNB()
                        else: # Regression
                            if model_choice == "Linear Regression":
                                model = LinearRegression()
                            elif model_choice == "Ridge":
                                model = Ridge(alpha=alpha_param, random_state=42)
                            elif model_choice == "Lasso":
                                model = Lasso(alpha=alpha_param, random_state=42)
                            elif model_choice == "ElasticNet": # Added ElasticNet manual training
                                model = ElasticNet(alpha=alpha_param, l1_ratio=l1_ratio_param, random_state=42)
                            elif model_choice == "SVR":
                                model = SVR(C=C_param)
                            elif model_choice == "K-Nearest Neighbors":
                                model = KNeighborsRegressor(n_neighbors=n_neighbors_param)
                            elif model_choice == "Random Forest":
                                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                            elif model_choice == "XGBoost":
                                st.write(f"Debug: X_train columns for {model_name} comparison: {X_train.columns.tolist()}") # DEBUG
                                model = XGBRegressor(objective='reg:squarederror', random_state=42)
                            elif model_choice == "LightGBM":
                                model = LGBMRegressor(objective='regression', random_state=42)

                        with st.spinner(f'🚂 Training {model_choice}...'):
                            model.fit(X_train, y_train_processed) # Use processed y_train
                            st.success("✅ Model trained successfully!")

                else: # Original else block for manual training if HPO is generally disabled
                    # Manual model training using sidebar sliders
                    if st.session_state.task_type == "Classification":
                        if model_choice == "Random Forest":
                            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                        elif model_choice == "XGBoost":
                            st.write(f"Debug: X_train columns before XGBoost training: {X_train.columns.tolist()}") # DEBUG
                            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, use_label_encoder=False, eval_metric='logloss', random_state=42)
                        elif model_choice == "LightGBM":
                            model = LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
                        elif model_choice == "Logistic Regression":
                            model = LogisticRegression(C=C_param, random_state=42, solver='liblinear', max_iter=1000)
                        elif model_choice == "SVC":
                            model = SVC(C=C_param, random_state=42, probability=True)
                        elif model_choice == "K-Nearest Neighbors":
                            model = KNeighborsClassifier(n_neighbors=n_neighbors_param)
                        elif model_choice == "Gaussian Naive Bayes":
                            model = GaussianNB()
                    else: # Regression
                        if model_choice == "Linear Regression":
                            model = LinearRegression()
                        elif model_choice == "Ridge":
                            model = Ridge(alpha=alpha_param, random_state=42)
                        elif model_choice == "Lasso":
                            model = Lasso(alpha=alpha_param, random_state=42)
                        elif model_choice == "ElasticNet": # Added ElasticNet manual training
                            model = ElasticNet(alpha=alpha_param, l1_ratio=l1_ratio_param, random_state=42)
                        elif model_choice == "SVR":
                            model = SVR(C=C_param)
                        elif model_choice == "K-Nearest Neighbors":
                            model = KNeighborsRegressor(n_neighbors=n_neighbors_param)
                        elif model_choice == "Random Forest":
                            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                        elif model_choice == "XGBoost":
                            st.write(f"Debug: X_train columns for {model_name} comparison: {X_train.columns.tolist()}") # DEBUG
                            model = XGBRegressor(objective='reg:squarederror', random_state=42)
                        elif model_choice == "LightGBM":
                            model = LGBMRegressor(objective='regression', random_state=42)

                    with st.spinner(f'🚂 Training {model_choice}...'):
                        model.fit(X_train, y_train_processed) # Use processed y_train
                        st.success("✅ Model trained successfully!")


                st.subheader("Cross-Validation Performance (Manual Mode)")
                # Define a pipeline for cross-validation that includes preprocessing and the model
                # This ensures that preprocessing is applied correctly within each CV fold
                cv_pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessing_pipeline), # Use the already defined preprocessor
                    ('model', model) # Use the trained model or best estimator
                ])

                if st.session_state.task_type == "Classification":
                    min_class_count_overall = pd.Series(y).value_counts().min()
                    if min_class_count_overall < 2:
                        st.warning(f"Smallest class in overall data has only {min_class_count_overall} sample(s). Cannot perform cross-validation (requires n_splits >= 2). Skipping cross-validation score calculation.")
                    else:
                        cv_param_for_cross_val_score = min(min_class_count_overall, 5) # Cap at 5 folds
                        st.info(f"Using cv={cv_param_for_cross_val_score} for cross_val_score based on overall data class distribution.")
                        cv_scores = cross_val_score(cv_pipeline, X_for_pipeline, y, cv=cv_param_for_cross_val_score, scoring='accuracy', n_jobs=-1)
                        st.write(f"Accuracy ({cv_param_for_cross_val_score}-Fold CV): Mean={cv_scores.mean():.2f}, Std={cv_scores.std():.2f}")
                else: # Regression
                    # Regression always defaults to 5-fold CV as there's no class imbalance issue
                    cv_scores = cross_val_score(cv_pipeline, X_for_pipeline, y, cv=5, scoring='r2', n_jobs=-1)
                    st.write(f"R2 Score (5-Fold CV): Mean={cv_scores.mean():.2f}, Std={cv_scores.std():0.2f}")


            # Save the trained model to session state
            st.session_state.model = model
            # Store the full pipeline for prediction
            st.session_state.full_ml_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessing_pipeline),
                ('classifier_regressor', model)
            ])


            # --- Evaluation and Visualizations Tabs ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Metrics", "📈 Visualizations", "🧠 Interpretability (SHAP)", "🏆 Model Comparison", "💾 Model Export"])

            with tab1:
                y_pred = model.predict(X_test)

                st.subheader(f"{st.session_state.task_type} Model Metrics")

                if st.session_state.task_type == "Classification":
                    unique_classes_in_y_test = np.unique(y_test_processed) # Use processed y_test for metrics
                    if unique_classes_in_y_test.size == 1:
                        st.warning(f"⚠️ **Warning:** The test set contains only one class (class: {unique_classes_in_y_test[0]}). "
                                   "Accuracy and other metrics might be misleading or trivially 0%/100%. "
                                   "Consider a larger dataset or different train/test split strategy.")


                    current_accuracy = accuracy_score(y_test_processed, y_pred) # Use processed y_test
                    if current_accuracy == 0.0:
                        st.warning("🚨 **Warning:** Achieved 0.00% accuracy. This often indicates issues such as: "
                                   "very small test set, extreme class imbalance, or a model failing to learn. "
                                   "Please check your dataset and preprocessing steps.")
                    elif current_accuracy == 1.0:
                         st.info("🎉 **Info:** Achieved 100.00% accuracy. While great, for small datasets this might indicate "
                                 "the test set contains only one class or that the problem is trivially solvable. "
                                 "Always consider the context of your data.")


                    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") and np.unique(y_test_processed).size == 2 else None # Use processed y_test for size check

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{current_accuracy:.2%}")
                    with col2:
                        st.metric("F1 Score (Weighted)", f"{f1_score(y_test_processed, y_pred, average='weighted'):.2f}") # Use processed y_test
                    with col3:
                        if y_proba is not None:
                            st.metric("AUC-ROC", f"{roc_auc_score(y_test_processed, y_proba):.2f}") # Use processed y_test
                        else:
                            st.info("AUC-ROC not available (not binary classification or model lacks predict_proba)")

                    st.write("Classification Report:")

                    # Determine the actual labels present in the test set and predictions
                    all_observed_labels = np.unique(np.concatenate((y_test_processed, y_pred)))
                    target_names_for_report = None

                    encoder_to_use = None
                    if 'target_encoder' in st.session_state and st.session_state.target_encoder:
                        encoder_to_use = st.session_state.target_encoder
                    elif st.session_state.label_encoder:
                        encoder_to_use = st.session_state.label_encoder

                    if encoder_to_use is not None:
                        try:
                            # Inverse transform only the labels that are actually observed in test data/predictions
                            target_names_for_report = [str(x) for x in encoder_to_use.inverse_transform(all_observed_labels.astype(int))]
                        except ValueError:
                            # Fallback if inverse_transform fails (e.g., due to unseen labels in test set if not handled by encoder)
                            st.warning("Could not inverse transform all observed labels for classification report. Using numerical labels as target names.")
                            target_names_for_report = [str(int(x)) for x in all_observed_labels]
                    else:
                        # If no encoder, use the numerical labels directly
                        target_names_for_report = [str(int(x)) for x in all_observed_labels]


                    st.text(classification_report(y_test_processed.astype(int) if y_test_processed.dtype != int else y_test_processed,
                                                  y_pred.astype(int) if y_pred.dtype != int else y_pred,
                                                  target_names=target_names_for_report, # Pass the dynamically generated target names
                                                  zero_division='warn' # Add zero_division to avoid warnings for classes with no true samples or no predicted samples
                                                 ))


                    st.write("Confusion Matrix:")
                    cm = confusion_matrix(y_test_processed, y_pred) # Use processed y_test
                    fig, ax = plt.subplots()
                    cm_labels = target_names_for_report if target_names_for_report else np.unique(y_test_processed)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=cm_labels,
                                yticklabels=cm_labels)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    ax.set_title('Confusion Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                else: # Regression Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{r2_score(y_test_processed, y_pred):.2f}") # Use processed y_test
                    with col2:
                        st.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test_processed, y_pred):.2f}") # Use processed y_test
                    with col3:
                        st.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test_processed, y_pred):.2f}") # Use processed y_test

                    st.write("Predicted vs Actual Plot:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.regplot(x=y_test_processed, y=y_pred, ax=ax, scatter_kws={'alpha':0.3}) # Use processed y_test
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title('Actual vs Predicted Values')

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)



            with tab2:
                st.subheader(f"{st.session_state.task_type} Model Visualizations")
                if st.session_state.task_type == "Classification" and np.unique(y_test_processed).size == 2 and y_proba is not None: # Use processed y_test
                    st.write("ROC Curve:")
                    fpr, tpr, _ = roc_curve(y_test_processed, y_proba) # Use processed y_test
                    fig, ax = plt.subplots()
                    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                elif st.session_state.task_type == "Classification" and (np.unique(y_test_processed).size != 2 or y_proba is None): # Use processed y_test
                    st.info("ROC Curve is typically for binary classification with probability outputs.")

                # Feature importance plot
                st.write("Feature Importance:")
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance.head(10), ax=ax)
                    ax.set_title('Top 10 Feature Importances')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    st.dataframe(importance)
                elif hasattr(model, 'coef_') and (model.coef_.ndim == 1 or (model.coef_.ndim == 2 and model.coef_.shape[0] == 1)): # Linear models
                    # For multiclass LogisticRegression, coef_ is (n_classes, n_features)
                    # For binary LogisticRegression and other linear models, coef_ is (n_features,) or (1, n_features)
                    if model.coef_.ndim == 2 and model.coef_.shape[0] > 1:
                        st.info("Feature importance for multi-class linear models (e.g., Logistic Regression) can be complex. Displaying average absolute coefficients.")
                        abs_coefs = np.mean(np.abs(model.coef_), axis=0)
                    else:
                        abs_coefs = np.abs(model.coef_).flatten()

                    importance = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': abs_coefs
                    }).sort_values('Importance', ascending=False)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importance.head(10), ax=ax)
                    ax.set_title('Top 10 Feature Importances (Absolute Coefficients)')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.dataframe(importance)
                else:
                    st.info("Feature importance not directly available for this model type or structure.")


            with tab3: # SHAP Interpretability Tab
                st.subheader("🧠 SHAP Model Interpretability")
                st.info("""
                    SHAP (SHapley Additive Explanations) helps explain individual predictions and overall model behavior.
                    It can be computationally intensive for large datasets.
                """)

                generate_shap = st.checkbox("Generate SHAP Explanations", key="generate_shap_checkbox")

                if generate_shap:
                    try:
                        # SHAP needs data to match the model's expected input feature names
                        X_test_for_shap = X_test.copy()

                        # Ensure X_test_for_shap columns and order match the features model was trained on
                        if st.session_state.feature_names:
                            X_test_for_shap = X_test_for_shap.reindex(columns=st.session_state.feature_names, fill_value=0)
                        else:
                            st.warning("SHAP: st.session_state.feature_names is empty. Cannot ensure column consistency for SHAP plots.")
                            st.stop()

                        # Ensure X_test_for_shap is fully numerical and float for SHAP
                        # Attempt to convert all columns to numeric, coercing errors to NaN, then fill NaNs.
                        for col in X_test_for_shap.columns:
                            X_test_for_shap[col] = pd.to_numeric(X_test_for_shap[col], errors='coerce').fillna(0)
                        X_test_for_shap = X_test_for_shap.astype(np.float64)


                        st.write(f"Debug SHAP: X_test_for_shap initial shape before sampling/reset: {X_test_for_shap.shape}")
                        st.write(f"Debug SHAP: X_test_for_shap columns before sampling/reset: {X_test_for_shap.columns.tolist()}")

                        if X_test_for_shap.empty:
                            st.warning("⚠️ SHAP explanation skipped: Test set (after preprocessing) is empty (0 samples).")
                            st.stop()
                        if X_test_for_shap.shape[1] == 0:
                            st.warning("⚠️ SHAP explanation skipped: Test set (after preprocessing) has 0 features (columns). Cannot generate SHAP plots.")
                            st.stop()
                        if X_test_for_shap.shape[0] < 2:
                             st.warning("⚠️ SHAP explanation skipped: Not enough samples in test set for meaningful SHAP plots (requires at least 2).")
                             st.stop()

                        shap_raw_values = None
                        explainer = None

                        # Determine the effective X for SHAP calculation
                        # For performance, limit the number of samples SHAP calculates on if X_test is very large
                        X_for_shap_calculation = X_test_for_shap.copy()
                        max_samples_for_shap_calculation = 1000 # Limit to 1000 samples for SHAP calculation
                        if X_for_shap_calculation.shape[0] > max_samples_for_shap_calculation:
                            st.info(f"Test set has {X_for_shap_calculation.shape[0]} samples. Sampling down to {max_samples_for_shap_calculation} for SHAP value calculation to improve performance.")
                            # Crucial: reset index AFTER sampling to ensure contiguous index for SHAP
                            X_for_shap_calculation = X_for_shap_calculation.sample(n=max_samples_for_shap_calculation, random_state=42).reset_index(drop=True)
                        else:
                            # Even if not sampled, reset index to ensure consistency
                            X_for_shap_calculation = X_for_shap_calculation.reset_index(drop=True)

                        st.write(f"Debug SHAP: X_for_shap_calculation shape after sampling/reset: {X_for_shap_calculation.shape}")
                        st.write(f"Debug SHAP: X_for_shap_calculation index type: {type(X_for_shap_calculation.index)}")
                        st.write(f"Debug SHAP: X_for_shap_calculation index: {X_for_shap_calculation.index}")


                        with st.spinner("Calculating SHAP values... This may take a while."):
                            try:
                                # TreeExplainer for tree-based models
                                if model_choice in ["Random Forest", "XGBoost", "LightGBM"]:
                                    # TreeExplainer can handle multi-output natively for multi-class classification
                                    explainer = shap.TreeExplainer(model)
                                    shap_raw_values = explainer.shap_values(X_for_shap_calculation)
                                # LinearExplainer for linear models
                                elif model_choice in ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "Logistic Regression"]:
                                    # Ensure X_train background data also has a clean index if it was subject to row removals/resampling
                                    X_train_for_explainer_background = X_train.copy().reset_index(drop=True)
                                    explainer = shap.LinearExplainer(model, X_train_for_explainer_background) # Use X_train for background for LinearExplainer
                                    shap_raw_values = explainer.shap_values(X_for_shap_calculation)
                                # KernelExplainer for other models (more general but slower)
                                else:
                                    X_train_for_shap_background = X_train.copy()
                                    # Ensure background data is numeric and float for KernelExplainer
                                    for col in X_train_for_shap_background.columns:
                                        X_train_for_shap_background[col] = pd.to_numeric(X_train_for_shap_background[col], errors='coerce').fillna(0)
                                    X_train_for_shap_background = X_train_for_shap_background.astype(np.float64).reset_index(drop=True) # Reset index for background too

                                    n_samples_for_shap_background = min(100, X_train_for_shap_background.shape[0])
                                    if n_samples_for_shap_background == 0:
                                        st.warning("Cannot generate SHAP plots: Background dataset for KernelExplainer is empty.")
                                        st.stop()

                                    background_data_for_kernel = shap.sample(X_train_for_shap_background, nsamples=n_samples_for_shap_background, random_state=42)

                                    explainer = shap.KernelExplainer(model.predict, background_data_for_kernel)
                                    shap_raw_values = explainer.shap_values(X_for_shap_calculation)

                            except Exception as shap_calc_e:
                                st.error(f"❌ Error during SHAP values calculation: {str(shap_calc_e)}. This often indicates issues with model type or data format for SHAP.")
                                st.exception(shap_calc_e)
                                st.stop()

                        if shap_raw_values is None:
                            st.warning("SHAP values could not be calculated. Skipping plots.")
                            st.stop()

                        # Determine the actual shap_values to use for plotting and ensure it's a consistent 2D array or list of 2D arrays
                        shap_values_for_plotting = None
                        if isinstance(shap_raw_values, list):
                            # Ensure shap_raw_values elements are also floats for plotting
                            shap_raw_values = [np.array(val).astype(np.float64) for val in shap_raw_values]

                            if st.session_state.task_type == "Classification" and len(np.unique(y_train_processed)) == 2:
                                # Binary classification: use SHAP values for the positive class (index 1)
                                shap_values_for_plotting = shap_raw_values[1] # This is a 2D array (samples, features)
                            elif st.session_state.task_type == "Classification" and len(np.unique(y_train_processed)) > 2:
                                # Multi-class: Keep as list of 2D arrays
                                shap_values_for_plotting = shap_raw_values # This is a list of 2D arrays
                            else: # Should not happen if task_type is well-defined, but for safety
                                st.warning("SHAP: Ambiguous list of SHAP values. Using the first element for plotting.")
                                shap_values_for_plotting = shap_raw_values[0]
                        else: # Already a single numpy array (e.g., for regression or some multi-class KernelExplainer outputs)
                            shap_values_for_plotting = np.array(shap_raw_values).astype(np.float64) # Ensure it's float

                        # IMPORTANT: X_test_display_for_plot must be the exact data that SHAP values were calculated *for*.
                        # This means it's X_for_shap_calculation. Now keep it as a DataFrame.
                        X_test_display_for_plot = X_for_shap_calculation.copy()

                        # Final checks (should ideally pass if logic is correct now)
                        # When shap_values_for_plotting is a list of 2D arrays, take shape from first element
                        if isinstance(shap_values_for_plotting, list):
                            num_shap_rows = shap_values_for_plotting[0].shape[0] if shap_values_for_plotting else 0
                            num_shap_cols = shap_values_for_plotting[0].shape[1] if shap_values_for_plotting else 0
                        else: # It's a single NumPy array
                            num_shap_rows = shap_values_for_plotting.shape[0]
                            num_shap_cols = shap_values_for_plotting.shape[1] if shap_values_for_plotting.ndim > 1 else 0 # 0 for 1D arrays

                        if num_shap_rows == 0:
                            st.warning("SHAP: No SHAP values rows to plot after final processing. Skipping plots.")
                            st.stop()

                        # If shap_values_for_plotting is a 3D array (samples, features, classes)
                        # then num_shap_cols will be features, and its last dim is classes.
                        # We need to compare X_test_display_for_plot.shape[1] with the 'features' dimension of SHAP.
                        if shap_values_for_plotting.ndim == 3: # (samples, features, classes)
                            if X_test_display_for_plot.shape[1] != shap_values_for_plotting.shape[1]:
                                 st.error(f"SHAP Fatal Error: Column count mismatch (features) after SHAP calculation ({shap_values_for_plotting.shape[1]} SHAP features vs {X_test_display_for_plot.shape[1]} data features). This should not happen with current logic.")
                                 st.stop()
                        else: # 2D array (samples, features)
                            if X_test_display_for_plot.shape[1] != num_shap_cols:
                                st.error(f"SHAP Fatal Error: Column count mismatch after SHAP calculation ({num_shap_cols} SHAP cols vs {X_test_display_for_plot.shape[1]} data cols). This should not happen with current logic.")
                                st.stop()

                        st.write(f"Debug SHAP: X_test_display_for_plot final shape before plot: {X_test_display_for_plot.shape}")
                        st.write(f"Debug SHAP: shap_values_for_plotting final shape before plot: {shap_values_for_plotting.shape if isinstance(shap_values_for_plotting, np.ndarray) else 'List of ' + str([s.shape for s in shap_values_for_plotting])}")
                        st.write(f"Debug SHAP: X_test_display_for_plot is Pandas DataFrame: {isinstance(X_test_display_for_plot, pd.DataFrame)}")
                        st.write(f"Debug SHAP: X_test_display_for_plot columns (final): {X_test_display_for_plot.columns.tolist()}")

                        st.write("#### SHAP Summary Plot (Feature Importance & Impact)")
                        with st.spinner("Generating SHAP Summary Plot..."):
                            if isinstance(shap_values_for_plotting, list): # Multi-class with list of 2D arrays
                                st.info("Multi-class SHAP values generated. Displaying summary for each class.")
                                for i, class_shap_values in enumerate(shap_values_for_plotting):
                                    fig, ax = plt.subplots(figsize=(10,6))
                                    # Passing the DataFrame here
                                    shap.summary_plot(class_shap_values, X_test_display_for_plot, show=False, color_bar_label=f"SHAP Value for Class {i}")
                                    # Safely get class name from label encoder if available
                                    class_name = str(i)
                                    if temp_target_encoder_for_model and i < len(temp_target_encoder_for_model.classes_):
                                        class_name = temp_target_encoder_for_model.inverse_transform([i])[0]
                                    elif st.session_state.label_encoder and i < len(st.session_state.label_encoder.classes_):
                                        class_name = st.session_state.label_encoder.inverse_transform([i])[0]
                                    ax.set_title(f"SHAP Summary for Class: {class_name}")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                            elif shap_values_for_plotting.ndim == 3: # Multi-class with 3D array (samples, features, classes)
                                st.info("Multi-class SHAP values generated (3D array). Displaying summary for each class.")
                                for i in range(shap_values_for_plotting.shape[2]):
                                    fig, ax = plt.subplots(figsize=(10,6))
                                    shap.summary_plot(shap_values_for_plotting[:, :, i], X_test_display_for_plot, show=False, color_bar_label=f"SHAP Value for Class {i}")
                                    class_name = str(i)
                                    if temp_target_encoder_for_model and i < len(temp_target_encoder_for_model.classes_):
                                        class_name = temp_target_encoder_for_model.inverse_transform([i])[0]
                                    elif st.session_state.label_encoder and i < len(st.session_state.label_encoder.classes_):
                                        class_name = st.session_state.label_encoder.inverse_transform([i])[0]
                                    ax.set_title(f"SHAP Summary for Class: {class_name}")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                            else: # Binary Classification or Regression (single 2D SHAP array)
                                fig, ax = plt.subplots(figsize=(10,6))
                                # Passing the DataFrame here
                                shap.summary_plot(shap_values_for_plotting, X_test_display_for_plot, show=False)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close(fig)

                        st.write("#### SHAP Dependence Plot (Feature Interaction)")
                        shap_feature_to_plot = st.selectbox(
                            "Select a feature for Dependence Plot:",
                            st.session_state.feature_names,
                            key="shap_dependence_feature_select"
                        )
                        with st.spinner(f"Generating SHAP Dependence Plot for {shap_feature_to_plot}..."):
                            # Get the feature's integer index
                            feature_index_for_plot = X_test_display_for_plot.columns.get_loc(shap_feature_to_plot)

                            # Get the SHAP values for the selected class/regression (as 2D array: samples x features)
                            current_shap_values_2d = None
                            selected_class_label_for_title = "Overall" # Default for regression/binary

                            if isinstance(shap_values_for_plotting, list): # Multi-class with list of 2D arrays
                                class_index_for_plot = st.radio("Select class to plot dependence for:",
                                                                options=list(range(len(np.unique(y_train_processed)))),
                                                                format_func=lambda x: (temp_target_encoder_for_model.inverse_transform([x])[0] if temp_target_encoder_for_model else (st.session_state.label_encoder.inverse_transform([x])[0] if st.session_state.label_encoder else str(x))),
                                                                key="shap_dependence_class_select")
                                current_shap_values_2d = np.asarray(shap_values_for_plotting[class_index_for_plot]).astype(np.float64)
                                if temp_target_encoder_for_model and class_index_for_plot < len(temp_target_encoder_for_model.classes_):
                                    selected_class_label_for_title = temp_target_encoder_for_model.inverse_transform([class_index_for_plot])[0]
                                elif st.session_state.label_encoder and class_index_for_plot < len(st.session_state.label_encoder.classes_):
                                    selected_class_label_for_title = st.session_state.label_encoder.inverse_transform([class_index_for_plot])[0]

                            elif shap_values_for_plotting.ndim == 3: # Multi-class with 3D array (samples, features, classes)
                                class_index_for_plot = st.radio("Select class to plot dependence for:",
                                                                options=list(range(shap_values_for_plotting.shape[2])),
                                                                format_func=lambda x: (temp_target_encoder_for_model.inverse_transform([x])[0] if temp_target_encoder_for_model else (st.session_state.label_encoder.inverse_transform([x])[0] if st.session_state.label_encoder else str(x))),
                                                                key="shap_dependence_class_select")
                                current_shap_values_2d = np.asarray(shap_values_for_plotting[:, :, class_index_for_plot]).astype(np.float64)
                                if temp_target_encoder_for_model and class_index_for_plot < len(temp_target_encoder_for_model.classes_):
                                    selected_class_label_for_title = temp_target_encoder_for_model.inverse_transform([class_index_for_plot])[0]
                                elif st.session_state.label_encoder and class_index_for_plot < len(st.session_state.label_encoder.classes_):
                                    selected_class_label_for_title = st.session_state.label_encoder.inverse_transform([class_index_for_plot])[0]

                            else: # Binary Classification or Regression (already a 2D array: samples x features)
                                current_shap_values_2d = np.asarray(shap_values_for_plotting).astype(np.float64)

                            # Now, current_shap_values_2d should ALWAYS be 2D (samples, features)
                            # Extract the 1D array for the chosen feature's SHAP values
                            s_data = current_shap_values_2d[:, feature_index_for_plot]
                            xv_data = X_test_display_for_plot.iloc[:, feature_index_for_plot].to_numpy() # Convert Series to numpy array

                            # --- NEW: Handle NaNs/Infs in the data before plotting ---
                            # Replace NaNs/Infs in xv_data and s_data with a finite value (e.g., 0)
                            xv_data = np.nan_to_num(xv_data, nan=0.0, posinf=0.0, neginf=0.0)
                            s_data = np.nan_to_num(s_data, nan=0.0, posinf=0.0, neginf=0.0)

                            st.write(f"Debug SHAP plot input: Feature '{shap_feature_to_plot}' (index {feature_index_for_plot})")
                            st.write(f"Debug SHAP plot input: Length of feature values (xv): {len(xv_data)}")
                            st.write(f"Debug SHAP plot input: Length of SHAP values for this feature (s): {len(s_data)}")
                            # --- END NEW DEBUGGING PRINTS ---

                            # Add constant feature check, very important for dependence plots
                            feature_values_for_plot = X_test_display_for_plot[shap_feature_to_plot] # Access by name from DataFrame

                            if np.all(feature_values_for_plot == feature_values_for_plot.iloc[0]):
                                st.warning(f"Selected feature '{shap_feature_to_plot}' has a constant value across all samples. Dependence plot cannot be generated for constant features as there's no variation to show. Try another feature.")
                                st.stop()

                            # Also check if the SHAP values for this feature are constant
                            shap_values_for_this_feature_check = current_shap_values_2d[:, feature_index_for_plot]

                            if np.all(shap_values_for_this_feature_check == shap_values_for_this_feature_check[0]):
                                st.warning(f"SHAP values for feature '{shap_feature_to_plot}' are constant across all samples. Dependence plot cannot be generated. This might indicate the feature has no impact on predictions, or issues in SHAP calculation.")
                                st.stop()

                            if len(xv_data) != len(s_data):
                                st.error(f"❌ Critical Error for SHAP Plotting: Length mismatch detected just before `scatter` call. "
                                         f"Feature values length: {len(xv_data)}, SHAP values length: {len(s_data)}. "
                                         "This indicates a deep inconsistency. Please review previous data transformations, especially sampling or row removals.")
                                st.stop()

                            fig, ax = plt.subplots(figsize=(10,6))
                            # Pass the integer feature index, the 2D SHAP values (for regression/binary/selected-class-for-multi), and the DataFrame of features
                            # Now, s_data is the correct 1D array of SHAP values for the selected feature
                            # The shap.dependence_plot function expects a 2D shap_values array, so we pass current_shap_values_2d
                            shap.dependence_plot(feature_index_for_plot, current_shap_values_2d, X_test_display_for_plot, interaction_index=None, show=False, ax=ax)

                            ax.set_title(f"SHAP Dependence for {shap_feature_to_plot} (Class: {selected_class_label_for_title})")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                    except Exception as shap_e:
                        st.error(f"❌ Error generating SHAP plots: {str(shap_e)}. "
                                 "SHAP may have issues with sparse data, or specific model types. "
                                 "Ensure the selected feature is not constant and that SHAP values for it are not constant.")
                        st.exception(shap_e)
                        st.stop()

            with tab4: # Model Comparison Tab (new tab)
                st.subheader("🏆 Best ML Model Comparison")

                if st.session_state.task_type == "Classification":
                    available_models = ["Random Forest", "XGBoost", "LightGBM", "Logistic Regression", "SVC", "K-Nearest Neighbors", "Gaussian Naive Bayes"]
                else: # Regression
                    available_models = ["Linear Regression", "Ridge", "Lasso", "ElasticNet", "SVR", "K-Nearest Neighbors", "Random Forest", "XGBoost", "LightGBM"]

                models_to_compare = st.multiselect(
                    "Select Models to Compare",
                    options=available_models,
                    default=available_models # Default to all available for comparison
                )

                if st.button("Compare Selected Models"):
                    if not models_to_compare:
                        st.warning("Please select at least one model to compare.")
                    else:
                        results = []
                        with st.spinner("🚀 Comparing models... This might take a while."):
                            for model_name in models_to_compare:
                                current_model = None
                                if st.session_state.task_type == "Classification":
                                    if model_name == "Random Forest":
                                        current_model = RandomForestClassifier(random_state=42)
                                    elif model_name == "XGBoost":
                                        st.write(f"Debug: X_train columns for {model_name} comparison: {X_train.columns.tolist()}") # DEBUG
                                        current_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                                    elif model_name == "LightGBM":
                                        current_model = LGBMClassifier(random_state=42)
                                    elif model_name == "Logistic Regression":
                                        current_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)
                                    elif model_name == "SVC":
                                        current_model = SVC(random_state=42, probability=True)
                                    elif model_name == "K-Nearest Neighbors":
                                        current_model = KNeighborsClassifier()
                                    elif model_name == "Gaussian Naive Bayes":
                                        current_model = GaussianNB()
                                else: # Regression
                                    if model_name == "Linear Regression":
                                        current_model = LinearRegression()
                                    elif model_name == "Ridge":
                                        current_model = Ridge(random_state=42)
                                    elif model_name == "Lasso":
                                        current_model = Lasso(alpha=alpha_param, random_state=42)
                                    elif model_name == "ElasticNet": # Added ElasticNet to comparison
                                        current_model = ElasticNet(random_state=42)
                                    elif model_name == "SVR":
                                        current_model = SVR()
                                    elif model_name == "K-Nearest Neighbors":
                                        current_model = KNeighborsRegressor()
                                    elif model_name == "Random Forest":
                                        current_model = RandomForestRegressor(random_state=42)
                                    elif model_name == "XGBoost":
                                        st.write(f"Debug: X_train columns for {model_name} comparison: {X_train.columns.tolist()}") # DEBUG
                                        current_model = XGBRegressor(objective='reg:squarederror', random_state=42)
                                    elif model_name == "LightGBM":
                                        current_model = LGBMRegressor(objective='regression', random_state=42)

                                try:
                                    if current_model is not None:
                                        # Use processed y_train and y_test for comparison models if needed (classification)
                                        comp_y_train = y_train_processed
                                        comp_y_test = y_test_processed

                                        current_model.fit(X_train, comp_y_train)
                                        y_pred_comp = current_model.predict(X_test)

                                        model_metrics = {"Model": model_name}
                                        if st.session_state.task_type == "Classification":
                                            acc = accuracy_score(comp_y_test, y_pred_comp)
                                            f1 = f1_score(comp_y_test, y_pred_comp, average='weighted')
                                            model_metrics["Accuracy"] = f"{acc:.4f}"
                                            model_metrics["F1 Score"] = f"{f1:.4f}"
                                            if hasattr(current_model, "predict_proba") and np.unique(comp_y_test).size == 2:
                                                y_proba_comp = current_model.predict_proba(X_test)[:, 1]
                                                auc_roc = roc_auc_score(comp_y_test, y_proba_comp)
                                                model_metrics["AUC-ROC"] = f"{auc_roc:.4f}"
                                            else:
                                                model_metrics["AUC-ROC"] = "N/A"
                                        else: # Regression
                                            r2 = r2_score(comp_y_test, y_pred_comp)
                                            mse = mean_squared_error(comp_y_test, y_pred_comp)
                                            mae = mean_absolute_error(comp_y_test, y_pred_comp)
                                            model_metrics["R² Score"] = f"{r2:.4f}"
                                            model_metrics["MSE"] = f"{mse:.4f}"
                                            model_metrics["MAE"] = f"{mae:.4f}"
                                        results.append(model_metrics)
                                    else:
                                         st.warning(f"Model '{model_name}' could not be initialized or configured for {st.session_state.task_type}. Skipping.")
                                except Exception as comp_e:
                                    st.error(f"Error training/evaluating {model_name}: {comp_e}")

                        if results:
                            st.session_state.comparison_results = pd.DataFrame(results)
                            st.subheader("Model Comparison Results:")
                            st.dataframe(st.session_state.comparison_results)

                            # Highlight the best model
                            if st.session_state.task_type == "Classification":
                                primary_metric = "Accuracy"
                                st.session_state.comparison_results[primary_metric] = st.session_state.comparison_results[primary_metric].astype(float)
                                best_model_row = st.session_state.comparison_results.loc[st.session_state.comparison_results[primary_metric].idxmax()]
                            else: # Regression
                                primary_metric = "R² Score"
                                st.session_state.comparison_results[primary_metric] = st.session_state.comparison_results[primary_metric].astype(float)
                                best_model_row = st.session_state.comparison_results.loc[st.session_state.comparison_results[primary_metric].idxmax()]

                            st.success(f"🏆 The best performing model based on {primary_metric} is: **{best_model_row['Model']}** with {best_model_row[primary_metric]:.4f}.")
                        else:
                            st.info("No models were successfully compared.")


            with tab5: # Model Export Tab
                st.write("### Save Model Pipeline")
                st.info("""
                    This will save the trained model along with the preprocessing pipeline
                    so that the entire workflow can be used for future predictions.
                """)
                model_format = st.radio("Select export format", ["Pickle", "Joblib", "ONNX (Info)"])

                if st.button("💾 Export Model Pipeline"):
                    import joblib
                    from io import BytesIO

                    # Create the full pipeline: preprocessing + model
                    if st.session_state.full_ml_pipeline == None:
                        st.warning("Full pipeline not found in session state. Please train a model first (run ML Mode).")
                        st.stop()

                    full_pipeline_to_export = st.session_state.full_ml_pipeline


                    buffer = BytesIO()
                    if model_format == "Pickle":
                        joblib.dump(full_pipeline_to_export, buffer)
                        st.download_button(
                            label="Download Full ML Pipeline (Pickle)",
                            data=buffer.getvalue(),
                            file_name="full_ml_pipeline.pkl",
                            mime="application/octet-stream"
                        )
                    elif model_format == "Joblib":
                        joblib.dump(full_pipeline_to_export, buffer)
                        st.download_button(
                            label="Download Full ML Pipeline (Joblib)",
                            data=buffer.getvalue(),
                            file_name="full_ml_pipeline.joblib",
                            mime="application/octet-stream"
                        )
                    else: # ONNX Info
                        st.info("""
                        ONNX export provides cross-platform compatibility.
                        It requires specific libraries (`skl2onnx`, `onnxruntime`) and conversion steps.
                        **Note:** Direct ONNX export for complex pipelines with multiple scikit-learn transformers
                        (like `OneHotEncoder`, `KNNImputer`, `PCA`, `SelectKBest`, `PolynomialFeatures`) can be challenging and
                        often requires ensuring all transformers are supported by `skl2onnx`.
                        See the official documentation for detailed instructions:
                        [ONNX conversion guide](https://onnx.ai/sklearn-onnx/)
                        """)

                # --- Test Prediction Section ---
                st.write("### Test Prediction on New Data")
                st.info("Input values for prediction will be processed through the same pipeline used for training.")

                # Create input fields dynamically based on the ORIGINAL feature names (before any custom FE)
                sample_input_values = {}
                original_input_cols_for_widgets = st.session_state.original_X_cols_before_custom_fe

                for col_name in original_input_cols_for_widgets:
                    # Get the original column's data type from the full DataFrame
                    original_col_dtype = df[col_name].dtype if col_name in df.columns else None

                    if pd.api.types.is_numeric_dtype(original_col_dtype):
                        # Ensure numerical default values are handled correctly, especially for empty datasets
                        default_value = df[col_name].median() if not df[col_name].empty and pd.notna(df[col_name].median()) else 0.0
                        sample_input_values[col_name] = st.number_input(
                            col_name,
                            value=float(default_value), # Ensure float type
                            key=f"predict_input_{col_name}"
                        )
                    elif pd.api.types.is_object_dtype(original_col_dtype) or pd.api.types.is_categorical_dtype(original_col_dtype):
                        options = df[col_name].unique().tolist()
                        options = [str(x) for x in options if pd.notna(x)] # Filter out NaNs and ensure string type
                        if not options: # Fallback if no options are found
                            options = ["No data / Default"]
                        default_index = 0
                        if not df[col_name].mode().empty and str(df[col_name].mode()[0]) in options:
                            mode_val = str(df[col_name].mode()[0])
                            if mode_val in options:
                                default_index = options.index(mode_val)

                        sample_input_values[col_name] = st.selectbox(
                            col_name,
                            options=options,
                            index=default_index,
                            key=f"predict_input_{col_name}"
                        )
                    elif pd.api.types.is_datetime64_any_dtype(original_col_dtype):
                        # Provide a sample datetime string as default
                        default_datetime_str = str(df[col_name].iloc[0]) if not df[col_name].empty else "2023-01-01 12:00:00"
                        sample_input_values[col_name] = st.text_input(
                            col_name,
                            value=default_datetime_str,
                            key=f"predict_input_{col_name}"
                        )
                    else:
                        sample_input_values[col_name] = st.text_input(
                            col_name,
                            value="", # Default empty for unknown types
                            key=f"predict_input_{col_name}_fallback"
                        )


                if st.button("Predict"):
                    if st.session_state.full_ml_pipeline is None:
                        st.warning("Please train a model and build the pipeline first before attempting prediction (Run ML Mode).")
                        st.stop()

                    # Create a DataFrame from the input values. This is the raw input.
                    input_df_raw_for_fe = pd.DataFrame([sample_input_values])

                    # Convert input_df_raw_for_fe columns to their original dtypes for consistent processing
                    for col in input_df_raw_for_fe.columns:
                        if col in df.columns:
                            original_dtype = df[col].dtype
                            if pd.api.types.is_numeric_dtype(original_dtype):
                                input_df_raw_for_fe[col] = pd.to_numeric(input_df_raw_for_fe[col], errors='coerce')
                            elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                                input_df_raw_for_fe[col] = pd.to_datetime(input_df_raw_for_fe[col], errors='coerce')

                    # Re-apply custom feature engineering steps exactly as done during training
                    # 1. Datetime feature extraction
                    if st.session_state.datetime_features_enabled:
                        input_df_after_dt_fe = apply_datetime_features(input_df_raw_for_fe, st.session_state.datetime_cols_extracted)
                    else:
                        input_df_after_dt_fe = input_df_raw_for_fe.copy()

                    # 2. Synthetic feature creation
                    # Only apply if the synthetic feature was enabled during training AND it exists in the training features
                    if st.session_state.add_synthetic_feature_enabled and 'synthetic_highly_correlated_feature' in st.session_state.processed_training_features_at_pipeline_entry:
                        # Pass the original_df (cleaned of target/ID) to helper for proxy logic
                        input_df_for_pipeline_entry = add_synthetic_feature_to_df(
                            input_df_after_dt_fe,
                            original_df_cleaned=df.drop(columns=st.session_state.id_cols_used_in_training + [st.session_state.target_col_used_in_training], errors='ignore'),
                            feature_names_from_training=st.session_state.processed_training_features_at_pipeline_entry # Use this to check if synthetic was added
                        )
                    else:
                        input_df_for_pipeline_entry = input_df_after_dt_fe.copy()


                    # Crucial step: Reindex the input DataFrame to precisely match the columns
                    # that entered the preprocessing pipeline (ColumnTransformer) during training.
                    # This ensures consistent feature set for the transformer.
                    final_input_df_for_pipeline_transform = input_df_for_pipeline_entry.reindex(
                        columns=st.session_state.processed_training_features_at_pipeline_entry,
                        fill_value=0 # Fill numerical columns not present (e.g., from one-hot-encoding only having trained categories) with 0)
                    )

                    # Handle potential categorical columns that might have been reindexed with NaN
                    # by converting to appropriate type before passing to pipeline.
                    # This ensures OneHotEncoder receives string inputs, not numbers from fill_value=0.
                    categorical_cols_at_pipeline_entry = [
                        col for col in st.session_state.processed_training_features_at_pipeline_entry
                        if col in df.columns and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]))
                    ]

                    for col in categorical_cols_at_pipeline_entry:
                        if col in final_input_df_for_pipeline_transform.columns:
                            # Ensure these columns are of object type for OneHotEncoder
                            final_input_df_for_pipeline_transform[col] = final_input_df_for_pipeline_transform[col].astype(str).fillna('') # Fill with empty string for categorical NaNs

                    st.write(f"Debug Pred: Input DF for pipeline after custom FE and reindexing: {final_input_df_for_pipeline_transform.columns.tolist()}")
                    st.write(f"Debug Pred: Expected DF columns (from training pipeline entry): {st.session_state.processed_training_features_at_pipeline_entry}")

                    # Final check for column count before predicting
                    if len(final_input_df_for_pipeline_transform.columns) != len(st.session_state.processed_training_features_at_pipeline_entry):
                        st.error("Column count mismatch in final prediction input. This indicates a severe issue in feature alignment.")
                        st.stop()

                    # Make prediction using the full pipeline (preprocessor + model)
                    prediction = st.session_state.full_ml_pipeline.predict(final_input_df_for_pipeline_transform)

                    # Inverse transform the prediction for classification if target was encoded
                    if st.session_state.task_type == "Classification" and st.session_state.label_encoder:
                        if isinstance(prediction, np.ndarray) and prediction.ndim == 0:
                            prediction = prediction.item()
                        if not isinstance(prediction, np.ndarray): # Ensure it's an array before inverse_transform
                            prediction = np.array([prediction])

                        # Use the correct encoder for inverse transformation: temp_target_encoder_for_model if it was used, else st.session_state.label_encoder
                        encoder_for_inverse_transform = None
                        if 'target_encoder' in st.session_state and st.session_state.target_encoder: # This is the one fitted just before model for contiguity
                            encoder_for_inverse_transform = st.session_state.target_encoder
                        elif st.session_state.label_encoder: # Fallback to the initial one if no re-encoding happened
                            encoder_for_inverse_transform = st.session_state.label_encoder

                        if encoder_for_inverse_transform:
                            prediction = encoder_for_inverse_transform.inverse_transform(prediction.astype(int))
                        else:
                            st.warning("No label encoder found for inverse transformation. Displaying raw numerical prediction.")


                    st.success(f"Prediction: {prediction[0]}")

        except Exception as e:
            st.error(f"❌ An unexpected error occurred in ML Mode: {str(e)}")
            st.exception(e) # This will display the full traceback in the Streamlit app.
        # --- End of enhanced error handling for ML Mode ---
