import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, mean_squared_error, r2_score, 
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import io
import joblib

# Set page config
st.set_page_config(page_title="Data Explorer & ML Pipeline", layout="wide", page_icon="🔍")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 90%;
    }
    .stButton>button {
        width: 100%;
    }
    .stSelectbox, .stMultiselect, .stFileUploader {
        margin-bottom: 10px;
    }
    .plot-container {
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================
# Data Exploration Functions (Your original code)
# ==============================================
def load_data(uploaded_file):
    """Load data from uploaded file (CSV or Excel)"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def show_data_issues(df):
    """Display data quality issues to the user"""
    st.subheader("Data Quality Report")
    
    # Missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("🚨 Missing Values Detected")
        st.write(missing_values[missing_values > 0])
    else:
        st.success("✅ No missing values found")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"🚨 Found {duplicates} duplicate rows")
    else:
        st.success("✅ No duplicate rows found")
    
    # Data types
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    # Numeric stats
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        st.subheader("Numeric Columns Statistics")
        st.write(df[numeric_cols].describe())
    
    # Categorical stats
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        st.subheader("Categorical Columns Summary")
        for col in cat_cols:
            st.write(f"**{col}**: {df[col].nunique()} unique values")
            st.write(df[col].value_counts().head())

def clean_data(df):
    """Clean the dataset based on user selections"""
    st.subheader("Data Cleaning Options")
    
    with st.expander("Missing Value Handling", expanded=True):
        # Display missing values info
        missing_values = df.isnull().sum()
        missing_cols = missing_values[missing_values > 0].index.tolist()
        
        if missing_cols:
            st.write("Columns with missing values:")
            for col in missing_cols:
                null_count = missing_values[col]
                null_pct = (null_count / len(df)) * 100
                st.write(f"- **{col}**: {null_count} nulls ({null_pct:.2f}%)")
                
                col1, col2 = st.columns(2)
                with col1:
                    action = st.selectbox(
                        f"Action for '{col}'",
                        ["Keep as is", "Drop column", "Drop rows", "Fill with mean/median/mode", "Fill with value"],
                        key=f"missing_{col}"
                    )
                with col2:
                    if action == "Fill with mean/median/mode":
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fill_method = st.selectbox(
                                "Fill method",
                                ["mean", "median"],
                                key=f"fill_method_{col}"
                            )
                        else:
                            fill_method = "mode"
                            st.write("Will fill with mode (most frequent value)")
                    elif action == "Fill with value":
                        fill_value = st.text_input("Fill value", key=f"fill_value_{col}")
                    else:
                        st.write("")
        else:
            st.info("No missing values found in the dataset")
    
    with st.expander("Duplicate Handling"):
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"Found {duplicates} duplicate rows in the dataset.")
            if st.checkbox("Remove duplicate rows", key="remove_duplicates"):
                df = df.drop_duplicates()
                st.success(f"Removed {duplicates} duplicate rows.")
        else:
            st.info("No duplicate rows found")
    
    if st.button("Apply Cleaning", key="apply_cleaning"):
        if missing_cols:
            for col in missing_cols:
                action = st.session_state.get(f"missing_{col}")
                if action == "Drop column":
                    df = df.drop(columns=[col])
                    st.success(f"Dropped column: {col}")
                elif action == "Drop rows":
                    df = df.dropna(subset=[col])
                    st.success(f"Dropped rows with missing values in: {col}")
                elif action == "Fill with mean/median/mode":
                    fill_method = st.session_state.get(f"fill_method_{col}")
                    if fill_method == "mean":
                        fill_value = df[col].mean()
                    elif fill_method == "median":
                        fill_value = df[col].median()
                    else:  # mode
                        fill_value = df[col].mode()[0]
                    df[col] = df[col].fillna(fill_value)
                    st.success(f"Filled '{col}' with {fill_method}: {fill_value}")
                elif action == "Fill with value":
                    fill_value = st.session_state.get(f"fill_value_{col}")
                    try:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fill_value = float(fill_value)
                        df[col] = df[col].fillna(fill_value)
                        st.success(f"Filled '{col}' with value: {fill_value}")
                    except ValueError:
                        st.error(f"Invalid fill value for column {col}")
        
        st.session_state.df = df.copy()
        st.session_state.data_cleaned = True
        st.experimental_rerun()
    
    return df

def scale_data(df, numeric_cols):
    """Apply scaling to numeric columns"""
    st.subheader("Feature Scaling Options")
    
    if not numeric_cols:
        st.info("No numeric columns found for scaling.")
        return df
    
    scaler_type = st.selectbox(
        "Select scaler type",
        ["Standard Scaler (mean=0, std=1)", "MinMax Scaler (0-1)", "Robust Scaler (resistant to outliers)"],
        index=0
    )
    
    cols_to_scale = st.multiselect(
        "Select columns to scale",
        numeric_cols,
        default=numeric_cols
    )
    
    if st.button("Apply Scaling"):
        if not cols_to_scale:
            st.warning("Please select at least one column to scale.")
            return df
        
        if scaler_type == "Standard Scaler (mean=0, std=1)":
            scaler = StandardScaler()
        elif scaler_type == "MinMax Scaler (0-1)":
            scaler = MinMaxScaler()
        else:  # Robust Scaler
            scaler = RobustScaler()
        
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
        st.success(f"Applied {scaler_type} to selected columns.")
        
        st.session_state.df = df_scaled.copy()
        st.session_state.data_scaled = True
        st.experimental_rerun()
    
    return df

def plot_data(df, features, target):
    """Create various plots to explore feature-target relationships"""
    st.subheader("Data Visualization")
    
    plot_type = st.selectbox(
        "Select plot type",
        ["Scatter Plot", "Box Plot", "Violin Plot", "Bar Plot", 
         "Line Plot", "Histogram", "Correlation Heatmap", "Pair Plot"]
    )
    
    if st.button("Generate Plot"):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if plot_type == "Scatter Plot":
                selected_feature = st.selectbox("Select feature to plot", features)
                sns.scatterplot(data=df, x=selected_feature, y=target, ax=ax)
                ax.set_title(f"Scatter Plot: {selected_feature} vs {target}")
            
            elif plot_type == "Box Plot":
                selected_feature = st.selectbox("Select feature to plot", features)
                sns.boxplot(data=df, x=target, y=selected_feature, ax=ax)
                ax.set_title(f"Box Plot: {selected_feature} by {target}")
            
            elif plot_type == "Violin Plot":
                selected_feature = st.selectbox("Select feature to plot", features)
                sns.violinplot(data=df, x=target, y=selected_feature, ax=ax)
                ax.set_title(f"Violin Plot: {selected_feature} by {target}")
            
            elif plot_type == "Bar Plot":
                selected_feature = st.selectbox("Select feature to plot", features)
                sns.barplot(data=df, x=selected_feature, y=target, ax=ax, estimator=np.mean)
                ax.set_title(f"Bar Plot: {selected_feature} vs {target}")
            
            elif plot_type == "Line Plot":
                selected_feature = st.selectbox("Select feature to plot", features)
                sns.lineplot(data=df, x=selected_feature, y=target, ax=ax)
                ax.set_title(f"Line Plot: {selected_feature} vs {target}")
            
            elif plot_type == "Histogram":
                selected_feature = st.selectbox("Select feature to plot", features)
                sns.histplot(data=df, x=selected_feature, kde=True, ax=ax)
                ax.set_title(f"Distribution of {selected_feature}")
            
            elif plot_type == "Correlation Heatmap":
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for correlation heatmap")
                    return
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title("Correlation Heatmap")
            
            elif plot_type == "Pair Plot":
                cols_to_plot = st.multiselect(
                    "Select columns for pair plot",
                    df.columns.tolist(),
                    default=features[:3] + ([target] if target else [])
                )
                if len(cols_to_plot) < 2:
                    st.warning("Please select at least 2 columns for pair plot")
                    return
                pair_plot = sns.pairplot(df[cols_to_plot], hue=target if target else None)
                st.pyplot(pair_plot)
                return
            
            plt.tight_layout()
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error creating plot: {e}")

# ==============================================
# Machine Learning Functions (Your provided code)
# ==============================================
class MachineLearningApp:
    def __init__(self):
        self.initialize_session_state()

    def initialize_session_state(self):
        initial_states = {
            'data': None,
            'X': None,
            'y': None,
            'model': None,
            'scaler': None,
            'label_encoder': None,
            'problem_type': None,
            'test_size': 0.2,
            'selected_features': [],
            'target_column': None,
            'selected_model': None,
            'model_results': None
        }
        
        for key, value in initial_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def sidebar_data_upload(self):
        with st.sidebar:
            st.header("📊 Data Upload")
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file", 
                type=['csv', 'xlsx', 'xls']
            )
            return uploaded_file

    def sidebar_feature_selection(self, df):
        with st.sidebar:
            st.header("🔍 Feature Selection")
            
            if df is None:
                st.warning("Please upload a dataset first.")
                return None, None, None
            
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            selected_features = st.multiselect(
                "Select Features", 
                options=list(df.columns),
                default=numeric_cols
            )
            
            target_column = st.selectbox(
                "Select Target Column", 
                options=list(df.columns)
            )
            
            test_size = st.slider(
                "Test Set Percentage", 
                min_value=0.1, 
                max_value=0.5, 
                value=0.2, 
                step=0.05
            )
            
            return selected_features, target_column, test_size

    def sidebar_model_selection(self, problem_type):
        with st.sidebar:
            st.header("🤖 Model Selection")
            
            if problem_type == 'classification':
                models = {
                    'Logistic Regression': LogisticRegression(),
                    'Decision Tree': DecisionTreeClassifier(),
                    'Random Forest': RandomForestClassifier(),
                    'SVM': SVC(),
                    'Naive Bayes (Gaussian)': GaussianNB(),
                    'K-Nearest Neighbors': KNeighborsClassifier(),
                    'Neural Network': MLPClassifier(max_iter=1000)
                }
            else:
                models = {
                    'Linear Regression': LinearRegression(),
                    'Decision Tree': DecisionTreeRegressor(),
                    'Random Forest': RandomForestRegressor(),
                    'SVR': SVR(),
                    'K-Nearest Neighbors': KNeighborsRegressor(),
                    'Neural Network': MLPRegressor(max_iter=1000)
                }
            
            selected_model = st.selectbox(
                "Choose a Model", 
                options=list(models.keys())
            )
            
            return models, selected_model

    def sidebar_prediction_input(self, selected_features):
        with st.sidebar:
            st.header("🔮 Prediction Input")
            
            if st.session_state.model is None:
                st.warning("Please train a model first.")
                return None
            
            prediction_inputs = {}
            for feature in selected_features:
                prediction_inputs[feature] = st.number_input(
                    f"Enter {feature}", 
                    value=0.0,
                    step=0.1
                )
            
            if st.button("Predict"):
                return prediction_inputs
            
            return None

    def load_and_display_data(self, uploaded_file):
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 Dataset Preview")
                    st.dataframe(df.head())
                
                with col2:
                    st.subheader("📊 Dataset Information")
                    st.write(f"Total Rows: {df.shape[0]}")
                    st.write(f"Total Columns: {df.shape[1]}")
                    
                    col_types = df.dtypes.value_counts()
                    st.write("Column Types:")
                    for dtype, count in col_types.items():
                        st.text(f"{dtype}: {count} columns")
                
                return df
            
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return None

    def train_and_evaluate_model(self, X, y, test_size, models, selected_model_name):
        results_container = st.container()
        
        with results_container:
            X_scaled = StandardScaler().fit_transform(X)
            
            problem_type = 'classification' if y.dtype == 'object' else 'regression'
            
            label_encoder = None
            if problem_type == 'classification':
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
            else:
                y_encoded = y
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=42
            )
            
            model = models[selected_model_name]
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            st.header("🔬 Model Training Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Model Performance")
                if problem_type == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    
                    st.subheader("Classification Report")
                    report = classification_report(
                        y_test, y_pred, 
                        target_names=label_encoder.classes_ if label_encoder else None,
                        output_dict=True
                    )
                    
                    for key, value in report.items():
                        if isinstance(value, dict):
                            st.text(f"{key}:")
                            for metric, score in value.items():
                                st.text(f"  {metric}: {score:.2f}")
                
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    st.metric("Mean Squared Error", f"{mse:.4f}")
                    st.metric("Mean Absolute Error", f"{mae:.4f}")
                    st.metric("R² Score", f"{r2:.4f}")
            
            with col2:
                st.subheader("📈 Model Details")
                st.write(f"Selected Model: {selected_model_name}")
                st.write(f"Problem Type: {problem_type}")
                st.write(f"Test Set Size: {test_size:.0%}")
                st.write(f"Features Used: {', '.join(X.columns)}")
                st.write(f"Target Column: {y.name}")
            
            st.session_state.model = model
            st.session_state.scaler = StandardScaler().fit(X)
            st.session_state.label_encoder = label_encoder
            st.session_state.problem_type = problem_type
            st.session_state.X = X

    def make_prediction(self, prediction_inputs):
        if st.session_state.model is None:
            st.error("Please train a model first.")
            return
        
        input_df = pd.DataFrame([prediction_inputs])
        
        input_scaled = st.session_state.scaler.transform(input_df)
        
        prediction = st.session_state.model.predict(input_scaled)
        
        if st.session_state.label_encoder:
            prediction = st.session_state.label_encoder.inverse_transform(prediction)
        
        st.header("🎯 Prediction Result")
        st.subheader("Input Data")
        st.dataframe(input_df)
        
        st.subheader("Predicted Value")
        st.write(prediction[0])

    def run_ml_mode(self):
        """Run the complete ML workflow from your provided code"""
        st.title("🚀 Machine Learning Pipeline")
        
        uploaded_file = self.sidebar_data_upload()
        df = self.load_and_display_data(uploaded_file)
        
        if df is not None:
            selected_features, target_column, test_size = self.sidebar_feature_selection(df)
            
            if selected_features and target_column:
                X = df[selected_features]
                y = df[target_column]
                
                problem_type = 'classification' if y.dtype == 'object' else 'regression'
                
                models, selected_model = self.sidebar_model_selection(problem_type)
                
                with st.sidebar:
                    if st.button("Train Model", type="primary"):
                        for key in ['model', 'scaler', 'label_encoder', 'problem_type']:
                            st.session_state[key] = None
                        
                        self.train_and_evaluate_model(
                            X, y, test_size, models, selected_model
                        )
                
                prediction_inputs = self.sidebar_prediction_input(selected_features)
                
                if prediction_inputs:
                    self.make_prediction(prediction_inputs)

# ==============================================
# Main App Function
# ==============================================
def main():
    st.title("🔍 Data Explorer & ML Pipeline")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_cleaned' not in st.session_state:
        st.session_state.data_cleaned = False
    if 'data_scaled' not in st.session_state:
        st.session_state.data_scaled = False
    
    # Navigation
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Data Exploration", "Machine Learning"]
    )
    
    # File upload section (shared by both modes)
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=['csv', 'xls', 'xlsx']
    )
    
    if uploaded_file is not None:
        if st.session_state.df is None:
            st.session_state.df = load_data(uploaded_file)
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            if app_mode == "Data Exploration":
                # Display basic info
                st.sidebar.success("Data loaded successfully!")
                st.sidebar.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Show raw data preview
                if st.sidebar.checkbox("Show raw data"):
                    st.subheader("Raw Data Preview")
                    st.dataframe(df.head())
                
                # Show data quality report
                if st.checkbox("Show Data Quality Report", key="show_quality_report"):
                    show_data_issues(df)
                
                # Select target column
                st.sidebar.header("Select Target Column")
                all_columns = df.columns.tolist()
                target = st.sidebar.selectbox(
                    "Select target variable (optional)",
                    [None] + all_columns
                )
                
                # Get features (all columns except target)
                if target:
                    features = [col for col in all_columns if col != target]
                else:
                    features = all_columns
                
                # Data cleaning section
                st.header("Data Cleaning & Preprocessing")
                if st.checkbox("Show data cleaning options", key="show_cleaning_options"):
                    df = clean_data(df)
                    if df is not None and not df.equals(st.session_state.df):
                        st.session_state.df = df.copy()
                        st.session_state.data_cleaned = True
                
                # Feature scaling section
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if numeric_cols and st.checkbox("Show feature scaling options", key="show_scaling_options"):
                    df = scale_data(df, numeric_cols)
                    if df is not None and not df.equals(st.session_state.df):
                        st.session_state.df = df.copy()
                        st.session_state.data_scaled = True
                
                # Data visualization section
                st.header("Data Exploration")
                plot_data(df, features, target)
                
                # Show processed data with download option
                if st.checkbox("Show processed data", key="show_processed_data"):
                    st.subheader("Processed Data")
                    
                    # Show processing status
                    cleaning_status = "✅ Cleaned" if st.session_state.data_cleaned else "❌ Not cleaned"
                    scaling_status = "✅ Scaled" if st.session_state.data_scaled else "❌ Not scaled"
                    
                    st.write(f"**Processing Status:** {cleaning_status} | {scaling_status}")
                    st.dataframe(df.head())
            
            elif app_mode == "Machine Learning":
                # Switch completely to your ML code
                ml_app = MachineLearningApp()
                ml_app.run_ml_mode()

if __name__ == "__main__":
    main()
