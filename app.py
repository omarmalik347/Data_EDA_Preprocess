import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, mean_squared_error, r2_score, 
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
import pickle
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
                        # Try to convert to appropriate type
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
        
        # Show before/after comparison
        st.subheader("Scaling Results Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before Scaling**")
            st.write(df[cols_to_scale].describe())
        with col2:
            st.write("**After Scaling**")
            st.write(df_scaled[cols_to_scale].describe())
        
        st.session_state.df = df_scaled.copy()
        st.session_state.data_scaled = True
        st.experimental_rerun()
    
    return df

def plot_data(df, features, target):
    """Create various plots to explore feature-target relationships"""
    st.subheader("Data Visualization")
    
    plot_type = st.selectbox(
        "Select plot type",
        [
            "Scatter Plot",
            "Box Plot",
            "Violin Plot",
            "Bar Plot",
            "Line Plot",
            "Histogram",
            "Correlation Heatmap",
            "Pair Plot",
            "Distribution Plot"
        ]
    )
    
    # For plots that need feature selection
    if plot_type not in ["Correlation Heatmap", "Pair Plot"]:
        if plot_type in ["Histogram", "Distribution Plot"]:
            selected_features = st.multiselect(
                "Select features to plot", 
                features,
                default=[features[0]] if features else []
            )
        else:
            selected_features = st.selectbox("Select feature to plot", features)
    
    # For plots that need target selection (if target is specified)
    if target and plot_type not in ["Correlation Heatmap", "Pair Plot", "Histogram", "Distribution Plot"]:
        use_target = st.checkbox("Use target variable in plot", True)
    else:
        use_target = False
    
    if st.button("Generate Plot"):
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if plot_type == "Scatter Plot":
                if use_target:
                    sns.scatterplot(data=df, x=selected_features, y=target, ax=ax)
                    ax.set_title(f"Scatter Plot: {selected_features} vs {target}")
                else:
                    st.warning("Scatter plot requires both x and y variables")
                    return
            
            elif plot_type == "Box Plot":
                if use_target:
                    sns.boxplot(data=df, x=target, y=selected_features, ax=ax)
                    ax.set_title(f"Box Plot: {selected_features} by {target}")
                else:
                    sns.boxplot(data=df, y=selected_features, ax=ax)
                    ax.set_title(f"Box Plot: {selected_features}")
            
            elif plot_type == "Violin Plot":
                if use_target:
                    sns.violinplot(data=df, x=target, y=selected_features, ax=ax)
                    ax.set_title(f"Violin Plot: {selected_features} by {target}")
                else:
                    sns.violinplot(data=df, y=selected_features, ax=ax)
                    ax.set_title(f"Violin Plot: {selected_features}")
            
            elif plot_type == "Bar Plot":
                if use_target:
                    sns.barplot(data=df, x=selected_features, y=target, ax=ax, estimator=np.mean)
                    ax.set_title(f"Bar Plot: {selected_features} vs {target}")
                else:
                    df[selected_features].value_counts().plot(kind='bar', ax=ax)
                    ax.set_title(f"Bar Plot: {selected_features}")
            
            elif plot_type == "Line Plot":
                if use_target:
                    sns.lineplot(data=df, x=selected_features, y=target, ax=ax)
                    ax.set_title(f"Line Plot: {selected_features} vs {target}")
                else:
                    st.warning("Line plot requires a target variable")
                    return
            
            elif plot_type == "Histogram":
                for feature in selected_features:
                    sns.histplot(data=df, x=feature, kde=True, ax=ax, alpha=0.4, label=feature)
                ax.set_title(f"Distribution of {', '.join(selected_features)}")
                if len(selected_features) > 1:
                    ax.legend()
            
            elif plot_type == "Distribution Plot":
                for feature in selected_features:
                    sns.kdeplot(data=df, x=feature, ax=ax, label=feature)
                ax.set_title(f"Distribution of {', '.join(selected_features)}")
                if len(selected_features) > 1:
                    ax.legend()
            
            elif plot_type == "Correlation Heatmap":
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for correlation heatmap")
                    return
                
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title("Correlation Heatmap")
            
            elif plot_type == "Pair Plot":
                cols_to_plot = st.multiselect(
                    "Select columns for pair plot",
                    df.columns.tolist(),
                    default=features[:3] + ([target] if target else [])  # Limit to first 3 features + target
                )
                
                if len(cols_to_plot) < 2:
                    st.warning("Please select at least 2 columns for pair plot")
                    return
                
                if target and target in cols_to_plot:
                    hue_col = target
                else:
                    hue_col = None
                
                pair_plot = sns.pairplot(df[cols_to_plot], hue=hue_col)
                st.pyplot(pair_plot)
                return  # Skip the regular fig display for pairplot
            
            plt.tight_layout()
            st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error creating plot: {e}")

def train_ml_model(df, target):
    """Train machine learning model based on user selections"""
    st.subheader("Machine Learning Model Training")
    
    # Determine problem type
    if pd.api.types.is_numeric_dtype(df[target]):
        unique_values = df[target].nunique()
        if unique_values < 10 and unique_values > 0:
            problem_type = st.radio(
                "Problem type",
                ["classification", "regression"],
                index=0
            )
        else:
            problem_type = "regression"
    else:
        problem_type = "classification"
    
    st.write(f"**Problem Type:** {problem_type.capitalize()}")
    
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    # Encode target if classification
    label_encoder = None
    if problem_type == "classification":
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = y
    
    # Train-test split
    test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
    
    # Model selection
    model_options = {
        "classification": {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "XGBoost": XGBClassifier(),
            "Naive Bayes": GaussianNB(),
            "Neural Network": MLPClassifier(max_iter=1000)
        },
        "regression": {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "SVR": SVR(),
            "Decision Tree": DecisionTreeRegressor(),
            "KNN": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Neural Network": MLPRegressor(max_iter=1000)
        }
    }
    
    model_name = st.selectbox(
        f"Select {problem_type} model",
        list(model_options[problem_type].keys())
    
    # Hyperparameter tuning
    st.markdown("### Hyperparameter Tuning (Optional)")
    if st.checkbox("Enable hyperparameter tuning", False):
        param_grid = {}
        if model_name == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        elif model_name == "Logistic Regression":
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            }
        # Add more parameter grids for other models...
        
        if param_grid:
            grid_search = GridSearchCV(
                model_options[problem_type][model_name],
                param_grid,
                cv=5,
                scoring='accuracy' if problem_type == 'classification' else 'r2'
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            st.success(f"Best parameters: {grid_search.best_params_}")
        else:
            model = model_options[problem_type][model_name]
    else:
        model = model_options[problem_type][model_name]
    
    # Train model
    if st.button("Train Model"):
        with st.spinner(f"Training {model_name}..."):
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate model
            st.subheader("Model Evaluation")
            
            if problem_type == "classification":
                st.write("**Classification Report:**")
                st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_ if label_encoder else None))
                
                st.write("**Confusion Matrix:**")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                st.pyplot(fig)
                
                st.write("**Metrics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
                with col2:
                    st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2f}")
                with col3:
                    st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2f}")
                
            else:  # regression
                st.write("**Regression Metrics:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
                with col2:
                    st.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
                with col3:
                    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
                
                # Plot actual vs predicted
                fig, ax = plt.subplots()
                sns.scatterplot(x=y_test, y=y_pred, ax=ax)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)
            
            # Save model artifacts to session state
            st.session_state.trained_model = model
            st.session_state.scaler = scaler
            st.session_state.label_encoder = label_encoder
            st.session_state.model_trained = True
            st.session_state.feature_columns = X.columns.tolist()
            st.session_state.target_column = target
            st.session_state.problem_type = problem_type
            
            # Model download
            st.subheader("Model Download")
            model_bytes = io.BytesIO()
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'feature_columns': X.columns.tolist(),
                'target_column': target,
                'problem_type': problem_type
            }, model_bytes)
            st.download_button(
                label="Download Trained Model",
                data=model_bytes.getvalue(),
                file_name=f"{model_name.replace(' ', '_')}.pkl",
                mime="application/octet-stream"
            )
            
            return model

def make_predictions():
    """Make predictions using trained model"""
    st.subheader("Make Predictions")
    
    prediction_type = st.radio(
        "Prediction input method",
        ["Use test file", "Manual input"],
        horizontal=True
    )
    
    if prediction_type == "Use test file":
        test_file = st.file_uploader(
            "Upload test file (CSV or Excel)",
            type=['csv', 'xls', 'xlsx']
        )
        
        if test_file:
            try:
                if test_file.name.endswith('.csv'):
                    test_df = pd.read_csv(test_file)
                else:
                    test_df = pd.read_excel(test_file)
                
                # Check if features match training data
                missing_features = set(st.session_state.feature_columns) - set(test_df.columns)
                if missing_features:
                    st.error(f"Missing features in test data: {missing_features}")
                else:
                    test_df = test_df[st.session_state.feature_columns]
                    
                    # Scale features
                    test_scaled = st.session_state.scaler.transform(test_df)
                    
                    # Make predictions
                    predictions = st.session_state.trained_model.predict(test_scaled)
                    
                    # Decode predictions if classification
                    if st.session_state.label_encoder:
                        predictions = st.session_state.label_encoder.inverse_transform(predictions)
                    
                    # Add predictions to test data
                    result_df = test_df.copy()
                    result_df[f"Predicted_{st.session_state.target_column}"] = predictions
                    
                    st.write("**Predictions:**")
                    st.dataframe(result_df)
                    
                    # Download predictions
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error processing test file: {e}")
    
    else:  # Manual input
        input_data = {}
        cols = st.columns(3)
        for i, feature in enumerate(st.session_state.feature_columns):
            with cols[i % 3]:
                input_data[feature] = [st.number_input(feature, key=f"pred_{feature}")]
        
        if st.button("Predict"):
            input_df = pd.DataFrame(input_data)
            
            # Scale features
            input_scaled = st.session_state.scaler.transform(input_df)
            
            # Make prediction
            prediction = st.session_state.trained_model.predict(input_scaled)
            
            # Decode prediction if classification
            if st.session_state.label_encoder:
                prediction = st.session_state.label_encoder.inverse_transform(prediction)
            
            st.success(f"Predicted {st.session_state.target_column}: {prediction[0]}")

def main():
    st.title("🔍 Data Explorer & ML Pipeline")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_cleaned' not in st.session_state:
        st.session_state.data_cleaned = False
    if 'data_scaled' not in st.session_state:
        st.session_state.data_scaled = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Navigation
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Data Exploration", "Machine Learning"]
    )
    
    # File upload section
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
                    
                    # Download processed data
                    st.markdown("### Download Processed Data")
                    download_format = st.selectbox("Select download format", ["CSV", "Excel"])
                    
                    if download_format == "CSV":
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="processed_data.csv",
                            mime="text/csv"
                        )
                    else:
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False)
                        excel_bytes = excel_buffer.getvalue()
                        st.download_button(
                            label="Download as Excel",
                            data=excel_bytes,
                            file_name="processed_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            
            elif app_mode == "Machine Learning":
                st.header("Machine Learning Pipeline")
                
                # Show processed data status
                cleaning_status = "✅ Cleaned" if st.session_state.data_cleaned else "❌ Not cleaned"
                scaling_status = "✅ Scaled" if st.session_state.data_scaled else "❌ Not scaled"
                st.write(f"**Data Status:** {cleaning_status} | {scaling_status}")
                
                # Select target
                all_columns = df.columns.tolist()
                target = st.selectbox(
                    "Select target variable",
                    all_columns
                )
                
                # Train model
                if st.checkbox("Show model training options"):
                    trained_model = train_ml_model(df, target)
                
                # Make predictions if model is trained
                if st.session_state.get('model_trained', False):
                    make_predictions()

if __name__ == "__main__":
    main()
