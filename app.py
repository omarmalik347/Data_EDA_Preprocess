import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

# Set page config
st.set_page_config(page_title="Data Explorer", layout="wide", page_icon="🔍")

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

def clean_data(df):
    """Clean the dataset based on user selections"""
    st.subheader("Data Cleaning Options")
    
    # Display missing values info
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.warning("Missing values detected in the dataset:")
        st.write(missing_values[missing_values > 0])
        
        # Handle missing values
        st.markdown("### Handle Missing Values")
        missing_cols = missing_values[missing_values > 0].index.tolist()
        for col in missing_cols:
            col1, col2 = st.columns(2)
            with col1:
                action = st.selectbox(
                    f"Action for '{col}' ({df[col].dtype})",
                    ["Drop column", "Drop rows", "Fill with mean/median/mode", "Fill with value"],
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
                elif action == "Fill with value":
                    fill_value = st.text_input("Fill value", key=f"fill_value_{col}")
                else:
                    st.write("")
        
        if st.button("Apply Missing Value Handling"):
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
            st.experimental_rerun()
    
    # Handle duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"Found {duplicates} duplicate rows in the dataset.")
        if st.checkbox("Remove duplicate rows"):
            df = df.drop_duplicates()
            st.success(f"Removed {duplicates} duplicate rows.")
            st.session_state.df = df.copy()
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
        
        return df_scaled
    
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
            "Pair Plot"
        ]
    )
    
    # For plots that need feature selection
    if plot_type not in ["Correlation Heatmap", "Pair Plot"]:
        selected_feature = st.selectbox("Select feature to plot", features)
    
    # For plots that need target selection (if target is specified)
    if target and plot_type not in ["Correlation Heatmap", "Pair Plot", "Histogram"]:
        use_target = st.checkbox("Use target variable in plot", True)
    else:
        use_target = False
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        if plot_type == "Scatter Plot":
            if use_target:
                sns.scatterplot(data=df, x=selected_feature, y=target, ax=ax)
            else:
                sns.scatterplot(data=df, x=selected_feature, ax=ax)
            ax.set_title(f"Scatter Plot: {selected_feature} vs {target if use_target else 'count'}")
        
        elif plot_type == "Box Plot":
            if use_target:
                sns.boxplot(data=df, x=target, y=selected_feature, ax=ax)
            else:
                sns.boxplot(data=df, y=selected_feature, ax=ax)
            ax.set_title(f"Box Plot: {selected_feature} by {target if use_target else ''}")
        
        elif plot_type == "Violin Plot":
            if use_target:
                sns.violinplot(data=df, x=target, y=selected_feature, ax=ax)
            else:
                sns.violinplot(data=df, y=selected_feature, ax=ax)
            ax.set_title(f"Violin Plot: {selected_feature} by {target if use_target else ''}")
        
        elif plot_type == "Bar Plot":
            if use_target:
                sns.barplot(data=df, x=selected_feature, y=target, ax=ax, estimator=np.mean)
            else:
                df[selected_feature].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Bar Plot: {selected_feature} vs {target if use_target else 'count'}")
        
        elif plot_type == "Line Plot":
            if use_target:
                sns.lineplot(data=df, x=selected_feature, y=target, ax=ax)
            else:
                st.warning("Line plot requires a target variable")
                return
            ax.set_title(f"Line Plot: {selected_feature} vs {target}")
        
        elif plot_type == "Histogram":
            sns.histplot(data=df, x=selected_feature, kde=True, ax=ax)
            ax.set_title(f"Distribution of {selected_feature}")
        
        elif plot_type == "Correlation Heatmap":
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation heatmap")
                return
            
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Correlation Heatmap")
        
        elif plot_type == "Pair Plot":
            cols_to_plot = st.multiselect(
                "Select columns for pair plot",
                df.columns.tolist(),
                default=features[:5]  # Limit to first 5 features by default
            )
            
            if len(cols_to_plot) < 2:
                st.warning("Please select at least 2 columns for pair plot")
                return
            
            if target:
                hue_col = target
            else:
                hue_col = None
            
            pair_plot = sns.pairplot(df[cols_to_plot + ([target] if target else [])], hue=hue_col)
            st.pyplot(pair_plot)
            return  # Skip the regular fig display for pairplot
        
        plt.tight_layout()
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error creating plot: {e}")

def main():
    st.title("🔍 Data Explorer & Preprocessor")
    st.markdown("Upload your dataset, explore features, and prepare data for machine learning.")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    
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
            
            # Display basic info
            st.sidebar.success("Data loaded successfully!")
            st.sidebar.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Show raw data preview
            if st.sidebar.checkbox("Show raw data"):
                st.subheader("Raw Data Preview")
                st.dataframe(df.head())
            
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
            if st.checkbox("Show data cleaning options"):
                df = clean_data(df)
                st.session_state.df = df.copy()
            
            # Feature scaling section
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols and st.checkbox("Show feature scaling options"):
                df = scale_data(df, numeric_cols)
                st.session_state.df = df.copy()
            
            # Data visualization section
            st.header("Data Exploration")
            plot_data(df, features, target)
            
            # Show processed data
            if st.checkbox("Show processed data"):
                st.subheader("Processed Data")
                st.dataframe(df.head())

if __name__ == "__main__":
    main()
