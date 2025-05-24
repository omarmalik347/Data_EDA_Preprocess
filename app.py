import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import io  # For download functionality

# ... [keep all previous functions until main()] ...

def main():
    st.title("🔍 Data Explorer & Preprocessor")
    st.markdown("Upload your dataset, explore features, and prepare data for machine learning.")
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_cleaned' not in st.session_state:
        st.session_state.data_cleaned = False
    if 'data_scaled' not in st.session_state:
        st.session_state.data_scaled = False
    
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

if __name__ == "__main__":
    main()
