import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from main import EnhancedDataQualityPlatform, DatasetInfo
from main_dynamic import load_best_model_config, get_model, get_embeddings
import great_expectations as ge  # Keep this import
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tkinter as tk
from tkinter import filedialog
import io
import warnings
import json
from langchain.vectorstores import FAISS
from langchain.schema import Document

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

def get_data_path():
    return os.path.join(os.path.dirname(__file__), "data", "Amazon Sales 2023")

def chunk_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

@st.cache_resource
def initialize_platform():
    config_path = "best_model_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        st.warning("Best model config not found. Using default configuration.")
        config = {
            "llm": "huggingface_t5",
            "embedding": "sentence-transformers/all-MiniLM-L6-v2"
        }
    return EnhancedDataQualityPlatform(
        llm=get_model(config),
        embeddings=get_embeddings(config),
        use_great_expectations=False  # Set this to False to disable Great Expectations
    )

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")
        
        # Replace empty strings with NaN
        df = df.replace('', np.nan)
        
        # Convert numeric columns
        numeric_columns = ['ratings', 'no_of_ratings', 'discount_price', 'actual_price']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure no_of_ratings is integer if it exists
        if 'no_of_ratings' in df.columns:
            df['no_of_ratings'] = df['no_of_ratings'].astype('Int64')
        
        return df
    except ValueError as ve:
        st.error(f"Value error: {str(ve)}")
        raise
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please upload a valid file.")
        raise
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        raise

@st.cache_data
def load_predefined_dataset(dataset_name):
    try:
        base_path = get_data_path()
        if dataset_name == "appliances":
            file_path = os.path.join(base_path, "All Appliances.csv")
        elif dataset_name == "electronics":
            file_path = os.path.join(base_path, "All Electronics.csv")
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
        
        # Read CSV with all columns as strings initially
        df = pd.read_csv(file_path, dtype=str)
        
        # Convert numeric columns
        numeric_columns = ['ratings', 'no_of_ratings', 'discount_price', 'actual_price']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure no_of_ratings is integer if it exists
        if 'no_of_ratings' in df.columns:
            df['no_of_ratings'] = df['no_of_ratings'].astype('Int64')
        
        # Truncate long strings in object columns
        max_string_length = 200  # Adjust this value as needed
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.slice(0, max_string_length)
        
        # Ensure index is of type int64
        df.index = df.index.astype('int64')
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def browse_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

def main():
    st.set_page_config(page_title="Enhanced Data Quality Platform", layout="wide")
    st.title("Enhanced Data Quality Platform")
    st.markdown("This platform provides advanced data quality assessment and analysis tools.")

    try:
        platform = initialize_platform()
        st.success("Data Quality Platform initialized successfully!")
    except Exception as e:
        st.error(f"Error initializing platform: {str(e)}")
        return

    with st.sidebar:
        st.header("Data Selection")
        st.markdown("Choose a data source to begin your analysis.")
        data_option = st.radio("Choose data source:", ["Upload Your Own", "Use Pre-defined Dataset"])
        
        if data_option == "Upload Your Own":
            uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])
            if uploaded_file is not None:
                data = load_data(uploaded_file)
                dataset_name = uploaded_file.name.split('.')[0]
                platform.load_dataset(DatasetInfo(name=dataset_name, file_path=uploaded_file.name, file_type='csv'))
        else:
            dataset_name = st.selectbox("Select a pre-defined dataset:", ["appliances", "electronics"])
            if st.button("Load Selected Dataset"):
                data = load_predefined_dataset(dataset_name)
                if data is not None:
                    file_path = os.path.join(get_data_path(), f"All {dataset_name.capitalize()}.csv")
                    platform.load_dataset(DatasetInfo(name=dataset_name, file_path=file_path, file_type='csv'))
                    st.success(f"Dataset '{dataset_name}' loaded successfully!")
                    
                    # Display a sample of the data
                    st.subheader("Data Sample")
                    st.dataframe(data.head())
                    
                    # Display summary statistics
                    st.subheader("Summary Statistics")
                    st.write(data.describe())
                else:
                    st.error("Failed to load dataset. Please check the file path and try again.")

    if hasattr(platform, 'datasets') and platform.datasets:
        st.header("Data Analysis")
        st.markdown("Explore various aspects of your data using the tabs below.")
        
        dataset_name = list(platform.datasets.keys())[0]  # Get the name of the loaded dataset
        data = platform.datasets[dataset_name]

        st.write(f"Analyzing dataset: {dataset_name}")
        st.write(f"Dataset shape: {data.shape}")

        tabs = st.tabs(["Data Overview", "Quality Assessment", "Metadata", "Semantic Search", "RAG Query", "Advanced Analyses"])
        
        with tabs[0]:
            st.subheader("Data Overview")
            st.write(f"Shape: {data.shape}")
            st.write(f"Columns: {', '.join(data.columns)}")
            
            if st.checkbox("Show data sample"):
                st.dataframe(data.head().to_dict('records'))
            
            if st.checkbox("Show data statistics"):
                st.write(data.describe().to_dict())
            
            if st.checkbox("Show data info"):
                buffer = io.StringIO()
                data.info(buf=buffer)
                st.text(buffer.getvalue())

        with tabs[1]:
            st.subheader("Data Quality Assessment")
            st.markdown("Assess the quality of your data, including missing values, data types, and basic statistics.")
            if st.button("Assess Data Quality"):
                with st.spinner("Assessing data quality..."):
                    try:
                        start_time = time.time()
                        quality_report = platform.assess_data_quality(dataset_name)
                        end_time = time.time()
                        st.success(f"Assessment completed in {end_time - start_time:.2f} seconds")
                        
                        st.subheader("Basic Statistics")
                        st.write(quality_report["basic_stats"])
                        
                        st.subheader("Missing Values")
                        missing_df = pd.DataFrame.from_dict(quality_report["missing_values"], orient="index", columns=["Count"])
                        missing_df["Percentage"] = missing_df["Count"] / quality_report["basic_stats"]["row_count"] * 100
                        st.dataframe(missing_df)
                        
                        st.subheader("Data Types")
                        st.write(quality_report["data_types"])
                        
                        st.subheader("Unique Values")
                        st.write(quality_report["unique_values"])
                        
                        st.subheader("Numeric Statistics")
                        st.dataframe(pd.DataFrame(quality_report["numeric_stats"]))
                        
                        st.subheader("Categorical Statistics")
                        for col, stats in quality_report["categorical_stats"].items():
                            st.write(f"Top 5 categories for {col}:")
                            st.write(pd.Series(stats).nlargest(5))
                        
                        st.subheader("Sample Data")
                        st.dataframe(pd.DataFrame(quality_report["sample_data"]))
                        
                    except Exception as e:
                        st.error(f"Error assessing data quality: {str(e)}")

        with tabs[2]:
            st.subheader("Metadata Generation")
            if st.button("Generate Metadata"):
                with st.spinner("Generating metadata..."):
                    try:
                        metadata = platform.generate_metadata(dataset_name)
                        st.json(metadata)
                    except Exception as e:
                        st.error(f"Error generating metadata: {str(e)}")
                        st.error("Full error:")
                        st.exception(e)

        with tabs[3]:
            st.subheader("Semantic Search")
            search_query = st.text_input("Enter a search query:")
            k = st.slider("Number of results", 1, 20, 5)
            if st.button("Perform Semantic Search"):
                with st.spinner("Searching..."):
                    try:
                        results = platform.semantic_search(search_query, dataset_name, k=k)
                        for i, result in enumerate(results, 1):
                            st.subheader(f"Result {i}")
                            st.write(f"Content: {result['content']}")
                            st.write(f"Metadata: {result['metadata']}")
                            st.write(f"Score: {result['score']}")
                            st.write("---")
                    except Exception as e:
                        st.error(f"Error performing semantic search: {str(e)}")
                        st.exception(e)

        with tabs[4]:
            st.subheader("RAG Query")
            rag_query = st.text_input("Enter a RAG query:")
            if st.button("Perform RAG Query"):
                with st.spinner("Executing query..."):
                    try:
                        result = platform.rag_query(rag_query, dataset_name)
                        st.subheader("Answer")
                        st.write(result['answer'])
                        st.subheader("Source Documents")
                        for i, doc in enumerate(result['source_documents'], 1):
                            st.write(f"Document {i}")
                            st.write(f"Content: {doc['content']}")
                            st.write(f"Metadata: {doc['metadata']}")
                            st.write("---")
                    except Exception as e:
                        st.error(f"Error performing RAG query: {str(e)}")
                        st.exception(e)

        with tabs[5]:
            st.subheader("Advanced Analyses")
            analysis_type = st.selectbox("Select Analysis Type", 
                                           ["Time Series Analysis", "Hypothesis Testing", "Anomaly Detection"])
            
            if analysis_type == "Time Series Analysis":
                date_column = st.selectbox("Select date column", data.columns)
                value_column = st.selectbox("Select value column", data.select_dtypes(include=['number']).columns)
                if st.button("Perform Time Series Analysis"):
                    with st.spinner("Analyzing time series..."):
                        try:
                            results = platform.perform_time_series_analysis(dataset_name, date_column, value_column)
                            st.line_chart(results)
                        except Exception as e:
                            st.error(f"Error performing time series analysis: {str(e)}")
                            st.exception(e)
            
            elif analysis_type == "Hypothesis Testing":
                group_column = st.selectbox("Select group column", data.select_dtypes(include=['object']).columns)
                value_column = st.selectbox("Select value column", data.select_dtypes(include=['number']).columns)
                if st.button("Perform Hypothesis Testing"):
                    with st.spinner("Performing hypothesis test..."):
                        try:
                            results = platform.perform_hypothesis_testing(dataset_name, group_column, value_column)
                            st.write(f"T-statistic: {results['t_statistic']}")
                            st.write(f"P-value: {results['p_value']}")
                        except Exception as e:
                            st.error(f"Error performing hypothesis testing: {str(e)}")
                            st.exception(e)
            
            elif analysis_type == "Anomaly Detection":
                column = st.selectbox("Select column for anomaly detection (optional)", [None] + list(data.select_dtypes(include=['number']).columns))
                if st.button("Detect Anomalies"):
                    with st.spinner("Detecting anomalies..."):
                        try:
                            anomalies = platform.detect_anomalies(dataset_name, column)
                            st.write(f"Detected {len(anomalies)} anomalies")
                            st.dataframe(anomalies)
                        except Exception as e:
                            st.error(f"Error detecting anomalies: {str(e)}")
                            st.exception(e)
    else:
        st.info("Please select a dataset in the sidebar to begin your analysis.")

    st.sidebar.markdown("---")
    st.sidebar.info("This app demonstrates the capabilities of the Enhanced Data Quality Platform. "
                    "You can upload your own dataset or use a pre-defined one to explore various data analysis and quality assessment features.")

    if 'data' not in locals() or data is None:
        st.warning("Please load a dataset to begin analysis.")
        return

if __name__ == "__main__":
    main()