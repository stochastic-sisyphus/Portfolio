import os
import logging
from typing import List, Dict, Any, Union, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
import gzip
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.feature_selection import mutual_info_classif, f_classif
from concurrent.futures import ThreadPoolExecutor, as_completed
import networkx as nx
from scipy.stats import ks_2samp, chi2_contingency, ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import io
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DataFrameLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import torch
import great_expectations as ge
from great_expectations.data_context.types.base import DataContextConfig, FilesystemStoreBackendDefaults
from great_expectations.data_context import get_context
from tqdm import tqdm
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from PIL import Image
import torchvision.transforms as transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from causalinference import CausalModel
from alibi_detect.cd import TabularDrift
from scipy.stats import pearsonr
import geopandas as gpd
from shapely.geometry import Point
from lifelines import KaplanMeierFitter
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.parsing.preprocessing import STOPWORDS
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from langchain.schema.runnable import RunnablePassthrough
import re
from json import JSONEncoder
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from scipy import stats

print("Script initialized")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DatasetInfo:
    name: str
    file_path: str
    file_type: str

class NaNHandler(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.NaT, pd.NA)):
            return None
        if pd.isna(obj):
            return None
        return JSONEncoder.default(self, obj)

def preprocess_dataframe(df):
    # Option 1: Replace NaN/NaT with a placeholder
    df = df.fillna("N/A")
    
    # Option 2: Drop rows with NaN/NaT
    # df = df.dropna()
    
    return df

def initialize_great_expectations():
    try:
        store_backend_defaults = FilesystemStoreBackendDefaults(root_directory="./great_expectations")
        data_context_config = DataContextConfig(
            store_backend_defaults=store_backend_defaults,
            checkpoint_store_name=None,
            expectations_store_name="expectations_store",
        )
        context = ge.data_context.BaseDataContext(project_config=data_context_config)
        return context
    except Exception as e:
        print(f"Failed to initialize Great Expectations: {str(e)}")
        return None

class EnhancedDataQualityPlatform:
    def __init__(self, llm=None, embeddings=None, use_great_expectations=False):
        self.datasets = {}
        self.quality_reports = {}
        self.metadata = {}
        self.recommendations = {}
        self.vector_stores = {}
        self.fine_tuned_models = {}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm = llm or HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="google/flan-t5-base", device=device))
        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        self.use_great_expectations = use_great_expectations
        if self.use_great_expectations:
            try:
                self.ge_context = ge.data_context.DataContext()
            except Exception as e:
                logging.error(f"Failed to initialize Great Expectations: {str(e)}")
                self.use_great_expectations = False

    def load_dataset(self, dataset_info: DatasetInfo):
        try:
            logging.info(f"Loading dataset: {dataset_info.name} from {dataset_info.file_path}")
            if dataset_info.file_type == 'csv':
                df = pd.read_csv(dataset_info.file_path)
            elif dataset_info.file_type == 'json':
                df = pd.read_json(dataset_info.file_path)
            else:
                raise ValueError(f"Unsupported file type: {dataset_info.file_type}")
            
            self.datasets[dataset_info.name] = df
            logging.info(f"Dataset {dataset_info.name} loaded successfully. Shape: {df.shape}")
        except Exception as e:
            logging.error(f"Error loading dataset {dataset_info.name}: {str(e)}")
            raise

    def assess_data_quality(self, dataset_name):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        report = {
            "basic_stats": {
                "row_count": len(df),
                "column_count": len(df.columns),
            },
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "unique_values": df.nunique().to_dict(),
            "sample_data": df.head().to_dict(),
            "numeric_stats": df.describe().to_dict(),
            "categorical_stats": {col: df[col].value_counts().to_dict() for col in df.select_dtypes(include=['object', 'category']).columns},
        }
        
        return report

    def _great_expectations_checks(self, dataset_name: str) -> Dict[str, Any]:
        if self.ge_context is None:
            return {"error": "Great Expectations not enabled or failed to initialize"}
        
        try:
            data = self.datasets[dataset_name]
            expectation_suite = self.ge_context.create_expectation_suite(f"{dataset_name}_suite")
            validator = self.ge_context.get_validator(
                batch_request=self.ge_context.get_batch_request(data),
                expectation_suite=expectation_suite
            )
            
            # Add some basic expectations
            validator.expect_column_values_to_not_be_null(column="*")
            validator.expect_column_values_to_be_unique(column="*")
            
            # Run the validation
            results = validator.validate()
            
            return {
                "success_percent": results.success_percent,
                "results": results.to_json_dict()
            }
        except Exception as e:
            logging.error(f"Error in Great Expectations checks: {str(e)}")
            return {"error": str(e)}

    def _basic_quality_checks(self, data: pd.DataFrame) -> Dict[str, Any]:
        checks = {
            "missing_values": data.isnull().sum().to_dict(),
            "missing_percentages": (data.isnull().sum() / len(data) * 100).to_dict(),
            "data_types": data.dtypes.astype(str).to_dict(),
            "unique_values": data.nunique().to_dict(),
            "numeric_stats": {},
            "categorical_stats": {},
            "datetime_stats": {}
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        for col in numeric_cols:
            checks["numeric_stats"][col] = {
                "mean": data[col].mean(),
                "median": data[col].median(),
                "std": data[col].std(),
                "min": data[col].min(),
                "max": data[col].max(),
                "skew": data[col].skew(),
                "kurtosis": data[col].kurtosis()
            }
        
        for col in categorical_cols:
            checks["categorical_stats"][col] = data[col].value_counts().to_dict()
        
        for col in datetime_cols:
            checks["datetime_stats"][col] = {
                "min": data[col].min().isoformat() if pd.notnull(data[col].min()) else None,
                "max": data[col].max().isoformat() if pd.notnull(data[col].max()) else None,
                "range": (data[col].max() - data[col].min()).total_seconds() / 86400 if pd.notnull(data[col].min()) and pd.notnull(data[col].max()) else None  # Range in days
            }
        
        return checks

    def _llm_quality_assessment(self, data: pd.DataFrame) -> str:
        if self.llm is None:
            return "LLM not initialized"
        
        summary = data.describe(include='all').to_string()
        prompt = f"""
        Analyze the following dataset summary and provide insights on data quality:
        
        {summary}
        
        Please comment on:
        1. The range and distribution of numeric variables
        2. Any potential data quality issues (e.g., outliers, missing values)
        3. Suggestions for data cleaning or preprocessing
        
        Provide your analysis in a structured format.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Error in LLM assessment: {str(e)}"

    def _advanced_quality_checks(self, data: pd.DataFrame) -> Dict[str, Any]:
        checks = {
            "high_correlation_pairs": [],
            "potential_outliers": {}
        }
        
        # Check for high correlations
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            corr_matrix = numeric_data.corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.8)
            high_corr_list = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                              for x, y in zip(*high_corr) if x != y and x < y]
            checks["high_correlation_pairs"] = high_corr_list
            
            # Check for potential outliers
            for col in numeric_data.columns:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = numeric_data[(numeric_data[col] < (Q1 - 1.5 * IQR)) | (numeric_data[col] > (Q3 + 1.5 * IQR))]
                checks["potential_outliers"][col] = len(outliers)
        
        return checks

    def generate_metadata(self, dataset_name):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        metadata = {
            "dataset_name": dataset_name,
            "num_rows": int(len(df)),
            "num_columns": int(len(df.columns)),
            "columns": {}
        }
        
        for col in df.columns:
            col_metadata = {
                "dtype": str(df[col].dtype),
                "num_unique_values": int(df[col].nunique()),
                "num_missing_values": int(df[col].isnull().sum())
            }
            if df[col].dtype in ['int64', 'float64']:
                col_metadata.update({
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
                })
            elif df[col].dtype == 'object':
                col_metadata.update({
                    "sample_values": df[col].dropna().sample(min(5, df[col].nunique())).tolist()
                })
            metadata["columns"][col] = col_metadata
        
        return metadata

    def build_vector_store(self, dataset_name: str):
        # Implement vector store building logic
        pass

    def semantic_search(self, query, dataset_name, k=5):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        if dataset_name not in self.vector_stores:
            self.build_vector_store(dataset_name)
        
        vector_store = self.vector_stores[dataset_name]
        results = vector_store.similarity_search_with_score(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            } for doc, score in results
        ]

    def rag_query(self, query, dataset_name):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        if dataset_name not in self.vector_stores:
            self.build_vector_store(dataset_name)
        
        vector_store = self.vector_stores[dataset_name]
        retriever = vector_store.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": query})
        
        return {
            "answer": result['result'],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result['source_documents']
            ]
        }

    def update_document_metadata(self, dataset_name: str, doc_id: int, new_metadata: Dict[str, Any]):
        # Implement document metadata update logic
        pass

    def generate_recommendations(self, dataset_name: str):
        logging.info(f"Generating recommendations for: {dataset_name}")
        quality_report = self.quality_reports[dataset_name]
        metadata = self.metadata[dataset_name]
        recommendations = self._llm_generate_recommendations(quality_report, metadata)
        self.recommendations[dataset_name] = recommendations
        return recommendations

    def _llm_generate_recommendations(self, quality_report: Dict[str, Any], metadata: Dict[str, str]) -> Dict[str, str]:
        template = """
        Based on the following data quality report and metadata, provide recommendations for improving the dataset:
        Quality Report:
        {quality_report}
        Metadata:
        {metadata}
        Provide a comprehensive list of specific recommendations for:
        1. Data cleaning
        2. Data enrichment
        3. Data transformation
        4. Analysis approaches
        5. Visualization ideas
        6. Integration possibilities
        7. Quality monitoring
        8. Ethical considerations
        9. Performance optimization
        10. Data governance
        For each recommendation, provide a brief explanation of its potential impact and implementation approach.
        """
        prompt = PromptTemplate(template=template, input_variables=["quality_report", "metadata"])
        chain = prompt | self.llm | RunnablePassthrough()
        recommendations_text = chain.invoke(quality_report=json.dumps(quality_report), metadata=json.dumps(metadata))
        
        # Parse the text response into a structured format
        recommendations = {}
        sections = recommendations_text.split("\n\n")
        for section in sections:
            if ":" in section:
                key, value = section.split(":", 1)
                recommendations[key.strip()] = value.strip()
        return recommendations

    def visualize_data_quality(self, dataset_name: str) -> None:
        """
        Generate a data quality dashboard for the specified dataset.
        
        Args:
            dataset_name (str): Name of the dataset to visualize.
        """
        try:
            # Implement visualization logic here
            fig = self._create_data_quality_figure(dataset_name)
            fig.update_layout(height=1200, width=1000, title_text=f"Data Quality Dashboard - {dataset_name}")
            fig.write_html(f"{dataset_name}_data_quality_dashboard.html")
            logging.info(f"Data quality dashboard generated for {dataset_name}")
        except Exception as e:
            logging.error(f"Error generating data quality dashboard for {dataset_name}: {str(e)}")

    def build_vector_store(self, dataset_name: str):
        logging.info(f"Building vector store for dataset: {dataset_name}")
        try:
            df = self.datasets[dataset_name]
            
            # Preprocess the dataframe
            df = preprocess_dataframe(df)
            
            # Convert to JSON using the custom encoder
            json_data = json.dumps(df.to_dict(orient='records'), cls=NaNHandler)
            
            # Continue with your vector store building logic using json_data
            # ...

        except Exception as e:
            logging.error(f"Error building vector store for dataset {dataset_name}: {str(e)}")
            raise

    def evaluate_search_performance(self, test_queries: List[str], ground_truth: List[List[int]]):
        performance_metrics = {
            "ndcg": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
        for query, truth in zip(test_queries, ground_truth):
            search_results = self.semantic_search(query, k=10)
            if not search_results:
                logging.warning(f"No search results found for query: {query}")
                continue
            
            dataset_name = list(search_results.keys())[0]
            predicted_indices = [int(json.loads(result['content'])['index']) for result in search_results[dataset_name]]
            
            ndcg = ndcg_score([truth], [predicted_indices])
            precision = precision_score(truth, predicted_indices, average='micro', zero_division=0)
            recall = recall_score(truth, predicted_indices, average='micro', zero_division=0)
            f1 = f1_score(truth, predicted_indices, average='micro', zero_division=0)
            
            performance_metrics["ndcg"].append(ndcg)
            performance_metrics["precision"].append(precision)
            performance_metrics["recall"].append(recall)
            performance_metrics["f1"].append(f1)
        
        # Calculate average metrics
        for metric in performance_metrics:
            if performance_metrics[metric]:
                performance_metrics[metric] = np.mean(performance_metrics[metric])
            else:
                performance_metrics[metric] = None
        
        return performance_metrics

    def fine_tune_model(self, dataset_name: str, model_name: str = "distilbert-base-uncased") -> None:
        """
        Fine-tune a model on the specified dataset.
        
        Args:
            dataset_name (str): Name of the dataset to use for fine-tuning.
            model_name (str): Name of the pre-trained model to fine-tune.
        """
        logging.info(f"Fine-tuning model for dataset: {dataset_name}")
        try:
            data = self.datasets[dataset_name]
            
            if 'text' not in data.columns or 'label' not in data.columns:
                raise ValueError(f"Dataset {dataset_name} does not have 'text' and 'label' columns for fine-tuning")
            
            texts = data['text'].tolist()
            labels = data['label'].tolist()
            
            train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            train_encodings = tokenizer(train_texts, truncation=True, padding=True)
            val_encodings = tokenizer(val_texts, truncation=True, padding=True)
            
            train_dataset = self._create_torch_dataset(train_encodings, train_labels)
            val_dataset = self._create_torch_dataset(val_encodings, val_labels)
            
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            trainer.train()
            
            model.save_pretrained(f"./fine_tuned_{dataset_name}")
            tokenizer.save_pretrained(f"./fine_tuned_{dataset_name}")
            
            self.fine_tuned_models[dataset_name] = model
            logging.info(f"Fine-tuning completed for dataset: {dataset_name}")
        except Exception as e:
            logging.error(f"Error fine-tuning model for dataset {dataset_name}: {str(e)}")

    def _create_torch_dataset(self, encodings, labels):
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        return Dataset(encodings, labels)

    def analyze_data_drift(self, dataset_name: str, new_data: pd.DataFrame):
        logging.info(f"Analyzing data drift for dataset: {dataset_name}")
        original_data = self.datasets[dataset_name]
        
        drift_report = {}
        
        # Check for schema drift
        original_columns = set(original_data.columns)
        new_columns = set(new_data.columns)
        added_columns = new_columns - original_columns
        removed_columns = original_columns - new_columns
        
        drift_report['schema_drift'] = {
            'added_columns': list(added_columns),
            'removed_columns': list(removed_columns)
        }
        
        # Check for statistical drift in numerical columns
        numerical_columns = original_data.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if col in new_data.columns:
                ks_statistic, p_value = ks_2samp(original_data[col], new_data[col])
                drift_report[f'{col}_drift'] = {
                    'ks_statistic': ks_statistic,
                    'p_value': p_value,
                    'significant_drift': p_value < 0.05
                }
        
        # Check for categorical drift
        categorical_columns = original_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col in new_data.columns:
                chi2, p_value, dof, expected = chi2_contingency([
                    original_data[col].value_counts(),
                    new_data[col].value_counts()
                ])
                drift_report[f'{col}_categorical_drift'] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'significant_drift': p_value < 0.05
                }
        
        return drift_report

    def detect_anomalies(self, dataset_name: str, method: str = 'isolation_forest'):
        logging.info(f"Detecting anomalies in dataset: {dataset_name} using method: {method}")
        data = self.datasets[dataset_name]
        numeric_data = data.select_dtypes(include=[np.number])
        
        if method == 'isolation_forest':
            clf = IsolationForest(contamination=0.1, random_state=42)
            anomalies = clf.fit_predict(numeric_data)
        elif method == 'local_outlier_factor':
            clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            anomalies = clf.fit_predict(numeric_data)
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
        
        anomaly_indices = np.where(anomalies == -1)[0]
        return data.iloc[anomaly_indices]

    def generate_synthetic_data(self, dataset_name: str, num_samples: int):
        logging.info(f"Generating synthetic data for dataset: {dataset_name}")
        data = self.datasets[dataset_name]
        
        synthetic_data = pd.DataFrame(index=range(num_samples))
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                synthetic_data[col] = np.random.uniform(data[col].min(), data[col].max(), num_samples)
            elif pd.api.types.is_categorical_dtype(data[col]):
                synthetic_data[col] = np.random.choice(data[col].cat.categories, num_samples)
            elif pd.api.types.is_object_dtype(data[col]):
                synthetic_data[col] = np.random.choice(data[col].unique(), num_samples)
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                start_date = data[col].min()
                end_date = data[col].max()
                date_range = (end_date - start_date).total_seconds()
                synthetic_data[col] = start_date + pd.to_timedelta(np.random.uniform(0, date_range, num_samples), unit='s')
            else:
                logging.warning(f"Unsupported dtype {data[col].dtype} for column {col}. Skipping.")
        
        return synthetic_data

    def perform_feature_selection(self, dataset_name: str, target_column: str, method: str = 'mutual_info', top_n: int = None):
        logging.info(f"Performing feature selection for dataset: {dataset_name} using method: {method}")
        data = self.datasets[dataset_name]
        
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        if not categorical_columns.empty:
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(X[categorical_columns])
            encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
            X_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
            X = pd.concat([X.drop(columns=categorical_columns), X_encoded], axis=1)
        
        # Handle non-numeric target
        if not pd.api.types.is_numeric_dtype(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        if X.empty:
            raise ValueError("No features available for selection after preprocessing")
        
        try:
            if method == 'mutual_info':
                mi_scores = mutual_info_classif(X, y)
                feature_importance = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
            elif method == 'f_classif':
                f_scores, _ = f_classif(X, y)
                feature_importance = pd.Series(f_scores, index=X.columns).sort_values(ascending=False)
            elif method == 'random_forest':
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                feature_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            else:
                raise ValueError(f"Unsupported feature selection method: {method}")
        except Exception as e:
            logging.error(f"Error during feature selection: {str(e)}")
            raise
        
        logging.info(f"Feature selection completed. Top 5 features: {feature_importance.head()}")
        
        if top_n is not None:
            feature_importance = feature_importance.head(top_n)
            logging.info(f"Returning top {top_n} features")
        
        return feature_importance

    def perform_dimensionality_reduction(self, dataset_name: str, n_components: int = 2):
        logging.info(f"Performing dimensionality reduction for dataset: {dataset_name}")
        data = self.datasets[dataset_name]
        
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            raise ValueError("No numeric features available for dimensionality reduction")
        
        # Standardize the features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        try:
            # Perform PCA
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(scaled_data)
            
            # Create a DataFrame with the reduced data
            reduced_df = pd.DataFrame(
                reduced_data, 
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=data.index
            )
            
            # Calculate explained variance ratio
            explained_variance_ratio = pca.explained_variance_ratio_
            
            logging.info(f"Dimensionality reduction completed. Explained variance ratio: {explained_variance_ratio}")
            
            return reduced_df, explained_variance_ratio
        except Exception as e:
            logging.error(f"Error during dimensionality reduction: {str(e)}")
            raise

    def perform_clustering(self, dataset_name: str, n_clusters: int = 3):
        logging.info(f"Performing clustering for dataset: {dataset_name}")
        data = self.datasets[dataset_name]
        numeric_data = data.select_dtypes(include=[np.number])
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        data['cluster'] = cluster_labels
        
        return data, kmeans.cluster_centers_

    def perform_correlation_analysis(self, dataset_name: str):
        logging.info(f"Performing correlation analysis for dataset: {dataset_name}")
        data = self.datasets[dataset_name]
        numeric_data = data.select_dtypes(include=[np.number])
        
        correlation_matrix = numeric_data.corr()
        
        # Create a network graph of correlations
        G = nx.Graph()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.5:  # Only add edges for correlations > 0.5
                    G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], 
                               weight=correlation_matrix.iloc[i, j])
        
        return correlation_matrix, G

    def perform_time_series_analysis(self, dataset_name, date_column, value_column):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        df = df.set_index(date_column)
        return df[value_column].resample('D').mean().to_dict()

    def perform_hypothesis_testing(self, dataset_name, group_column, value_column):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        groups = df[group_column].unique()
        if len(groups) != 2:
            raise ValueError("Hypothesis testing requires exactly two groups")
        group1 = df[df[group_column] == groups[0]][value_column]
        group2 = df[df[group_column] == groups[1]][value_column]
        t_stat, p_value = stats.ttest_ind(group1, group2)
        return {"t_statistic": t_stat, "p_value": p_value}

    def detect_anomalies(self, dataset_name, column=None):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        if column:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")
            data = df[[column]]
        else:
            data = df.select_dtypes(include=[np.number])
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(data)
        return df[anomalies == -1]

    def generate_report(self, dataset_name: str):
        logging.info(f"Generating comprehensive report for dataset: {dataset_name}")
        
        report = {
            'dataset_name': dataset_name,
            'quality_report': self.quality_reports[dataset_name],
            'metadata': self.metadata[dataset_name],
            'recommendations': self.recommendations[dataset_name]
        }
        
        # Add visualizations
        self.visualize_data_quality(dataset_name)
        report['visualizations'] = f"{dataset_name}_data_quality_dashboard.html"
        
        # Add feature importance
        if 'label' in self.datasets[dataset_name].columns:
            feature_importance = self.perform_feature_selection(dataset_name, 'label', method='random_forest')
            report['feature_importance'] = feature_importance.to_dict()
        
        # Add dimensionality reduction results
        reduced_data, explained_variance_ratio = self.perform_dimensionality_reduction(dataset_name)
        report['pca_results'] = {
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance_ratio': np.cumsum(explained_variance_ratio).tolist()
        }
        
        # Add clustering results
        clustered_data, cluster_centers = self.perform_clustering(dataset_name)
        report['clustering_results'] = {
            'num_clusters': len(cluster_centers),
            'cluster_sizes': clustered_data['cluster'].value_counts().to_dict()
        }
        
        # Add correlation analysis
        correlation_matrix, correlation_graph = self.perform_correlation_analysis(dataset_name)
        report['correlation_analysis'] = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': [
                (correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
                for i in range(len(correlation_matrix.columns))
                for j in range(i+1, len(correlation_matrix.columns))
                if abs(correlation_matrix.iloc[i, j]) > 0.7
            ]
        }
        
        return report

    def save_report(self, report: Dict[str, Any], output_file: str):
        logging.info(f"Saving report to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

    def run_pipeline(self, dataset_info: DatasetInfo):
        logging.info(f"Running pipeline for dataset: {dataset_info.name}")
        steps = [
            ("Load dataset", lambda: self.load_dataset(dataset_info.name)),
            ("Assess data quality", lambda: self.assess_data_quality(dataset_info.name)),
            ("Generate improvement suggestions", lambda: self.generate_improvement_suggestions(dataset_info.name)),
            ("Generate metadata", lambda: self.generate_metadata(dataset_info.name)),
            ("Generate recommendations", lambda: self.generate_recommendations(dataset_info.name)),
            ("Visualize data quality", lambda: self.visualize_data_quality(dataset_info.name)),
            ("Build vector store", lambda: self.build_vector_store(dataset_info.name)),
            ("Perform feature selection", lambda: self.perform_feature_selection(dataset_info.name, 'label') if 'label' in self.datasets[dataset_info.name].columns else None),
            ("Perform dimensionality reduction", lambda: self.perform_dimensionality_reduction(dataset_info.name)),
            ("Perform clustering", lambda: self.perform_clustering(dataset_info.name)),
        ]
        
        results = {}
        for step_name, step_func in tqdm(steps, desc=f"Processing {dataset_info.name}"):
            try:
                result = step_func()
                if result is not None:
                    results[step_name] = result
                logging.info(f"Completed step: {step_name}")
            except Exception as e:
                logging.error(f"Error in step '{step_name}' for dataset {dataset_info.name}: {str(e)}")
        
        try:
            # Generate comprehensive report
            report = self.generate_report(dataset_info.name)
            report.update(results)
            report['improvement_suggestions'] = results.get("Generate improvement suggestions")
            self.save_report(report, f"{dataset_info.name}_comprehensive_report.json")
            logging.info(f"Pipeline completed for dataset: {dataset_info.name}")
        except Exception as e:
            logging.error(f"Error generating report for dataset {dataset_info.name}: {str(e)}")

    def generate_improvement_suggestions(self, dataset_name: str):
        # Implement your logic here
        pass

    def stream_rag_query(self, query: str, dataset_name: str):
        vector_store = self.vector_stores[dataset_name]
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        for chunk in rag_chain.stream({"query": query}):
            yield chunk

    def predict_multimodal(self, dataset_name: str, text: str, image_path: str):
        model = self.fine_tuned_models.get(dataset_name)
        if not model:
            raise ValueError(f"No fine-tuned model found for {dataset_name}")
        
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        encoded_text = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(encoded_text['input_ids'], encoded_text['attention_mask'], image)
        
        return torch.argmax(output, dim=1).item()

    def update_document_metadata(self, dataset_name: str, doc_id: int, new_metadata: Dict[str, Any]):
        vector_store = self.vector_stores.get(dataset_name)
        if not vector_store:
            raise ValueError(f"No vector store found for {dataset_name}")
        
        vector_store.update_document_metadata(doc_id, new_metadata)

    def perform_text_analysis(self, dataset_name: str):
        logging.info("Performing text analysis")
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.datasets[dataset_name]['text'])
        
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_output = lda.fit_transform(tfidf_matrix)
        
        topic_words = {}
        for topic, comp in enumerate(lda.components_):
            word_idx = np.argsort(comp)[::-1][:10]
            topic_words[topic] = [vectorizer.get_feature_names_out()[i] for i in word_idx]
        
        return {
            'top_words': dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0])),
            'topics': topic_words
        }

    def perform_sentiment_analysis(self, dataset_name: str):
        logging.info("Performing sentiment analysis")
        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity
        
        self.datasets[dataset_name]['sentiment_score'] = self.datasets[dataset_name]['text'].apply(get_sentiment)
        
        return {
            'average_sentiment': self.datasets[dataset_name]['sentiment_score'].mean(),
            'sentiment_distribution': self.datasets[dataset_name]['sentiment_score'].value_counts(normalize=True).to_dict()
        }

    def perform_network_analysis(self, dataset_name: str, source_col: str, target_col: str):
        logging.info(f"Performing network analysis for {source_col} and {target_col}")
        G = nx.from_pandas_edgelist(self.datasets[dataset_name], source_col, target_col)
        
        return {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'avg_degree': sum(dict(G.degree()).values()) / float(G.number_of_nodes()),
            'clustering_coefficient': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G)
        }

    def perform_association_rule_mining(self, dataset_name: str, min_support: float = 0.01, min_confidence: float = 0.5):
        logging.info("Performing association rule mining")
        # Assuming we have a transaction dataset
        transactions = self.datasets[dataset_name].groupby('user_id')['product_id'].apply(list).reset_index()
        encoded_transactions = transactions['product_id'].apply(lambda x: pd.Series([1 if i in x else 0 for i in self.datasets[dataset_name]['product_id'].unique()]))
        
        frequent_itemsets = apriori(encoded_transactions, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        return rules.to_dict()

    def perform_causal_inference(self, dataset_name: str, treatment: str, outcome: str, covariates: List[str]):
        logging.info(f"Performing causal inference for treatment: {treatment}, outcome: {outcome}")
        causal_model = CausalModel(
            Y=self.datasets[dataset_name][outcome].values,
            D=self.datasets[dataset_name][treatment].values,
            X=self.datasets[dataset_name][covariates].values
        )
        
        causal_model.est_via_ols()
        ate = causal_model.estimates['ate']
        
        return {
            'average_treatment_effect': ate.value,
            'ate_confidence_interval': (ate.conf_int[0], ate.conf_int[1])
        }

    def generate_data_dictionary(self, dataset_name: str):
        logging.info("Generating data dictionary")
        data_dict = {}
        for column in self.datasets[dataset_name].columns:
            data_dict[column] = {
                'type': str(self.datasets[dataset_name][column].dtype),
                'unique_values': self.datasets[dataset_name][column].nunique(),
                'missing_values': self.datasets[dataset_name][column].isnull().sum(),
                'description': ''  # This could be filled manually or using NLP techniques
            }
            if self.datasets[dataset_name][column].dtype in ['int64', 'float64']:
                data_dict[column].update({
                    'min': self.datasets[dataset_name][column].min(),
                    'max': self.datasets[dataset_name][column].max(),
                    'mean': self.datasets[dataset_name][column].mean(),
                    'median': self.datasets[dataset_name][column].median()
                })
        return data_dict

    def generate_improvement_suggestions(self, dataset_name: str):
        logging.info("Generating improvement suggestions")
        suggestions = []
        
        # Check for missing values
        missing_percentages = self.datasets[dataset_name].isnull().mean() * 100
        high_missing = missing_percentages[missing_percentages > 5].index.tolist()
        if high_missing:
            suggestions.append(f"Consider imputing or removing columns with high missing values: {', '.join(high_missing)}")
        
        # Check for high cardinality in categorical variables
        categorical_columns = self.datasets[dataset_name].select_dtypes(include=['object']).columns
        high_cardinality = [col for col in categorical_columns if self.datasets[dataset_name][col].nunique() / len(self.datasets[dataset_name]) > 0.1]
        if high_cardinality:
            suggestions.append(f"Consider encoding or grouping categories in high cardinality columns: {', '.join(high_cardinality)}")
        
        # Check for class imbalance
        if 'label' in self.datasets[dataset_name].columns:
            class_balance = self.datasets[dataset_name]['label'].value_counts(normalize=True)
            if class_balance.min() < 0.1:
                suggestions.append("Consider addressing class imbalance using techniques like oversampling, undersampling, or SMOTE")
        
        # Suggest feature engineering
        suggestions.append("Consider creating new features by combining existing ones or extracting information from text fields")
        
        return suggestions

    def evaluate_model_performance(self, dataset_name: str, X_test, y_test):
        logging.info("Evaluating model performance")
        predictions = self.fine_tuned_models[dataset_name].predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted'),
            'recall': recall_score(y_test, predictions, average='weighted'),
            'f1_score': f1_score(y_test, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }

    def generate_data_profile_report(self, dataset_name: str):
        logging.info("Generating data profile report")
        import pandas_profiling
        
        profile = pandas_profiling.ProfileReport(self.datasets[dataset_name], title="Data Profile Report", explorative=True)
        profile.to_file("data_profile_report.html")
        
        return "Data profile report generated: data_profile_report.html"

    def perform_data_drift_detection(self, dataset_name: str, reference_data: pd.DataFrame):
        logging.info("Performing data drift detection")
        cd = TabularDrift(
            p_val=.05, 
            x_ref=reference_data.values, 
            x_ref_preprocessed=False,
            preprocess_fn=None, 
            correction='bonferroni'
        )
        
        predictions = cd.predict(self.datasets[dataset_name].values)
        
        return {
            'drift_detected': predictions['data']['is_drift'],
            'p_val': predictions['data']['p_val']
        }

    def perform_multivariate_analysis(self):
        logging.info("Performing multivariate analysis")
        from scipy.stats import pearsonr
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        corr_matrix = self.dataset[numeric_cols].corr()
        
        # Generate heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        # Perform pairwise correlation tests
        pairwise_tests = {}
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                corr, p_value = pearsonr(self.dataset[col1], self.dataset[col2])
                pairwise_tests[f"{col1} vs {col2}"] = {'correlation': corr, 'p_value': p_value}
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'pairwise_tests': pairwise_tests,
            'heatmap_path': 'correlation_heatmap.png'
        }

    def perform_geospatial_analysis(self, location_column: str):
        logging.info(f"Performing geospatial analysis on column: {location_column}")
        import geopandas as gpd
        from shapely.geometry import Point
        
        # Assuming the location_column contains latitude and longitude
        gdf = gpd.GeoDataFrame(
            self.dataset, 
            geometry=gpd.points_from_xy(self.dataset[f'{location_column}_longitude'], self.dataset[f'{location_column}_latitude'])
        )
        
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # Plot the world map with data points
        fig, ax = plt.subplots(figsize=(15, 10))
        world.plot(ax=ax, alpha=0.4, color='grey')
        gdf.plot(ax=ax, markersize=1, color='red', alpha=0.1)
        plt.title('Geospatial Distribution of Data Points')
        plt.savefig('geospatial_distribution.png')
        plt.close()
        
        return {
            'total_points': len(gdf),
            'unique_locations': gdf.geometry.nunique(),
            'bounding_box': gdf.total_bounds.tolist(),
            'map_path': 'geospatial_distribution.png'
        }

    def perform_survival_analysis(self, duration_column: str, event_column: str):
        logging.info(f"Performing survival analysis on duration: {duration_column}, event: {event_column}")
        from lifelines import KaplanMeierFitter
        
        kmf = KaplanMeierFitter()
        kmf.fit(self.dataset[duration_column], self.dataset[event_column], label='Kaplan Meier Estimate')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        kmf.plot(ax=ax)
        plt.title('Kaplan-Meier Survival Curve')
        plt.savefig('survival_curve.png')
        plt.close()
        
        return {
            'median_survival_time': kmf.median_survival_time_,
            'survival_curve_path': 'survival_curve.png'
        }

    def perform_topic_modeling(self, text_column: str, num_topics: int = 5):
        logging.info(f"Performing topic modeling on column: {text_column}")
        from gensim import corpora
        from gensim.models import LdaMulticore
        from gensim.parsing.preprocessing import STOPWORDS
        from nltk.tokenize import word_tokenize
        import nltk
        nltk.download('punkt')
        
        def preprocess(text):
            tokens = word_tokenize(text.lower())
            return [token for token in tokens if token not in STOPWORDS and len(token) > 3]
        
        processed_docs = self.dataset[text_column].apply(preprocess)
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        
        topics = lda_model.print_topics(num_words=10)
        
        return {
            'topics': dict(topics),
            'coherence_score': lda_model.log_perplexity(corpus)
        }

    def perform_anomaly_detection(self, method='isolation_forest'):
        logging.info(f"Performing anomaly detection using method: {method}")
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        
        numeric_data = self.dataset.select_dtypes(include=[np.number])
        
        if method == 'isolation_forest':
            clf = IsolationForest(contamination=0.1, random_state=42)
            anomalies = clf.fit_predict(numeric_data)
        elif method == 'local_outlier_factor':
            clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            anomalies = clf.fit_predict(numeric_data)
        else:
            raise ValueError(f"Unsupported anomaly detection method: {method}")
        
        anomaly_indices = np.where(anomalies == -1)[0]
        return self.dataset.iloc[anomaly_indices]

    def semantic_search(self, query, dataset_name, k=5):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        if dataset_name not in self.vector_stores:
            self.build_vector_store(dataset_name)
        
        vector_store = self.vector_stores[dataset_name]
        results = vector_store.similarity_search_with_score(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            } for doc, score in results
        ]

    def rag_query(self, query, dataset_name):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        if dataset_name not in self.vector_stores:
            self.build_vector_store(dataset_name)
        
        vector_store = self.vector_stores[dataset_name]
        retriever = vector_store.as_retriever()
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain({"query": query})
        
        return {
            "answer": result['result'],
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in result['source_documents']
            ]
        }

    def perform_time_series_analysis(self, dataset_name, date_column, value_column):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        df = df.set_index(date_column)
        return df[value_column].resample('D').mean().to_dict()

    def perform_hypothesis_testing(self, dataset_name, group_column, value_column):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        groups = df[group_column].unique()
        if len(groups) != 2:
            raise ValueError("Hypothesis testing requires exactly two groups")
        group1 = df[df[group_column] == groups[0]][value_column]
        group2 = df[df[group_column] == groups[1]][value_column]
        t_stat, p_value = stats.ttest_ind(group1, group2)
        return {"t_statistic": t_stat, "p_value": p_value}

    def detect_anomalies(self, dataset_name, column=None):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        if column:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset")
            data = df[[column]]
        else:
            data = df.select_dtypes(include=[np.number])
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(data)
        return df[anomalies == -1]

def main():
    logging.info("Starting main function")
    platform = EnhancedDataQualityPlatform()
    
    base_path = "/Users/student/Desktop/FALL24/IDS560/bosch/llmproj/data/Amazon Sales 2023"
    
    # Load Appliances dataset
    appliances_info = DatasetInfo("appliances", f"{base_path}/All Appliances.csv", "csv")
    platform.load_dataset(appliances_info)
    
    # Load Electronics dataset
    electronics_info = DatasetInfo("electronics", f"{base_path}/All Electronics.csv", "csv")
    platform.load_dataset(electronics_info)
    
    # Perform operations on both datasets
    for dataset_name in ["appliances", "electronics"]:
        platform.assess_data_quality(dataset_name)
        platform.generate_metadata(dataset_name)
        platform.build_vector_store(dataset_name)
    
    # Example operations (you may need to adjust these based on your specific requirements)
    search_results = platform.semantic_search("high-quality appliances", k=5)
    logging.info(f"Semantic Search Results:\n{json.dumps(search_results, indent=2)}")
    
    rag_result = platform.rag_query("What are common features of top-selling electronics?", "electronics")
    logging.info(f"RAG Query Result:\n{rag_result}")
    
    # Update document metadata example (adjust doc_id as needed)
    platform.update_document_metadata("appliances", 0, {"verified_product": True})
    
    logging.info("Main function completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()
