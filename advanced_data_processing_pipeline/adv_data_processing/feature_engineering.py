# feature_engineering.py
import dask.dataframe as dd
import numpy as np
from dask_ml.preprocessing import PolynomialFeatures
from dask_ml import feature_selection
import logging
from tqdm import tqdm
from typing import Dict, Any, List
from sklearn.feature_selection import mutual_info_regression
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def auto_feature_engineering(df: dd.DataFrame, target_column: str, config: Dict[str, Any]) -> dd.DataFrame:
    try:
        if config.get('create_polynomial_features', False):
            df = create_polynomial_features(df, config.get('polynomial_degree', 2))
        
        if config.get('create_interaction_features', False):
            df = create_interaction_features(df)
        
        if config.get('create_time_features', False):
            df = create_time_features(df, config.get('time_column'))
        
        if config.get('create_text_features', False):
            df = create_text_features(df, config.get('text_columns', []))
        
        if config.get('select_top_features', False):
            df = select_top_features(df, target_column, config.get('n_top_features', 50))
        
        if config.get('extract_html_features', False):
            df = extract_html_features(df, config.get('html_column'))
        
        return df
    except Exception as e:
        logger.error(f"Error in auto feature engineering: {str(e)}")
        raise

def create_polynomial_features(df: dd.DataFrame, degree: int = 2) -> dd.DataFrame:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[numeric_columns])
    poly_feature_names = poly.get_feature_names(numeric_columns)
    
    for i, name in enumerate(poly_feature_names):
        df[f'poly_{name}'] = poly_features[:, i]
    
    return df

def create_interaction_features(df: dd.DataFrame) -> dd.DataFrame:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):
            col1, col2 = numeric_columns[i], numeric_columns[j]
            df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
    return df

def create_time_features(df: dd.DataFrame, time_column: str) -> dd.DataFrame:
    df[time_column] = dd.to_datetime(df[time_column])
    df['year'] = df[time_column].dt.year
    df['month'] = df[time_column].dt.month
    df['day'] = df[time_column].dt.day
    df['day_of_week'] = df[time_column].dt.dayofweek
    df['quarter'] = df[time_column].dt.quarter
    return df

def create_text_features(df: dd.DataFrame, text_columns: List[str]) -> dd.DataFrame:
    for column in text_columns:
        df[f'{column}_length'] = df[column].str.len()
        df[f'{column}_word_count'] = df[column].str.split().str.len()
    return df

def select_top_features(df: dd.DataFrame, target_column: str, n_top_features: int = 50) -> dd.DataFrame:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    mi_scores = mutual_info_regression(X, y)
    mi_scores = dd.from_array(mi_scores, columns=X.columns)
    top_features = mi_scores.nlargest(n_top_features).index.compute().tolist()
    return df[top_features + [target_column]]

def extract_html_features(df: dd.DataFrame, html_column: str) -> dd.DataFrame:
    def extract_features(html):
        soup = BeautifulSoup(html, 'html.parser')
        return {
            'title_length': len(soup.title.string) if soup.title else 0,
            'num_paragraphs': len(soup.find_all('p')),
            'num_links': len(soup.find_all('a')),
            'num_images': len(soup.find_all('img'))
        }
    
    features = df[html_column].apply(extract_features, meta=('html', 'object'))
    return df.assign(**features)
