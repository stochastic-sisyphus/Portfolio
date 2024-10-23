import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
import logging
import dask.dataframe as dd
from dask_ml.impute import SimpleImputer

logger = logging.getLogger(__name__)

def validate_data(df: dd.DataFrame, schema: Dict[str, Any]) -> bool:
    """Validate the dataframe against the given schema."""
    try:
        for column, rules in schema.items():
            if 'type' in rules:
                if not df[column].dtype == rules['type']:
                    logger.error(f"Column {column} has incorrect type. Expected {rules['type']}, got {df[column].dtype}")
                    return False
            if 'range' in rules:
                min_val, max_val = rules['range']
                if df[column].min().compute() < min_val or df[column].max().compute() > max_val:
                    logger.error(f"Column {column} has values outside the expected range [{min_val}, {max_val}]")
                    return False
        return True
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}")
        return False

def clean_data(df: dd.DataFrame, config: Dict[str, Any]) -> dd.DataFrame:
    """Clean the dataframe based on the provided configuration."""
    try:
        if not validate_data(df, config.get('schema', {})):
            raise ValueError("Data validation failed")
        
        df = handle_missing_values(df, config.get('missing_values', {}))
        df = handle_outliers(df, config.get('outlier_columns', []))
        df = remove_duplicates(df)
        df = convert_datatypes(df, config.get('dtype_conversions', {}))
        
        return df
    except Exception as e:
        logger.error(f"Error in data cleaning: {str(e)}")
        raise

def handle_missing_values(df: dd.DataFrame, column: str, strategy: Dict[str, str]) -> dd.DataFrame:
    """Handle missing values based on the provided strategy."""
    try:
        if df[column].isnull().any().compute():
            if strategy.get('missing') == 'drop':
                df = df.dropna(subset=[column])
            elif strategy.get('missing') == 'mean':
                df[column] = df[column].fillna(df[column].mean().compute())
            elif strategy.get('missing') == 'median':
                df[column] = df[column].fillna(df[column].median().compute())
            elif strategy.get('missing') == 'mode':
                df[column] = df[column].fillna(df[column].mode().compute().iloc[0])
            elif strategy.get('missing') == 'constant':
                df[column] = df[column].fillna(strategy.get('constant_value', 0))
        return df
    except Exception as e:
        logger.error(f"Error handling missing values for column {column}: {str(e)}")
        raise

def handle_outliers(df: dd.DataFrame, columns: List[str], method: str = 'iqr') -> dd.DataFrame:
    """Handle outliers using the specified method."""
    for column in columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower_bound, upper_bound)
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            df[column] = df[column].clip(mean - 3*std, mean + 3*std)
    return df

def remove_duplicates(df: dd.DataFrame) -> dd.DataFrame:
    """Remove duplicate rows from the dataframe."""
    try:
        return df.drop_duplicates()
    except Exception as e:
        logger.error(f"Error removing duplicates: {str(e)}")
        raise

def handle_outliers_zscore(df: dd.DataFrame, columns: List[str], threshold: float = 3) -> dd.DataFrame:
    """Handle outliers using the Z-score method."""
    try:
        for col in columns:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            df[col] = df[col].where(z_scores.abs() <= threshold, df[col].mean())
        return df
    except Exception as e:
        logger.error(f"Error handling outliers using Z-score method: {str(e)}")
        raise

def convert_datatypes(df: dd.DataFrame, dtype_dict: Dict[str, str]) -> dd.DataFrame:
    """Convert column datatypes based on the provided dictionary."""
    try:
        for col, dtype in dtype_dict.items():
            df[col] = df[col].astype(dtype)
        return df
    except Exception as e:
        logger.error(f"Error converting datatypes: {str(e)}")
        raise

def impute_missing_values(df: dd.DataFrame, strategies: Dict[str, str]) -> dd.DataFrame:
    """Impute missing values using specified strategies."""
    try:
        for col, strategy in strategies.items():
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
            elif strategy == 'constant':
                df[col] = df[col].fillna(0)  # You can change 0 to any other constant value
        return df
    except Exception as e:
        logger.error(f"Error imputing missing values: {str(e)}")
        raise

