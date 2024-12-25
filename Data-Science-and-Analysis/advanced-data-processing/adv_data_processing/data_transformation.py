import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def transform_data(df: dd.DataFrame, config: Dict[str, Any]) -> dd.DataFrame:
    """
    Transform the data based on the provided configuration.
    """
    try:
        df = transform_numerical_features(df, config.get('numerical_features', []), config.get('scaling_method', 'standard'))
        df = transform_categorical_features(df, config.get('categorical_features', []), config.get('encoding_method', 'onehot'))
        df = transform_text_features(df, config.get('text_features', []), config.get('text_vectorization_method', 'tfidf'))
        return df
    except Exception as e:
        logger.error(f"Error in data transformation: {str(e)}")
        raise

def transform_numerical_features(df: dd.DataFrame, numerical_features: List[str], method: str = 'standard') -> dd.DataFrame:
    """Transform numerical features using the specified scaling method."""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")
    
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    return df

def transform_categorical_features(df: dd.DataFrame, categorical_features: List[str], method: str = 'onehot') -> dd.DataFrame:
    """Transform categorical features using the specified encoding method."""
    if method == 'onehot':
        encoder = OneHotEncoder(sparse=False)
        encoded = encoder.fit_transform(df[categorical_features])
        encoded_df = dd.from_array(encoded, columns=encoder.get_feature_names(categorical_features))
        df = dd.concat([df.drop(columns=categorical_features), encoded_df], axis=1)
    elif method == 'label':
        for feature in categorical_features:
            encoder = LabelEncoder()
            df[feature] = encoder.fit_transform(df[feature])
    else:
        raise ValueError(f"Unsupported encoding method: {method}")
    
    return df

def transform_text_features(df: dd.DataFrame, text_features: List[str], method: str = 'tfidf') -> dd.DataFrame:
    """Transform text features using the specified vectorization method."""
    if method == 'tfidf':
        vectorizer = TfidfVectorizer()
        for feature in text_features:
            text_data = df[feature].compute()  # Compute to bring data to memory
            vectorized = vectorizer.fit_transform(text_data)
            feature_names = vectorizer.get_feature_names()
            vectorized_df = dd.from_array(vectorized.toarray(), columns=[f'{feature}_{name}' for name in feature_names])
            df = dd.concat([df.drop(columns=[feature]), vectorized_df], axis=1)
    else:
        raise ValueError(f"Unsupported text vectorization method: {method}")
    
    return df

