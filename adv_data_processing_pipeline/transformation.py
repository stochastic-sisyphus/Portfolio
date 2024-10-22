import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from typing import List, Optional
import logging
import dask.dataframe as dd
from dask_ml.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PolynomialFeatures
from dask_ml.compose import ColumnTransformer
from dask_ml.feature_extraction.text import HashingVectorizer
from dask_ml.feature_extraction import DictVectorizer
from dask_ml.feature_selection import VarianceThreshold

logger = logging.getLogger(__name__)

def transform_data(
    df: dd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_strategy: str = 'standard',
    encode_strategy: str = 'onehot'
) -> dd.DataFrame:
    """
    Transform the dataframe by scaling numeric features and encoding categorical features.
    
    :param df: dask DataFrame
    :param numeric_features: list of numeric column names
    :param categorical_features: list of categorical column names
    :param scale_strategy: 'standard' or 'minmax' scaling for numeric features
    :param encode_strategy: 'onehot' or 'label' encoding for categorical features
    :return: transformed dask DataFrame
    """
    try:
        numeric_features = numeric_features or df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = categorical_features or df.select_dtypes(include=['object', 'category']).columns.tolist()

        scaler = get_scaler(scale_strategy)
        encoder = get_encoder(encode_strategy)
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', scaler, numeric_features),
                ('cat', encoder, categorical_features)
            ])

        transformed_df = preprocessor.fit_transform(df)
        
        feature_names = numeric_features + get_encoded_feature_names(encoder, categorical_features)
        
        return dd.from_array(transformed_df, columns=feature_names)
    
    except Exception as e:
        logger.error(f"Error during data transformation: {str(e)}")
        raise

def get_scaler(strategy: str):
    """Get the appropriate scaler based on the strategy."""
    if strategy == 'standard':
        return StandardScaler()
    elif strategy == 'minmax':
        return MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling strategy: {strategy}")

def get_encoder(strategy: str):
    """Get the appropriate encoder based on the strategy."""
    if strategy == 'onehot':
        return OneHotEncoder(sparse=False)
    elif strategy == 'label':
        return LabelEncoder()
    else:
        raise ValueError(f"Unsupported encoding strategy: {strategy}")

def get_encoded_feature_names(encoder, categorical_features):
    """Get feature names after encoding."""
    if isinstance(encoder, OneHotEncoder):
        return [f"{feature}_{category}" for feature, categories in zip(categorical_features, encoder.categories_) for category in categories]
    else:
        return categorical_features

def add_polynomial_features(df: dd.DataFrame, degree: int = 2, interaction_only: bool = False) -> dd.DataFrame:
    """Add polynomial features to the dataframe."""
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    poly_features = poly.fit_transform(df[numeric_features])
    feature_names = poly.get_feature_names(numeric_features)
    poly_df = dd.from_array(poly_features, columns=feature_names)
    return dd.concat([df, poly_df], axis=1)

def bin_continuous_variable(df: dd.DataFrame, column: str, bins: int, labels: Optional[List[str]] = None) -> dd.DataFrame:
    """Bin a continuous variable into categories."""
    df[f'{column}_binned'] = dd.cut(df[column], bins=bins, labels=labels)
    return df

def text_vectorization(df: dd.DataFrame, text_column: str, method: str = 'hashing', **kwargs) -> dd.DataFrame:
    """Vectorize text data using either hashing or dictionary vectorization."""
    if method == 'hashing':
        vectorizer = HashingVectorizer(**kwargs)
    elif method == 'dict':
        vectorizer = DictVectorizer(**kwargs)
    else:
        raise ValueError(f"Unsupported vectorization method: {method}")
    
    text_features = vectorizer.fit_transform(df[text_column])
    feature_names = [f'text_feature_{i}' for i in range(text_features.shape[1])]
    text_df = dd.from_array(text_features, columns=feature_names)
    return dd.concat([df, text_df], axis=1)

def create_interaction_features(df: dd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> dd.DataFrame:
    """Create interaction features between specified pairs of features."""
    for f1, f2 in feature_pairs:
        df[f'{f1}_{f2}_interaction'] = df[f1] * df[f2]
    return df

def select_features_variance(df: dd.DataFrame, threshold: float = 0.0) -> dd.DataFrame:
    """Select features based on their variance."""
    selector = VarianceThreshold(threshold=threshold)
    selected_features = selector.fit_transform(df)
    return dd.from_array(selected_features, columns=df.columns[selector.get_support()])

