# feature_engineering.py
import dask.dataframe as dd
import numpy as np
from dask_ml.preprocessing import PolynomialFeatures
from dask_ml.feature_selection import mutual_info_regression
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def auto_feature_engineering(df: dd.DataFrame, target_column: str) -> dd.DataFrame:
    try:
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns.remove(target_column)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(df[numeric_columns])
        poly_feature_names = poly.get_feature_names(numeric_columns)
        
        # Add polynomial features to the dataframe
        for i, name in enumerate(poly_feature_names):
            df[f'poly_{name}'] = poly_features[:, i]
        
        # Create interaction features
        for i in tqdm(range(len(numeric_columns)), desc="Creating interaction features"):
            for j in range(i+1, len(numeric_columns)):
                col1, col2 = numeric_columns[i], numeric_columns[j]
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        # Select top features based on mutual information
        mi_scores = mutual_info_regression(df.drop(columns=[target_column]), df[target_column])
        mi_scores = dd.from_array(mi_scores, columns=df.columns.drop(target_column))
        top_features = mi_scores.nlargest(50).index.compute().tolist()
        
        return df[top_features + [target_column]]
    except Exception as e:
        logger.error(f"Error in auto feature engineering: {str(e)}")
        raise
