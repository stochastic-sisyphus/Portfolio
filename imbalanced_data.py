# imbalanced_data.py
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from typing import Literal
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def handle_imbalanced_data(df: pd.DataFrame, target_column: str, method: Literal['smote', 'undersampling', 'smoteenn'] = 'smote') -> pd.DataFrame:
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if method == 'smote':
            sampler = SMOTE()
        elif method == 'undersampling':
            sampler = RandomUnderSampler()
        elif method == 'smoteenn':
            sampler = SMOTETomek()
        else:
            raise ValueError(f"Unsupported imbalance handling method: {method}")
        
        logger.info(f"Handling imbalanced data using {method} method")
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(f"Resampled data shape: {X_resampled.shape}")
        return pd.concat([X_resampled, y_resampled], axis=1)
    except Exception as e:
        logger.error(f"Error handling imbalanced data: {str(e)}")
        raise
