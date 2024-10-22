from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

def evaluate_model(y_true: np.array, y_pred: np.array, task: str = 'classification') -> Dict[str, float]:
    """Evaluate the model based on the task (classification or regression)."""
    if task == 'classification':
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    elif task == 'regression':
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2_score': r2_score(y_true, y_pred)
        }
    else:
        raise ValueError(f"Unsupported task: {task}")

