from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X: pd.DataFrame, y: pd.Series, k: int = 10) -> List[str]:
    """Select the top k features using ANOVA F-value."""
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features

