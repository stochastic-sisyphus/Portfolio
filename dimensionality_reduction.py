from sklearn.decomposition import PCA

def reduce_dimensions(X: pd.DataFrame, n_components: int) -> pd.DataFrame:
    """Reduce the dimensionality of the dataset using PCA."""
    pca = PCA(n_components=n_components)
    reduced_X = pca.fit_transform(X)
    return pd.DataFrame(reduced_X, columns=[f'PC{i+1}' for i in range(n_components)], index=X.index)

