import unittest
import pandas as pd
import dask.dataframe as dd
from data_preprocessing import preprocess_data
from cleaning import clean_data
from feature_engineering import auto_feature_engineering
from data_transformation import transform_data
from loading import load_data

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset for testing
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 5.5],
            'D': [True, False, True, False, True]
        })
        self.ddf = dd.from_pandas(self.df, npartitions=2)

    def test_load_data(self):
        # Test load_data function (assuming we're loading from a CSV)
        self.df.to_csv('test_data.csv', index=False)
        loaded_data = load_data('test_data.csv')
        self.assertIsInstance(loaded_data, dd.DataFrame)
        self.assertEqual(loaded_data.shape[0].compute(), 5)
        self.assertEqual(loaded_data.shape[1], 4)

    def test_preprocess_data(self):
        config = {
            'missing_values': {'A': 'mean'},
            'categorical_columns': ['B'],
            'numerical_columns': ['A', 'C']
        }
        preprocessed_data = preprocess_data(self.ddf, config)
        self.assertIsInstance(preprocessed_data, dd.DataFrame)
        self.assertGreater(preprocessed_data.shape[1], self.ddf.shape[1])  # Due to one-hot encoding

    def test_clean_data(self):
        config = {
            'schema': {'A': {'type': 'int64', 'range': [1, 10]}},
            'missing_values': {'A': 'mean'},
            'outlier_columns': ['C']
        }
        cleaned_data = clean_data(self.ddf, config)
        self.assertIsInstance(cleaned_data, dd.DataFrame)
        self.assertEqual(cleaned_data.shape, self.ddf.shape)

    def test_auto_feature_engineering(self):
        config = {
            'create_polynomial_features': True,
            'create_interaction_features': True
        }
        engineered_data = auto_feature_engineering(self.ddf, 'D', config)
        self.assertIsInstance(engineered_data, dd.DataFrame)
        self.assertGreater(engineered_data.shape[1], self.ddf.shape[1])

    def test_transform_data(self):
        config = {
            'numerical_features': ['A', 'C'],
            'categorical_features': ['B'],
            'scaling_method': 'standard',
            'encoding_method': 'onehot'
        }
        transformed_data = transform_data(self.ddf, config)
        self.assertIsInstance(transformed_data, dd.DataFrame)
        self.assertGreater(transformed_data.shape[1], self.ddf.shape[1])

if __name__ == '__main__':
    unittest.main()
