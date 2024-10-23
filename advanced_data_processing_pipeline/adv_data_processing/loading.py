import pandas as pd
from sqlalchemy import create_engine
import logging
import json
from bs4 import BeautifulSoup
import dask.dataframe as dd
from typing import Dict, Any
import requests
import boto3

logger = logging.getLogger(__name__)

def load_data(source: str, chunk_size: int = None, **kwargs) -> dd.DataFrame:
    """
    Load data from various sources and formats, optimized for large files.
    
    :param source: str, path to file or URL or SQL connection string
    :param chunk_size: int, size of chunks for processing large files
    :param kwargs: Additional arguments for specific loaders
    :return: dask DataFrame
    """
    try:
        if isinstance(source, str):
            if source.endswith('.csv'):
                return load_from_csv_chunked(source, chunk_size=chunk_size, **kwargs)
            elif source.endswith('.xlsx'):
                return load_from_excel(source, **kwargs)
            elif source.endswith('.json'):
                return load_from_json_chunked(source, chunk_size=chunk_size, **kwargs)
            elif source.endswith('.parquet'):
                return load_from_parquet(source, **kwargs)
            elif source.endswith('.html'):
                return load_from_html(source, **kwargs)
            elif source.startswith(('http://', 'https://')):
                return load_from_url(source, chunk_size=chunk_size, **kwargs)
            elif 'sql' in source.lower():
                return load_from_sql_chunked(source, kwargs.get('query'), chunk_size=chunk_size, **kwargs)
            elif source.startswith('s3://'):
                return load_from_s3(source, chunk_size=chunk_size, **kwargs)
        else:
            raise ValueError("Unsupported data source or format")
    except Exception as e:
        logger.error(f"Error loading data from {source}: {str(e)}")
        raise

def load_from_csv(file_path: str, chunk_size: int = None, **kwargs) -> dd.DataFrame:
    """Load data from a CSV file."""
    try:
        return dd.read_csv(file_path, blocksize=chunk_size, **kwargs)
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise

def load_from_excel(file_path: str, **kwargs) -> dd.DataFrame:
    """Load data from an Excel file."""
    try:
        return dd.from_pandas(pd.read_excel(file_path, **kwargs), npartitions=kwargs.get('npartitions', 1))
    except Exception as e:
        logger.error(f"Error loading Excel file: {str(e)}")
        raise

def load_from_json(file_path: str, chunk_size: int = None, **kwargs) -> dd.DataFrame:
    """Load data from a JSON file."""
    try:
        return dd.read_json(file_path, blocksize=chunk_size, **kwargs)
    except Exception as e:
        logger.error(f"Error loading JSON file: {str(e)}")
        raise

def load_from_parquet(file_path: str, **kwargs) -> dd.DataFrame:
    """Load data from a Parquet file."""
    try:
        return dd.read_parquet(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error loading Parquet file: {str(e)}")
        raise

def load_from_html(source: str, **kwargs) -> dd.DataFrame:
    """Load data from HTML file or URL."""
    try:
        with open(source, 'r') as f:
            soup = BeautifulSoup(f, 'html.parser')
        
        data = []
        table = soup.find('table')
        for row in table.find_all('tr'):
            cols = row.find_all('td')
            if cols:
                data.append([col.text.strip() for col in cols])
        
        return dd.from_pandas(pd.DataFrame(data, columns=kwargs.get('columns', None)), npartitions=kwargs.get('npartitions', 1))
    except Exception as e:
        logger.error(f"Error loading HTML file: {str(e)}")
        raise

def load_from_url(url: str, chunk_size: int = None, **kwargs) -> dd.DataFrame:
    """Load data from a URL."""
    try:
        return dd.read_csv(url, blocksize=chunk_size, **kwargs)
    except Exception as e:
        logger.error(f"Error loading data from URL: {str(e)}")
        raise

def load_from_sql(connection_string: str, **kwargs) -> dd.DataFrame:
    """Load data from a SQL database."""
    try:
        engine = create_engine(connection_string)
        return dd.read_sql_table(kwargs.get('table_name'), engine, index_col=kwargs.get('index_col'))
    except Exception as e:
        logger.error(f"Error loading data from SQL: {str(e)}")
        raise

def load_from_s3(s3_path: str, chunk_size: int = None, **kwargs) -> dd.DataFrame:
    """Load data from an S3 bucket."""
    try:
        return dd.read_csv(s3_path, blocksize=chunk_size, storage_options={'anon': False}, **kwargs)
    except Exception as e:
        logger.error(f"Error loading data from S3: {str(e)}")
        raise

def load_from_api(url: str, params: Dict[str, Any], **kwargs) -> dd.DataFrame:
    """Load data from an API endpoint."""
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return dd.from_pandas(pd.DataFrame(data), npartitions=kwargs.get('npartitions', 1))
    except Exception as e:
        logger.error(f"Error loading data from API: {str(e)}")
        raise

def load_from_csv_chunked(file_path: str, chunk_size: int = 100000, **kwargs) -> dd.DataFrame:
    """Load data from a CSV file in chunks."""
    try:
        return dd.read_csv(file_path, blocksize=chunk_size, **kwargs)
    except Exception as e:
        logger.error(f"Error loading CSV file in chunks: {str(e)}")
        raise

def load_from_json_chunked(file_path: str, chunk_size: int = 100000, **kwargs) -> dd.DataFrame:
    """Load data from a JSON file in chunks."""
    try:
        return dd.read_json(file_path, blocksize=chunk_size, **kwargs)
    except Exception as e:
        logger.error(f"Error loading JSON file in chunks: {str(e)}")
        raise

def load_from_sql_chunked(connection_string: str, query: str, chunk_size: int = 100000, **kwargs) -> dd.DataFrame:
    """Load data from a SQL database in chunks."""
    try:
        engine = create_engine(connection_string)
        return dd.read_sql_query(query, engine, index_col=kwargs.get('index_col'), chunksize=chunk_size)
    except Exception as e:
        logger.error(f"Error loading data from SQL in chunks: {str(e)}")
        raise
