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

def load_data(source: str, **kwargs) -> pd.DataFrame:
    """
    Load data from various sources and formats, optimized for large files.
    
    :param source: str, path to file or URL or SQL connection string
    :param kwargs: Additional arguments for specific loaders
    :return: pandas DataFrame or dask DataFrame for large files
    """
    try:
        if isinstance(source, str):
            if source.endswith('.csv'):
                return dd.read_csv(source, blocksize=chunk_size, **kwargs) if chunk_size else dd.read_csv(source, **kwargs)
            elif source.endswith('.xlsx'):
                return dd.read_excel(source, **kwargs)
            elif source.endswith('.json'):
                return dd.read_json(source, **kwargs)
            elif source.endswith('.parquet'):
                return dd.read_parquet(source, **kwargs)
            elif source.endswith('.html'):
                return load_html(source, **kwargs)
            elif source.startswith(('http://', 'https://')):
                return dd.read_csv(source, **kwargs)
            elif 'sql' in source.lower():
                engine = create_engine(source)
                return dd.read_sql_table(kwargs.get('table_name'), engine)
        else:
            raise ValueError("Unsupported data source or format")
    except Exception as e:
        logger.error(f"Error loading data from {source}: {str(e)}")
        raise

def load_html(source: str, **kwargs) -> pd.DataFrame:
    """
    Load data from HTML file or URL.
    
    :param source: str, path to HTML file or URL
    :param kwargs: Additional arguments for HTML parsing
    :return: pandas DataFrame
    """
    with open(source, 'r') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    # Extract data from HTML (this is a simple example, adjust as needed)
    data = []
    table = soup.find('table')
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        if cols:
            data.append([col.text.strip() for col in cols])
    
    return pd.DataFrame(data, columns=kwargs.get('columns', None))

def load_from_s3(bucket: str, key: str, **kwargs) -> dd.DataFrame:
    """Load data from an S3 bucket."""
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        return dd.read_csv(obj['Body'], **kwargs)
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

def load_from_csv(file_path: str, **kwargs) -> dd.DataFrame:
    """Load data from a CSV file."""
    try:
        return dd.read_csv(file_path, **kwargs)
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise

# Add more specific loading functions as needed, e.g., load_from_s3, load_from_api, etc.

