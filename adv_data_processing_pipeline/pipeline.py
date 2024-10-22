from typing import Dict, List, Optional, Any, Callable
from .loading import load_data
from .cleaning import clean_data
from .transformation import transform_data
import logging
import time
from tqdm import tqdm
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, progress, get_client
from dask import delayed
import dask.bag as db
from text_analytics import perform_sentiment_analysis, summarize_text
from entity_recognition import extract_entities
from topic_modeling import perform_topic_modeling
from feature_selection import select_features
from dimensionality_reduction import reduce_dimensions
from data_validation import validate_data_schema
from feature_engineering import auto_feature_engineering
from imbalanced_data import handle_imbalanced_data

logger = logging.getLogger(__name__)

def apply_custom_transformations(df: dd.DataFrame, custom_funcs: List[Callable], pbar: tqdm) -> dd.DataFrame:
    """Apply custom transformation functions to the dataframe."""
    for func in custom_funcs:
        df = func(df)
        pbar.update(1 / len(custom_funcs))
    return df

def process_data(
    source: str,
    steps: List[str] = ['load', 'clean', 'transform'],
    cleaning_strategies: Optional[Dict[str, Dict[str, str]]] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_strategy: str = 'standard',
    encode_strategy: str = 'onehot',
    custom_transformations: Optional[List[Callable]] = None,
    chunk_size: Optional[int] = None,
    n_workers: int = 4,
    save_intermediate: bool = False,
    intermediate_path: str = './intermediate/',
    memory_limit: Optional[int] = None,
    **kwargs: Any
) -> dd.DataFrame:
    """
    Main function to load, clean, and transform data, optimized for large files.
    
    :param source: str, path to file or URL or SQL connection string
    :param steps: list of steps to perform in the pipeline
    :param cleaning_strategies: dict, cleaning strategies for each column
    :param numeric_features: list of numeric column names
    :param categorical_features: list of categorical column names
    :param scale_strategy: strategy for scaling numeric features
    :param encode_strategy: strategy for encoding categorical features
    :param custom_transformations: list of custom transformation functions
    :param chunk_size: size of chunks for processing large files
    :param n_workers: number of workers for parallel processing
    :param save_intermediate: whether to save intermediate results
    :param intermediate_path: path to save intermediate results
    :param memory_limit: memory limit for handling large datasets
    :param kwargs: Additional arguments for data loading
    :return: processed dask DataFrame
    """
    try:
        start_time = time.time()
        
        logger.info("Starting data processing pipeline")
        
        # Set up Dask client for parallel processing
        client = Client(n_workers=n_workers)
        logger.info(f"Dask client set up with {n_workers} workers")
        
        df = None
        with tqdm(total=len(steps), desc="Processing Data") as pbar:
            # Group steps that can be executed in parallel
            parallel_steps = [['load'], ['clean', 'transform'], ['custom']]
            
            for step_group in parallel_steps:
                step_results = []
                for step in step_group:
                    if step in steps:
                        try:
                            if step == 'load':
                                pbar.set_description("Loading data")
                                df = delayed(load_data)(source, chunk_size=chunk_size, **kwargs)
                            elif step == 'clean' and df is not None:
                                pbar.set_description("Cleaning data")
                                df = delayed(clean_data)(df, cleaning_strategies)
                            elif step == 'transform' and df is not None:
                                pbar.set_description("Transforming data")
                                df = delayed(transform_data)(df, numeric_features, categorical_features, scale_strategy, encode_strategy)
                            elif step == 'custom' and df is not None and custom_transformations:
                                pbar.set_description("Applying custom transformations")
                                df = delayed(apply_custom_transformations)(df, custom_transformations, pbar)
                            
                            step_results.append(df)
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error in {step} step: {str(e)}")
                            raise
                
                # Compute and persist the results of parallel steps
                step_results = client.compute(step_results)
                df = step_results[-1]  # Use the result of the last step
                
                # Handle large datasets
                if memory_limit and df.memory_usage().sum().compute() > memory_limit:
                    df = df.to_bag().repartition(npartitions=n_workers)
                
                # Save intermediate result if specified
                if save_intermediate:
                    intermediate_file = f"{intermediate_path}{step}_result.parquet"
                    df.to_parquet(intermediate_file)
                    logger.info(f"Intermediate result saved to {intermediate_file}")
                
                logger.info(f"Step group completed. Current shape: {df.shape.compute()}")
        
        # Compute the final result
        df = df.compute()
        
        end_time = time.time()
        logger.info(f"Data processing completed in {end_time - start_time:.2f} seconds")
        
        return df
    
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {str(e)}")
        raise
    finally:
        if 'client' in locals():
            client.close()

