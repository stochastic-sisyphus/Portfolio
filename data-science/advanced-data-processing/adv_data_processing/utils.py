import yaml
from typing import Dict, Any, List, Callable
import json
import logging
from itertools import islice
import importlib
import os
import dask
import dask.dataframe as dd
from dask.distributed import Client
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to a YAML file."""
    with open(config_path, 'w') as config_file:
        yaml.dump(config, config_file)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(file_path, 'r') as json_file:
        return json.load(json_file)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save configuration to a JSON file."""
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate the configuration dictionary."""
    required_keys = ['source', 'steps', 'file_type', 'text_column', 'model_type', 'output_file']
    return all(key in config for key in required_keys)

def get_pipeline_steps(config: Dict[str, Any]) -> List[str]:
    """Get the list of pipeline steps from the configuration."""
    return config.get('steps', [])

def chunk_generator(file_path: str, chunk_size: int = 1000):
    """Generator function to read large files in chunks."""
    with open(file_path, 'r') as file:
        while True:
            chunk = list(islice(file, chunk_size))
            if not chunk:
                break
            yield chunk

def log_step(step: str, success: bool):
    """Log the status of a pipeline step."""
    status = "completed successfully" if success else "failed"
    logger.info(f"Step '{step}' {status}")

def get_model(model_type: str) -> Any:
    """Get the appropriate model based on the model type."""
    if model_type == 'linear_regression':
        return LinearRegression()
    elif model_type == 'random_forest':
        return RandomForestClassifier()
    # Add more model types as needed
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def load_custom_function(function_path: str) -> Callable:
    """Load a custom function from a given path."""
    try:
        module_name, function_name = function_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading custom function {function_path}: {str(e)}")
        raise

def validate_data(df, schema: Dict[str, str]) -> bool:
    """Validate the dataframe against a given schema."""
    for column, dtype in schema.items():
        if column not in df.columns or df[column].dtype != dtype:
            return False
    return True

def generate_summary_statistics(df):
    """Generate summary statistics for the dataframe."""
    return df.describe()

def load_custom_plugins(plugin_paths):
    plugins = {}
    for path in plugin_paths:
        module_name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plugins[module_name] = module.run
    return plugins

def distributed_cache_result(step: str, data: dd.DataFrame, client: Client):
    """Cache the result of a step in distributed memory."""
    future = client.scatter(data, broadcast=True)
    dask.config.set({'distributed.worker.memory.target': 0.95})
    return future

def load_distributed_cached_result(step: str, client: Client):
    """Load a cached result from distributed memory."""
    try:
        return client.get_dataset(step)
    except KeyError:
        return None

def save_pipeline_state(state: Dict[str, Any], file_path: str) -> None:
    """Save the current state of the pipeline."""
    with open(file_path, 'wb') as f:
        joblib.dump(state, f)

def load_pipeline_state(file_path: str) -> Dict[str, Any]:
    """Load a saved pipeline state."""
    with open(file_path, 'rb') as f:
        return joblib.load(f)

