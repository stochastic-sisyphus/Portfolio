import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from pathlib import Path
import yaml
from pipeline import process_data
import dask.dataframe as dd
import argparse
from utils import (
    load_config, validate_config, get_pipeline_steps, log_step, get_model, 
    save_pipeline_state, load_pipeline_state, validate_data, generate_summary_statistics,
    load_custom_plugins, cache_result, load_cached_result
)
from visualization import visualize_data, plot_entity_distribution, generate_word_cloud
from text_analytics import perform_sentiment_analysis, summarize_text
from entity_recognition import extract_entities
from topic_modeling import perform_topic_modeling
from feature_selection import select_features
from dimensionality_reduction import reduce_dimensions
from model_evaluation import evaluate_model
from data_validation import validate_data_schema
from error_handling import handle_error
from dask.distributed import Client, progress
from tqdm import tqdm
import joblib
import os
from feature_engineering import auto_feature_engineering
from imbalanced_data import handle_imbalanced_data
from dask_ml.model_selection import GridSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Advanced Data Processing Pipeline")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--output', type=str, help='Output file path (overrides config file)')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--analyze_text', action='store_true', help='Perform text analytics')
    parser.add_argument('--extract_entities', action='store_true', help='Perform named entity recognition')
    parser.add_argument('--model_topics', action='store_true', help='Perform topic modeling')
    parser.add_argument('--resume', type=str, help='Resume from a saved pipeline state')
    parser.add_argument('--select_features', action='store_true', help='Perform feature selection')
    parser.add_argument('--reduce_dimensions', action='store_true', help='Perform dimensionality reduction')
    parser.add_argument('--validate_schema', action='store_true', help='Validate data schema')
    parser.add_argument('--summary_stats', action='store_true', help='Generate summary statistics')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate results')
    parser.add_argument('--intermediate_path', type=str, default='./intermediate/', help='Path to save intermediate results')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for parallel processing')
    parser.add_argument('--scheduler_address', type=str, help='Address of the Dask scheduler')
    parser.add_argument('--plugins', type=str, nargs='+', help='Custom plugins to load')
    parser.add_argument('--generate_report', action='store_true', help='Generate a comprehensive report')
    parser.add_argument('--use_cache', action='store_true', help='Use cached results if available')
    parser.add_argument('--auto_feature_engineering', action='store_true', help='Perform automatic feature engineering')
    parser.add_argument('--handle_imbalanced', action='store_true', help='Handle imbalanced datasets')
    parser.add_argument('--auto_tune', action='store_true', help='Perform automatic hyperparameter tuning')
    parser.add_argument('--memory_limit', type=int, help='Set memory limit for Dask workers')
    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    try:
        # Load and validate configuration
        config = load_config(args.config)
        if not validate_config(config):
            raise ValueError("Invalid configuration file")

        # Set up Dask client
        if args.scheduler_address:
            client = Client(args.scheduler_address)
        else:
            client = Client(n_workers=args.n_workers)
        logger.info(f"Dask client set up with {client.ncores} cores")

        # Load custom plugins
        custom_plugins = load_custom_plugins(args.plugins) if args.plugins else {}

        # Resume from saved state if specified
        if args.resume:
            pipeline_state = load_pipeline_state(args.resume)
            processed_data = pipeline_state['data']
            completed_steps = pipeline_state['completed_steps']
            logger.info(f"Resumed from state: {args.resume}")
        else:
            processed_data = None
            completed_steps = []

        # Get pipeline steps
        steps = get_pipeline_steps(config)

        # Process data using the configuration
        with tqdm(total=len(steps), desc="Processing Pipeline") as pbar:
            for step in steps:
                if step not in completed_steps:
                    # Check if cached result is available
                    if args.use_cache:
                        cached_result = load_cached_result(step)
                        if cached_result is not None:
                            processed_data = cached_result
                            logger.info(f"Loaded cached result for step: {step}")
                            continue
                    log_step(step)
                    if step == 'load':
                        processed_data = process_data(config['source'], steps=['load'], **config.get('load_options', {}))
                    elif step == 'clean':
                        processed_data = process_data(processed_data, steps=['clean'], cleaning_strategies=config.get('cleaning_strategies'))
                    elif step == 'transform':
                        processed_data = process_data(processed_data, steps=['transform'], 
                                                      numeric_features=config.get('numeric_features'),
                                                      categorical_features=config.get('categorical_features'),
                                                      scale_strategy=config.get('scale_strategy', 'standard'),
                                                      encode_strategy=config.get('encode_strategy', 'onehot'))
                    elif step == 'validate_schema' and args.validate_schema:
                        schema_valid = validate_data_schema(processed_data, config['data_schema'])
                        logger.info(f"Data schema validation result: {schema_valid}")
                    elif step == 'summary_stats' and args.summary_stats:
                        stats = generate_summary_statistics(processed_data)
                        logger.info(f"Summary statistics:\n{stats}")
                    elif step == 'visualize' and args.visualize:
                        visualize_data(processed_data, config.get('visualization_config', {}))
                    elif step == 'analyze_text' and args.analyze_text:
                        text_column = config['text_column']
                        sentiment = perform_sentiment_analysis(processed_data[text_column])
                        summary = summarize_text(processed_data[text_column])
                        logger.info(f"Sentiment analysis results: {sentiment}")
                        logger.info(f"Text summary: {summary}")
                    elif step == 'extract_entities' and args.extract_entities:
                        text_column = config['text_column']
                        entities = extract_entities(processed_data[text_column])
                        logger.info(f"Extracted entities: {entities}")
                        plot_entity_distribution(entities)
                    elif step == 'model_topics' and args.model_topics:
                        text_column = config['text_column']
                        topics = perform_topic_modeling(processed_data[text_column], num_topics=config.get('num_topics', 5))
                        logger.info(f"Discovered topics: {topics}")
                        generate_word_cloud(topics)
                    elif step == 'select_features' and args.select_features:
                        selected_features = select_features(processed_data, config['target_column'], config.get('feature_selection_method', 'mutual_info'))
                        processed_data = processed_data[selected_features + [config['target_column']]]
                    elif step == 'reduce_dimensions' and args.reduce_dimensions:
                        reduced_data = reduce_dimensions(processed_data, config.get('n_components', 2), config.get('reduction_method', 'pca'))
                        processed_data = dd.concat([processed_data, reduced_data], axis=1)
                    elif step == 'auto_feature_engineering' and args.auto_feature_engineering:
                        processed_data = auto_feature_engineering(processed_data, config['target_column'])
                    elif step == 'handle_imbalanced' and args.handle_imbalanced:
                        processed_data = handle_imbalanced_data(processed_data, config['target_column'], config.get('imbalance_method', 'smote'))
                    elif step == 'train_model' and 'model_type' in config:
                        model = get_model(config['model_type'])
                        X = processed_data.drop(columns=[config['target_column']])
                        y = processed_data[config['target_column']]
                        evaluation_results = evaluate_model(model, X, y, config['evaluation_metrics'])
                        if args.auto_tune:
                            try:
                                param_grid = config.get('param_grid', {})
                                grid_search = GridSearchCV(model, param_grid, cv=3)
                                grid_search.fit(X, y)
                                model = grid_search.best_estimator_
                                logger.info(f"Best parameters: {grid_search.best_params_}")
                            except Exception as e:
                                logger.error(f"Error during auto-tuning: {str(e)}")
                                logger.info("Falling back to default model fitting")
                                model.fit(X, y)
                        else:
                            model.fit(X, y)
                        
                        logger.info(f"Model evaluation results: {evaluation_results}")
                    # Execute custom plugins
                    elif step in custom_plugins:
                        processed_data = custom_plugins[step](processed_data, config)
                    
                    # Cache the result
                    if args.use_cache:
                        cache_result(step, processed_data)
                    
                    completed_steps.append(step)
                    pbar.update(1)
                
                # Save pipeline state after each step
                save_pipeline_state({'data': processed_data, 'completed_steps': completed_steps}, f'pipeline_state_{step}.pkl')

        # Save processed data
        output_file = args.output or config['output_file']
        processed_data.to_csv(output_file, index=False, single_file=True)
        logger.info(f"Processed data saved to {output_file}")

        # Generate comprehensive report
        if args.generate_report:
            generate_report(processed_data, completed_steps, config, output_file)

    except Exception as e:
        handle_error(e)
    finally:
        if 'client' in locals():
            client.close()

def generate_report(data: dd.DataFrame, steps: List[str], config: Dict[str, Any], output_file: str) -> None:
    report = f"Data Processing Pipeline Report\n{'='*30}\n\n"
    report += f"Configuration: {config}\n\n"
    report += f"Steps Completed: {steps}\n\n"
    report += f"Data Shape: {data.shape}\n\n"
    report += f"Data Types:\n{data.dtypes}\n\n"
    report += f"Summary Statistics:\n{data.describe()}\n\n"
    report += f"Output File: {output_file}\n\n"
    
    report_file = 'pipeline_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Comprehensive report generated: {report_file}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
