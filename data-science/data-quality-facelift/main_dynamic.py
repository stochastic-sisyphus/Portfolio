import json
import logging
import torch
from main import EnhancedDataQualityPlatform, DatasetInfo
from model_evaluation import evaluate_models, load_dataset_sample
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_best_model_config():
    try:
        with open('best_model_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("Best model config not found. Using default configuration.")
        return {
            "llm": "huggingface_t5",
            "embedding": "sentence-transformers/all-MiniLM-L6-v2"
        }

def get_model(config):
    if config['llm'].startswith('huggingface'):
        model_name = config['llm'].split('_')[1]
        if model_name == 't5':
            model_name = 't5-small'  # Use a specific T5 model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HuggingFacePipeline(pipeline=pipeline("text2text-generation", model=model_name, device=device, max_new_tokens=50))
    else:
        raise ValueError(f"Unsupported LLM: {config['llm']}")

def get_embeddings(config):
    return HuggingFaceEmbeddings(model_name=config['embedding'])

def chunk_text(text, max_length=500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def main():
    try:
        # Load the best model configuration
        config = load_best_model_config()

        # Initialize the platform with the best models
        platform = EnhancedDataQualityPlatform(
            llm=get_model(config),
            embeddings=get_embeddings(config)
        )

        # Define datasets
        datasets = [
            DatasetInfo("customer_data", "data/customer_data.csv", "csv"),
            DatasetInfo("product_reviews", "data/product_reviews.json", "json"),
        ]

        # Run pipeline for each dataset
        for dataset_info in datasets:
            platform.run_pipeline(dataset_info)

        # Perform a sample semantic search across all datasets
        search_query = "customer satisfaction"
        chunked_query = chunk_text(search_query)
        search_results = []
        for chunk in chunked_query:
            results = platform.semantic_search(chunk, k=5)
            search_results.extend(results)
        logging.info(f"\nSemantic Search Results for query: '{search_query}':")
        logging.info(json.dumps(search_results[:5], indent=2))  # Limit to top 5 results

        # Evaluate search performance
        test_queries = ["product quality", "delivery speed"]
        ground_truth = [[0, 2, 5], [1, 3, 4]]  # Example relevant document indices
        performance_metrics = platform.evaluate_search_performance(test_queries, ground_truth)
        logging.info("\nSearch Performance Metrics:")
        logging.info(json.dumps(performance_metrics, indent=2))

        # Perform additional analyses as needed
        for dataset_name in platform.datasets.keys():
            # Time series analysis
            if 'date' in platform.datasets[dataset_name].columns:
                time_series_results = platform.perform_time_series_analysis(dataset_name, 'date', 'value')
                logging.info(f"\nTime Series Analysis Results for {dataset_name}:")
                logging.info(json.dumps(time_series_results, indent=2))

            # Hypothesis testing
            if 'group' in platform.datasets[dataset_name].columns and 'value' in platform.datasets[dataset_name].columns:
                hypothesis_results = platform.perform_hypothesis_testing(dataset_name, 'group', 'value')
                logging.info(f"\nHypothesis Testing Results for {dataset_name}:")
                logging.info(json.dumps(hypothesis_results, indent=2))

            # Anomaly detection
            anomalies = platform.detect_anomalies(dataset_name)
            logging.info(f"\nDetected Anomalies in {dataset_name}:")
            logging.info(anomalies.head())

            # RAG query
            rag_query = "Summarize the main trends in customer satisfaction"
            chunked_query = chunk_text(rag_query)
            rag_results = []
            for chunk in chunked_query:
                result = platform.rag_query(chunk, dataset_name)
                rag_results.append(result)
            combined_result = " ".join(rag_results)
            logging.info(f"\nRAG Query Results for {dataset_name}:")
            logging.info(combined_result)

        logging.info("Dynamic analysis complete.")

    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()