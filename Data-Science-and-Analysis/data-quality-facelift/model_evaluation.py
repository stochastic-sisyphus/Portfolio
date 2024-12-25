import json
import pandas as pd
from typing import Dict, Any
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from main import EnhancedDataQualityPlatform, DatasetInfo
import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset_sample(file_path: str, n_samples: int = 1000) -> pd.DataFrame:
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, nrows=n_samples)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path, lines=True, nrows=n_samples)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def evaluate_models(dataset_sample: pd.DataFrame):
    models = {
        "huggingface_t5": HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="t5-small")),
        "huggingface_bart": HuggingFacePipeline(pipeline=pipeline("text2text-generation", model="facebook/bart-base")),
    }
    embeddings_dict = {
        "sentence-transformers": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        "mpnet": HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
    }
    results = {}
    for model_name, model in models.items():
        for embedding_name, embedding in embeddings_dict.items():
            logging.info(f"Evaluating {model_name} with {embedding_name}")
            platform = EnhancedDataQualityPlatform(llm=model, embeddings=embedding)
            
            platform.datasets["test"] = dataset_sample
            
            try:
                metadata = platform.generate_metadata("test")
                metadata_score = len(json.dumps(metadata))
            except Exception as e:
                logging.error(f"Error generating metadata with {model_name}: {e}")
                metadata_score = 0
            
            try:
                quality_report = platform.assess_data_quality("test")
                recommendations = platform.generate_recommendations("test")
                recommendation_score = len(json.dumps(recommendations))
            except Exception as e:
                logging.error(f"Error generating recommendations with {model_name}: {e}")
                recommendation_score = 0
            
            try:
                platform.build_vector_store("test")
                search_results = platform.semantic_search("test query", k=5)
                search_score = len(json.dumps(search_results))
            except Exception as e:
                logging.error(f"Error performing semantic search with {model_name}: {e}")
                search_score = 0
            
            total_score = metadata_score + recommendation_score + search_score
            results[f"{model_name}_{embedding_name}"] = {
                "metadata_score": metadata_score,
                "recommendation_score": recommendation_score,
                "search_score": search_score,
                "total_score": total_score
            }
    
    best_model = max(results, key=lambda x: results[x]["total_score"])
    
    with open('model_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    best_config = {
        "llm": best_model.split('_')[0],
        "embedding": best_model.split('_')[1]
    }
    with open('best_model_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    logging.info(f"Best model configuration: {best_config}")
    return best_config

if __name__ == "__main__":
    sample_data = load_dataset_sample("data/sample_dataset.csv")
    evaluate_models(sample_data)