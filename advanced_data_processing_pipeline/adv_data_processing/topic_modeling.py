from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

def perform_topic_modeling(texts: List[str], num_topics: int = 5) -> List[List[Tuple[str, float]]]:
    """Perform topic modeling on the given texts."""
    # Preprocess the texts
    processed_texts = [[word for word in simple_preprocess(doc) if word not in STOPWORDS] for doc in texts]
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # Train LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100)
    
    return lda_model.print_topics()

