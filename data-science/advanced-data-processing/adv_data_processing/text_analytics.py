from textblob import TextBlob
from gensim.summarization import summarize

def perform_sentiment_analysis(text: str) -> Dict[str, float]:
    """Perform sentiment analysis on the given text."""
    blob = TextBlob(text)
    return {'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}

def summarize_text(text: str, ratio: float = 0.2) -> str:
    """Summarize the given text."""
    return summarize(text, ratio=ratio)

