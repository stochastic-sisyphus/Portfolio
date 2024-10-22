import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Extract named entities from the given text."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

