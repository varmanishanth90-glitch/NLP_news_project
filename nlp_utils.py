import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def process_text(text):
    tokens = word_tokenize(text)
    stems = [stemmer.stem(t) for t in tokens]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}

def summarize_text(text, word_limit=30):
    tokens = word_tokenize(text)
    return " ".join(tokens[:word_limit])

def categorize(text):
    text = text.lower()
    if "finance" in text or "bank" in text:
        return "Finance"
    elif "sports" in text:
        return "Sports"
    elif "politics" in text:
        return "Politics"
    elif "tech" in text or "ai" in text:
        return "Technology"
    else:
        return "General"
