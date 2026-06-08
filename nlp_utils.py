import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict, Counter
import math
import os
import pickle
import re

# Ensure NLTK downloads and data paths work in Render and local environments.
PROJECT_ROOT = os.path.dirname(__file__)
NLTK_DATA_DIR = os.path.join(PROJECT_ROOT, 'nltk_data')
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.insert(0, NLTK_DATA_DIR)

RESOURCE_PATHS = {
    'punkt_tab': 'tokenizers/punkt_tab',
    'punkt': 'tokenizers/punkt',
    'wordnet': 'corpora/wordnet',
    'stopwords': 'corpora/stopwords',
}
for resource, path in RESOURCE_PATHS.items():
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(resource, download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception:
            pass

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
except LookupError:
    stop_words = set()


def safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except LookupError:
        return re.findall(r"\b\w+\b", text)

# Model persistence
MODEL_FILE = os.path.join(os.path.dirname(__file__), 'category_model.pkl')

CATEGORY_TRAINING_DATA = [
    ("Stocks surge after strong corporate earnings", "Finance"),
    ("Bank regulators prepare new financial rules", "Finance"),
    ("Cryptocurrency market extends rally", "Finance"),
    ("Investment fund closes record quarter", "Finance"),
    ("Global markets react to interest rate news", "Finance"),
    ("Government approves new tax policy", "Politics"),
    ("Election debate highlights healthcare and economy", "Politics"),
    ("Senators propose legislation on tech privacy", "Politics"),
    ("President meets foreign leaders to discuss trade", "Politics"),
    ("Local government votes on education budget", "Politics"),
    ("Championship match ends with a thrilling finish", "Sports"),
    ("Olympic team prepares for the next summer games", "Sports"),
    ("Football coach discusses new season strategy", "Sports"),
    ("Tennis star wins another major title", "Sports"),
    ("Basketball league announces playoff schedule", "Sports"),
    ("New smartphone launch adds AI-powered camera", "Technology"),
    ("Startups compete in artificial intelligence challenge", "Technology"),
    ("Cybersecurity breach exposes millions of records", "Technology"),
    ("Software update brings new productivity features", "Technology"),
    ("Electric vehicle manufacturer expands production", "Technology"),
    ("Movie premiere draws crowds to the theater", "Entertainment"),
    ("Artist releases new album after world tour", "Entertainment"),
    ("Fashion week showcases the latest seasonal trends", "Entertainment"),
    ("Celebrity interview highlights upcoming film", "Entertainment"),
    ("Streaming service adds several new shows", "Entertainment"),
    ("Health experts recommend new wellness practices", "Health"),
    ("Medical study reveals new treatment progress", "Health"),
    ("Nutrition guidelines emphasize balanced diets", "Health"),
    ("Fitness app launches new training program", "Health"),
    ("Hospital expands services with advanced care", "Health"),
    ("Ceasefire talks begin after weeks of intense fighting", "War"),
    ("Military forces advance near strategic city", "War"),
    ("Conflict causes displacement of thousands of civilians", "War"),
    ("Automaker recalls thousands of vehicles over safety defect", "Automobile"),
    ("New electric car model promises increased range", "Automobile"),
    ("Auto sales rise as demand for SUVs continues", "Automobile"),
    ("Airline extends cancellation policy amid weather disruptions", "Airlines"),
    ("Budget carrier launches new international routes", "Airlines"),
    ("Airline reports record passenger numbers this quarter", "Airlines"),
    ("High-speed rail project receives government approval", "Train"),
    ("Train services disrupted after signal failure", "Train"),
    ("Commuter rail upgrades improve punctuality", "Train"),
    ("Tourism rebounds as countries ease travel restrictions", "Travel"),
    ("Top 10 destinations for summer travel announced", "Travel"),
    ("Travel app introduces new itinerary planning features", "Travel"),
    ("University launches scholarship program for first-generation students", "Education"),
    ("New study finds improvements in remote learning engagement", "Education"),
    ("School district approves curriculum changes for STEM classes", "Education"),
]

CATEGORY_PRIORS = {}
WORD_COUNTS = defaultdict(Counter)
VOCABULARY = set()
TOTAL_DOCS = 0


def save_model(path=MODEL_FILE):
    try:
        data = {
            'priors': CATEGORY_PRIORS,
            'word_counts': {k: dict(v) for k, v in WORD_COUNTS.items()},
            'vocabulary': list(VOCABULARY),
            'total_docs': TOTAL_DOCS,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass


def load_model(path=MODEL_FILE):
    global CATEGORY_PRIORS, WORD_COUNTS, VOCABULARY, TOTAL_DOCS
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        CATEGORY_PRIORS = data.get('priors', {})
        WORD_COUNTS = defaultdict(Counter)
        for k, v in data.get('word_counts', {}).items():
            WORD_COUNTS[k] = Counter(v)
        VOCABULARY = set(data.get('vocabulary', []))
        TOTAL_DOCS = int(data.get('total_docs', 0))
        return True
    except Exception:
        return False


def normalize_text(text):
    tokens = safe_word_tokenize(text.lower())
    normalized = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            stem = stemmer.stem(lemma)
            normalized.append(stem)
    return normalized


def train_category_model():
    global CATEGORY_PRIORS, WORD_COUNTS, VOCABULARY, TOTAL_DOCS
    class_document_counts = Counter()
    for text, category in CATEGORY_TRAINING_DATA:
        TOTAL_DOCS += 1
        class_document_counts[category] += 1
        words = normalize_text(text)
        for word in words:
            WORD_COUNTS[category][word] += 1
            VOCABULARY.add(word)

    for category, count in class_document_counts.items():
        CATEGORY_PRIORS[category] = math.log(count / TOTAL_DOCS)


def predict_category(text):
    words = normalize_text(text)
    if not words:
        return "General"

    best_category = None
    best_score = -math.inf
    vocab_size = len(VOCABULARY)

    for category, prior in CATEGORY_PRIORS.items():
        score = prior
        total_words_in_category = sum(WORD_COUNTS[category].values())
        for word in words:
            word_count = WORD_COUNTS[category].get(word, 0)
            # Laplace smoothing
            score += math.log((word_count + 1) / (total_words_in_category + vocab_size))
        if score > best_score:
            best_score = score
            best_category = category

    return best_category or "General"


# Load persisted model if available, otherwise train and save
if not load_model():
    train_category_model()
    try:
        save_model()
    except Exception:
        pass


def process_text(text):
    tokens = safe_word_tokenize(text)
    stems = [stemmer.stem(t) for t in tokens]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}


def summarize_text(text, word_limit=30):
    tokens = safe_word_tokenize(text)
    return " ".join(tokens[:word_limit])


def categorize(text, description="", content=""):
    # Combine text fields for better categorization
    combined_text = f"{text} {description} {content}"
    predicted = predict_category(combined_text)
    # Map the prediction to more application-friendly categories
    if predicted in {"Finance", "Sports", "Politics", "Technology", "Health", "Entertainment", "War", "Automobile", "Airlines", "Train", "Travel", "Education"}:
        return predicted
    return "General"
