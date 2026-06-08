import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import defaultdict, Counter
import math

RESOURCE_PATHS = {
    'punkt': 'tokenizers/punkt',
    'wordnet': 'corpora/wordnet',
    'stopwords': 'corpora/stopwords',
}
for resource, path in RESOURCE_PATHS.items():
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(resource)
        except Exception:
            pass

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
except LookupError:
    stop_words = set()

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
]

CATEGORY_PRIORS = {}
WORD_COUNTS = defaultdict(Counter)
VOCABULARY = set()
TOTAL_DOCS = 0


def normalize_text(text):
    tokens = word_tokenize(text.lower())
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


train_category_model()


def process_text(text):
    tokens = word_tokenize(text)
    stems = [stemmer.stem(t) for t in tokens]
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}


def summarize_text(text, word_limit=30):
    tokens = word_tokenize(text)
    return " ".join(tokens[:word_limit])


def categorize(text):
    predicted = predict_category(text)
    # Map the prediction to more application-friendly categories
    if predicted in {"Finance", "Sports", "Politics", "Technology", "Health", "Entertainment"}:
        return predicted
    return "General"
