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
    # Finance - Expanded (20 samples)
    ("Stocks surge after strong corporate earnings report", "Finance"),
    ("Bank regulators prepare new financial rules and regulations", "Finance"),
    ("Cryptocurrency market extends rally to record highs", "Finance"),
    ("Investment fund closes record quarter with profits", "Finance"),
    ("Global markets react to interest rate news announcement", "Finance"),
    ("Federal Reserve makes decision on monetary policy", "Finance"),
    ("Stock market closes with significant gains today", "Finance"),
    ("Wall Street analysts predict economic growth", "Finance"),
    ("Treasury bonds reach new yield levels", "Finance"),
    ("Banking sector shows strength in quarterly earnings", "Finance"),
    ("Gold prices reach highest level in months", "Finance"),
    ("Forex markets respond to currency fluctuations", "Finance"),
    ("Dividend payments announced by major corporations", "Finance"),
    ("Asset management firms report record inflows", "Finance"),
    ("Economic indicators show positive consumer spending", "Finance"),
    ("Stock exchange implements new trading regulations", "Finance"),
    ("Inflation data influences market sentiment", "Finance"),
    ("Earnings season brings corporate profit surprises", "Finance"),
    ("Investment portfolio performance exceeds expectations", "Finance"),
    ("Mergers and acquisitions deal announced today", "Finance"),
    
    # Politics - Expanded (18 samples)
    ("Government approves new tax policy legislation", "Politics"),
    ("Election debate highlights healthcare and economic issues", "Politics"),
    ("Senators propose legislation on tech privacy concerns", "Politics"),
    ("President meets foreign leaders to discuss trade deals", "Politics"),
    ("Local government votes on education budget proposal", "Politics"),
    ("Political campaign launches new initiative", "Politics"),
    ("Congress debates bill on environmental protection", "Politics"),
    ("Mayor announces new urban development project", "Politics"),
    ("Senate confirms cabinet nominee after hearing", "Politics"),
    ("Political rally draws thousands of supporters", "Politics"),
    ("Supreme Court makes ruling on constitutional issue", "Politics"),
    ("Governor signs legislation into law", "Politics"),
    ("Parliament debates immigration policy reform", "Politics"),
    ("Political leaders announce coalition agreement", "Politics"),
    ("Election results announced after voting day", "Politics"),
    ("Congress appropriates funds for infrastructure", "Politics"),
    ("Political debate centers on healthcare reform", "Politics"),
    ("Leaders negotiate international trade agreement", "Politics"),
    
    # Sports - Expanded (18 samples)
    ("Championship match ends with thrilling finish", "Sports"),
    ("Olympic team prepares for next summer games", "Sports"),
    ("Football coach discusses new season strategy", "Sports"),
    ("Tennis star wins another major tournament title", "Sports"),
    ("Basketball league announces playoff schedule", "Sports"),
    ("Baseball team secures division title", "Sports"),
    ("Soccer championship final draws record crowd", "Sports"),
    ("Marathon runner sets new world record", "Sports"),
    ("Golf tournament features international players", "Sports"),
    ("Boxing champion retains title with knockout victory", "Sports"),
    ("Swimming competition breaks multiple records", "Sports"),
    ("Rugby team qualifies for world tournament", "Sports"),
    ("Cricket series ends in thrilling conclusion", "Sports"),
    ("Formula racing season reaches climax", "Sports"),
    ("Athlete signs record sponsorship deal", "Sports"),
    ("Sports injury sidelines star player", "Sports"),
    ("Coaching change announced by major franchise", "Sports"),
    ("Youth sports program expands to new city", "Sports"),
    
    # Technology - Expanded (18 samples)
    ("New smartphone launch adds AI-powered camera features", "Technology"),
    ("Startups compete in artificial intelligence challenge", "Technology"),
    ("Cybersecurity breach exposes millions of records", "Technology"),
    ("Software update brings new productivity features", "Technology"),
    ("Electric vehicle manufacturer expands production", "Technology"),
    ("Tech company announces quantum computing breakthrough", "Technology"),
    ("Cloud computing platform launches new services", "Technology"),
    ("Machine learning model improves accuracy", "Technology"),
    ("Virtual reality technology shows promising applications", "Technology"),
    ("Blockchain technology gains enterprise adoption", "Technology"),
    ("5G network deployment accelerates", "Technology"),
    ("Data center expansion announced by tech giant", "Technology"),
    ("Programming language releases new version", "Technology"),
    ("Artificial intelligence outperforms humans on task", "Technology"),
    ("Cybersecurity firm discovers new vulnerability", "Technology"),
    ("Augmented reality glasses unveiled to market", "Technology"),
    ("Internet privacy tools gain popularity", "Technology"),
    ("Tech startup raises record funding round", "Technology"),
    
    # Entertainment - Expanded (16 samples)
    ("Movie premiere draws crowds to theater", "Entertainment"),
    ("Artist releases new album after world tour", "Entertainment"),
    ("Fashion week showcases latest seasonal trends", "Entertainment"),
    ("Celebrity interview highlights upcoming film", "Entertainment"),
    ("Streaming service adds several new shows", "Entertainment"),
    ("Concert tour announces additional dates", "Entertainment"),
    ("Television series renewal confirmed by network", "Entertainment"),
    ("Award show ceremony celebrates industry achievements", "Entertainment"),
    ("Film festival opens with highly anticipated premiere", "Entertainment"),
    ("Music festival announces headline performers", "Entertainment"),
    ("Documentary wins prestigious filmmaking award", "Entertainment"),
    ("Theater production opens on Broadway", "Entertainment"),
    ("Celebrity collaboration creates buzz online", "Entertainment"),
    ("Entertainment platform reports record viewership", "Entertainment"),
    ("Voice actor wins award for performance", "Entertainment"),
    ("Animation studio announces new project", "Entertainment"),
    
    # Health - Expanded (16 samples)
    ("Health experts recommend new wellness practices", "Health"),
    ("Medical study reveals new treatment progress", "Health"),
    ("Nutrition guidelines emphasize balanced diets", "Health"),
    ("Fitness app launches new training program", "Health"),
    ("Hospital expands services with advanced care", "Health"),
    ("Pharmaceutical company approves new drug", "Health"),
    ("Mental health initiative receives government funding", "Health"),
    ("Vaccine development accelerates against disease", "Health"),
    ("Healthcare technology improves patient outcomes", "Health"),
    ("Coronavirus cases decline in region", "Health"),
    ("Medical device receives regulatory approval", "Health"),
    ("Public health campaign fights disease", "Health"),
    ("Fitness industry reports growth in membership", "Health"),
    ("Research shows benefits of exercise", "Health"),
    ("Health insurance coverage expands", "Health"),
    ("Wellness program improves employee health", "Health"),
    
    # War/Conflict - Expanded (12 samples)
    ("Ceasefire talks begin after weeks of fighting", "War"),
    ("Military forces advance near strategic city", "War"),
    ("Conflict causes displacement of thousands", "War"),
    ("International coalition intervenes in conflict", "War"),
    ("Troops withdraw from disputed territory", "War"),
    ("Peace negotiations resume after break", "War"),
    ("Military deployment announced for region", "War"),
    ("Humanitarian crisis worsens in war zone", "War"),
    ("Weapons embargo imposed by world powers", "War"),
    ("Diplomatic efforts to end conflict", "War"),
    ("Casualties reported in ongoing conflict", "War"),
    ("Reconstruction efforts begin after war ends", "War"),
    
    # Automobile - Expanded (12 samples)
    ("Automaker recalls thousands of vehicles for safety defect", "Automobile"),
    ("New electric car model promises increased range", "Automobile"),
    ("Auto sales rise as demand for SUVs continues", "Automobile"),
    ("Luxury car brand launches new line", "Automobile"),
    ("Autonomous vehicle testing expands to cities", "Automobile"),
    ("Truck manufacturer announces innovation", "Automobile"),
    ("Electric charging infrastructure expands", "Automobile"),
    ("Vehicle safety rating improves", "Automobile"),
    ("Auto industry reports quarterly earnings", "Automobile"),
    ("New vehicle model wins design award", "Automobile"),
    ("Car company invests in sustainability", "Automobile"),
    ("Motor show features latest models", "Automobile"),
    
    # Airlines - Expanded (12 samples)
    ("Airline extends cancellation policy amid disruptions", "Airlines"),
    ("Budget carrier launches new international routes", "Airlines"),
    ("Airline reports record passenger numbers", "Airlines"),
    ("Airport expansion project approved", "Airlines"),
    ("Airline increases flight frequency", "Airlines"),
    ("Travel demand rebounds after restrictions", "Airlines"),
    ("Aircraft order placed with manufacturer", "Airlines"),
    ("Airline merges with competitor", "Airlines"),
    ("Flight safety record recognized", "Airlines"),
    ("Baggage handling improvements announced", "Airlines"),
    ("Airline loyalty program expanded", "Airlines"),
    ("Air traffic control system upgraded", "Airlines"),
    
    # Education - Expanded (14 samples)
    ("University launches scholarship program for students", "Education"),
    ("New study finds improvements in remote learning", "Education"),
    ("School district approves curriculum changes for STEM", "Education"),
    ("Educational technology company raises funding", "Education"),
    ("University rankings released for year", "Education"),
    ("Student loan reform proposed by lawmakers", "Education"),
    ("College enrollment increases this semester", "Education"),
    ("Teaching methods improved with research", "Education"),
    ("Online education platform expands offerings", "Education"),
    ("School receives grant for improvement", "Education"),
    ("Teacher recruitment program launched", "Education"),
    ("Educational standards updated nationally", "Education"),
    ("Scholarship awards announced for students", "Education"),
    ("Distance learning tools evaluated", "Education"),
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


def augment_training_data():
    """Augment training data by creating variations"""
    augmented = []
    for text, category in CATEGORY_TRAINING_DATA:
        # Original
        augmented.append((text, category))
        # Variations with different capitalization
        augmented.append((text.lower(), category))
        augmented.append((text.upper(), category))
    return augmented


def train_category_model():
    global CATEGORY_PRIORS, WORD_COUNTS, VOCABULARY, TOTAL_DOCS
    
    # Reset model state before training
    CATEGORY_PRIORS = {}
    WORD_COUNTS = defaultdict(Counter)
    VOCABULARY = set()
    TOTAL_DOCS = 0
    
    # Use augmented data for better training
    training_data = augment_training_data()
    
    class_document_counts = Counter()
    for text, category in training_data:
        TOTAL_DOCS += 1
        class_document_counts[category] += 1
        words = normalize_text(text)
        
        for word in words:
            WORD_COUNTS[category][word] += 1
            VOCABULARY.add(word)

    # Calculate priors with smoothing to handle rare categories
    smoothing_factor = 1.0
    for category in set(cat for _, cat in CATEGORY_TRAINING_DATA):
        count = class_document_counts.get(category, 1)
        CATEGORY_PRIORS[category] = math.log((count + smoothing_factor) / (TOTAL_DOCS + smoothing_factor))


def predict_category(text):
    """Predict category with improved confidence calculation"""
    words = normalize_text(text)
    if not words:
        return "General"

    best_category = None
    best_score = -math.inf
    scores = {}
    vocab_size = max(len(VOCABULARY), 1)

    # Calculate score for each category
    for category in CATEGORY_PRIORS.keys():
        score = CATEGORY_PRIORS.get(category, 0)
        total_words_in_category = max(sum(WORD_COUNTS[category].values()), 1)
        
        for word in words:
            word_count = WORD_COUNTS[category].get(word, 0)
            # Enhanced Laplace smoothing
            prob = (word_count + 0.5) / (total_words_in_category + 0.5 * vocab_size)
            score += math.log(max(prob, 1e-10))  # Avoid log(0)
        
        scores[category] = score
        if score > best_score:
            best_score = score
            best_category = category

    # Always return the best scoring category when words are present
    return best_category or "General"


# Load persisted model if available, otherwise train and save
if not load_model():
    train_category_model()
    try:
        save_model()
    except Exception:
        pass


def process_text(text):
    """Process text with multiple NLP techniques"""
    if not text:
        return {"tokens": [], "stems": [], "lemmas": []}
    
    tokens = safe_word_tokenize(text)
    stems = [stemmer.stem(t.lower()) for t in tokens if t.isalpha()]
    lemmas = [lemmatizer.lemmatize(t.lower()) for t in tokens if t.isalpha()]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}


def summarize_text(text, word_limit=30):
    """Create a summary of text with improved extraction"""
    if not text:
        return ""
    
    tokens = safe_word_tokenize(text)
    # Remove very short tokens
    tokens = [t for t in tokens if len(t) > 2]
    summary = " ".join(tokens[:min(word_limit, len(tokens))])
    return summary if summary else text[:100]


def get_text_statistics(text):
    """Get statistics about the text"""
    if not text:
        return {"word_count": 0, "char_count": 0, "sentence_count": 0}
    
    tokens = safe_word_tokenize(text)
    char_count = len(text)
    word_count = len(tokens)
    
    # Rough sentence count
    sentences = text.split('.')
    sentence_count = len([s for s in sentences if s.strip()])
    
    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": char_count / max(word_count, 1)
    }


def categorize(title, description="", content=""):
    """Categorize news with improved robustness"""
    # Combine text fields for better categorization
    combined_text = f"{title} {description} {content}".strip()
    
    if not combined_text or len(combined_text) < 3:
        return "General"
    
    predicted = predict_category(combined_text)
    
    # Ensure we return valid categories
    valid_categories = {
        "Finance", "Sports", "Politics", "Technology", "Health", 
        "Entertainment", "War", "Automobile", "Airlines", "Education"
    }
    
    if predicted in valid_categories:
        return predicted
    
    return "General"
