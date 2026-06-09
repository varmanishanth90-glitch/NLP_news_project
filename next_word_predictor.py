import os
import re
from collections import Counter, defaultdict
import pickle
import math

BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data", "next_word_dataset.txt")
MODEL_FILE = os.path.join(BASE_DIR, "bigram_model.pkl")


def load_dataset(path=DATA_FILE):
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


def tokenize(text):
    """Tokenize text into words, preserving some punctuation context"""
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if len(w) > 0]


def build_ngram_model(lines, n=2):
    """Build n-gram model with frequency counts"""
    ngrams = defaultdict(Counter)
    unigrams = Counter()
    
    for line in lines:
        tokens = tokenize(line)
        if len(tokens) < 1:
            continue
            
        for token in tokens:
            unigrams[token] += 1
        
        # Build n-grams
        if n == 2:  # Bigrams
            for a, b in zip(tokens, tokens[1:]):
                ngrams[a][b] += 1
        elif n == 3:  # Trigrams
            for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
                ngrams[(a, b)][c] += 1
                
    return ngrams, unigrams


def calculate_probabilities(ngrams):
    """Convert frequency counts to probabilities"""
    prob_model = {}
    for context, followers in ngrams.items():
        total = sum(followers.values())
        prob_model[context] = {
            word: count / total for word, count in followers.items()
        }
    return prob_model


def save_model(ngrams, unigrams, path=MODEL_FILE):
    """Save model to disk"""
    try:
        data = {
            'ngrams': {str(k): dict(v) for k, v in ngrams.items()},
            'unigrams': dict(unigrams),
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def load_model(path=MODEL_FILE):
    """Load model from disk"""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        ngrams = defaultdict(Counter)
        for k, v in data.get('ngrams', {}).items():
            try:
                ngrams[k] = Counter(v)
            except:
                pass
        unigrams = Counter(data.get('unigrams', {}))
        return ngrams, unigrams
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


# Load dataset and build models
DATA_LINES = load_dataset()

# Try to load from disk first
BIGRAM_MODEL, UNIGRAM_COUNTS = load_model()

# If not available, build from dataset
if BIGRAM_MODEL is None or UNIGRAM_COUNTS is None:
    BIGRAM_MODEL, UNIGRAM_COUNTS = build_ngram_model(DATA_LINES, n=2)
    save_model(BIGRAM_MODEL, UNIGRAM_COUNTS)

# Get most common words as fallback
TOP_UNIGRAMS = [word for word, _ in UNIGRAM_COUNTS.most_common(15)]


def predict_next_words(prefix, top_n=5):
    """Predict next words using bigram model with smoothing"""
    prefix = prefix or ""
    tokens = tokenize(prefix)
    
    # Fallback to top words if no input
    if not tokens:
        return TOP_UNIGRAMS[:top_n]

    last_word = tokens[-1]
    
    # Get bigram predictions if available
    if last_word in BIGRAM_MODEL and BIGRAM_MODEL[last_word]:
        suggestions = [word for word, _ in BIGRAM_MODEL[last_word].most_common(top_n)]
        if suggestions:
            return suggestions

    # Fallback to most common words
    return TOP_UNIGRAMS[:top_n]


def get_sample_prompts(count=5):
    """Get sample prompts from dataset"""
    return DATA_LINES[:count] if DATA_LINES else ["Type to get started"]


def augment_training_data(lines):
    """Augment training data by creating variations"""
    augmented = list(lines)
    for line in lines:
        # Create capitalized variation
        augmented.append(line.capitalize())
        # Create title case variation
        augmented.append(" ".join(word.capitalize() for word in line.split()))
    return augmented
