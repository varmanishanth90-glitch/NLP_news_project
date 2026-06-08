import os
import re
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "data", "next_word_dataset.txt")


def load_dataset(path=DATA_FILE):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_bigram_model(lines):
    bigrams = defaultdict(Counter)
    unigrams = Counter()
    for line in lines:
        tokens = tokenize(line)
        for token in tokens:
            unigrams[token] += 1
        for a, b in zip(tokens, tokens[1:]):
            bigrams[a][b] += 1
    return bigrams, unigrams


DATA_LINES = load_dataset()
BIGRAM_MODEL, UNIGRAM_COUNTS = build_bigram_model(DATA_LINES)
TOP_UNIGRAMS = [word for word, _ in UNIGRAM_COUNTS.most_common(10)]


def predict_next_words(prefix, top_n=5):
    prefix = prefix or ""
    tokens = tokenize(prefix)
    if not tokens:
        return TOP_UNIGRAMS[:top_n]

    last_word = tokens[-1]
    if last_word in BIGRAM_MODEL and BIGRAM_MODEL[last_word]:
        return [word for word, _ in BIGRAM_MODEL[last_word].most_common(top_n)]

    return TOP_UNIGRAMS[:top_n]


def get_sample_prompts(count=5):
    return DATA_LINES[:count]
