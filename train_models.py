#!/usr/bin/env python3
"""
Training script for NLP models - improves category classification and next-word prediction
Run this script to retrain models with enhanced data
"""

import os
import sys
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from nlp_utils import (
    CATEGORY_TRAINING_DATA, train_category_model, save_model,
    categorize, normalize_text, CATEGORY_PRIORS, WORD_COUNTS, VOCABULARY
)
from next_word_predictor import (
    load_dataset, build_ngram_model, save_model as save_ngram_model,
    augment_training_data, DATA_FILE
)


def train_all_models():
    """Train all models with enhanced data"""
    print("=" * 60)
    print("NLP Models Training Script")
    print("=" * 60)
    
    # Train category classification model
    print("\n[1/3] Training category classification model...")
    print(f"      Training data: {len(CATEGORY_TRAINING_DATA)} samples")
    
    train_category_model()
    save_model()
    
    print(f"      ✓ Categories trained: {len(CATEGORY_PRIORS)}")
    print(f"      ✓ Vocabulary size: {len(VOCABULARY)}")
    print(f"      ✓ Model saved to: category_model.pkl")
    
    # Train next-word prediction model
    print("\n[2/3] Training next-word prediction model...")
    data_lines = load_dataset(DATA_FILE)
    print(f"      Dataset samples: {len(data_lines)}")
    
    if data_lines:
        # Augment dataset
        augmented = augment_training_data(data_lines)
        print(f"      Augmented samples: {len(augmented)}")
        
        # Build n-gram model
        bigram_model, unigram_counts = build_ngram_model(augmented, n=2)
        save_ngram_model(bigram_model, unigram_counts)
        
        print(f"      ✓ Bigrams learned: {len(bigram_model)}")
        print(f"      ✓ Unique words: {len(unigram_counts)}")
        print(f"      ✓ Model saved to: bigram_model.pkl")
    else:
        print("      ⚠ Warning: No data found in dataset file")
        print(f"      Expected file: {DATA_FILE}")
    
    # Test models
    print("\n[3/3] Testing models...")
    test_categories()


def test_categories():
    """Test category classification with sample news"""
    test_samples = [
        ("Apple releases new iPhone with AI features", "Technology"),
        ("Federal Reserve raises interest rates", "Finance"),
        ("Championship game ends with overtime victory", "Sports"),
        ("New legislation passed by Senate", "Politics"),
        ("Hospital develops breakthrough treatment", "Health"),
        ("Celebrity wins award at ceremony", "Entertainment"),
        ("Auto company recalls defective vehicles", "Automobile"),
        ("Airline adds new international routes", "Airlines"),
        ("University announces scholarship program", "Education"),
    ]
    
    correct = 0
    print("\n      Testing classification accuracy:")
    print("      " + "-" * 50)
    
    for text, expected in test_samples:
        predicted = categorize(text)
        is_correct = predicted == expected
        correct += is_correct
        status = "✓" if is_correct else "✗"
        print(f"      {status} '{text[:40]}...'")
        print(f"         Expected: {expected}, Got: {predicted}")
    
    accuracy = (correct / len(test_samples)) * 100
    print("      " + "-" * 50)
    print(f"      Accuracy: {accuracy:.1f}% ({correct}/{len(test_samples)})")
    
    if accuracy >= 80:
        print("      ✓ Model performance: GOOD")
    elif accuracy >= 60:
        print("      ⚠ Model performance: ACCEPTABLE (consider adding more training data)")
    else:
        print("      ✗ Model performance: NEEDS IMPROVEMENT")


def show_model_stats():
    """Display model statistics"""
    print("\nModel Statistics:")
    print("-" * 40)
    print(f"Categories: {len(CATEGORY_PRIORS)}")
    for cat, count in sorted(CATEGORY_PRIORS.items()):
        words = sum(WORD_COUNTS[cat].values())
        print(f"  - {cat}: {words} words")
    print(f"\nTotal vocabulary: {len(VOCABULARY)} unique words")


if __name__ == "__main__":
    try:
        train_all_models()
        show_model_stats()
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
