"""
Gaming Console Sentiment Analysis - Data Preprocessing Module
Strict filtering for gaming console hardware reviews only
"""

import json
import gzip
import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_amazon_dataset(file_path: str) -> pd.DataFrame:
    """Load Amazon Video Games dataset from gzipped JSON file."""
    print(f"Loading Amazon dataset from: {file_path}")
    
    reviews = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                review = json.loads(line.strip())
                reviews.append(review)
                if line_num % 50000 == 0:
                    print(f"Loaded {line_num:,} reviews...")
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(reviews)
    print(f"Total reviews loaded: {len(df):,}")
    return df

def strict_gaming_console_filter(text: str) -> bool:
    """
    Strict filtering for gaming console hardware reviews only.
    Returns True only if text contains specific gaming console keywords.
    """
    if not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Primary gaming console keywords (must contain at least one)
    primary_keywords = [
        'playstation', 'ps4', 'ps5', 'ps3', 'ps2',
        'xbox', 'xbox one', 'xbox series', 'xbox 360',
        'nintendo switch', 'nintendo wii', 'wii u',
        'gaming console', 'video game console', 'game console'
    ]
    
    # Hardware-specific terms (strengthen gaming console context)
    hardware_terms = [
        'console', 'system', 'hardware', 'controller', 'gamepad',
        'hdmi', 'storage', 'hard drive', 'ssd', 'graphics'
    ]
    
    # Exclusion terms (filter out non-console gaming content)
    exclusion_terms = [
        'alexa', 'echo', 'fire tv', 'fire stick', 'kindle',
        'tablet', 'phone', 'mobile', 'pc game', 'computer game',
        'board game', 'card game', 'toy', 'action figure'
    ]
    
    # Check for exclusion terms first
    for term in exclusion_terms:
        if term in text_lower:
            return False
    
    # Must contain at least one primary keyword
    has_primary = any(keyword in text_lower for keyword in primary_keywords)
    
    # Bonus points for hardware terms
    has_hardware = any(term in text_lower for term in hardware_terms)
    
    return has_primary and has_hardware

def filter_gaming_console_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset for gaming console reviews only."""
    print("Applying strict gaming console filtering...")
    
    # Combine reviewText and summary for filtering
    df['combined_text'] = df['reviewText'].fillna('') + ' ' + df['summary'].fillna('')
    
    # Apply strict filtering
    console_mask = df['combined_text'].apply(strict_gaming_console_filter)
    
    filtered_df = df[console_mask].copy()
    
    print(f"Reviews before filtering: {len(df):,}")
    print(f"Reviews after filtering: {len(filtered_df):,}")
    print(f"Filtering ratio: {len(filtered_df)/len(df)*100:.2f}%")
    
    return filtered_df

def verify_gaming_console_content(df: pd.DataFrame, sample_size: int = 50) -> List[Dict]:
    """
    Manually verify a sample of filtered reviews to ensure gaming console focus.
    Returns verification samples for inspection.
    """
    print(f"Verifying {sample_size} random samples for gaming console content...")
    
    # Sample reviews across different ratings
    verification_samples = []
    
    for rating in [1, 2, 3, 4, 5]:
        rating_reviews = df[df['overall'] == rating]
        if len(rating_reviews) > 0:
            sample_count = min(10, len(rating_reviews))
            samples = rating_reviews.sample(n=sample_count, random_state=42)
            
            for _, review in samples.iterrows():
                verification_samples.append({
                    'rating': rating,
                    'reviewText': review['reviewText'][:500] + '...' if len(str(review['reviewText'])) > 500 else review['reviewText'],
                    'summary': review['summary'],
                    'asin': review.get('asin', 'N/A')
                })
    
    return verification_samples[:sample_size]

def map_ratings_to_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Map Amazon 5-star ratings to 3-class sentiment labels."""
    df = df.copy()
    
    # Map ratings to sentiment
    sentiment_mapping = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}  # 0=negative, 1=neutral, 2=positive
    df['sentiment'] = df['overall'].map(sentiment_mapping)
    
    # Remove rows with missing sentiment
    df = df.dropna(subset=['sentiment'])
    df['sentiment'] = df['sentiment'].astype(int)
    
    return df

def clean_text(text: str) -> str:
    """Clean and preprocess text for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation but keep apostrophes for contractions
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Remove extra spaces
    text = text.strip()
    
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the filtered gaming console dataset."""
    print("Preprocessing gaming console reviews...")
    
    df = df.copy()
    
    # Clean text fields
    df['clean_reviewText'] = df['reviewText'].fillna('').apply(clean_text)
    df['clean_summary'] = df['summary'].fillna('').apply(clean_text)
    
    # Combine cleaned text
    df['clean_text'] = df['clean_reviewText'] + ' ' + df['clean_summary']
    
    # Remove very short reviews (less than 10 characters)
    df = df[df['clean_text'].str.len() >= 10]
    
    # Remove duplicates based on cleaned text
    df = df.drop_duplicates(subset=['clean_text'])
    
    print(f"Reviews after preprocessing: {len(df):,}")
    
    return df

def balance_dataset(df: pd.DataFrame, max_samples_per_class: int = 5000) -> pd.DataFrame:
    """Balance the dataset to avoid extreme class imbalance."""
    print("Balancing dataset across sentiment classes...")
    
    balanced_dfs = []
    
    for sentiment in [0, 1, 2]:  # negative, neutral, positive
        sentiment_df = df[df['sentiment'] == sentiment]
        
        if len(sentiment_df) > max_samples_per_class:
            sentiment_df = sentiment_df.sample(n=max_samples_per_class, random_state=42)
        
        balanced_dfs.append(sentiment_df)
        print(f"Sentiment {sentiment}: {len(sentiment_df):,} samples")
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"Balanced dataset size: {len(balanced_df):,}")
    return balanced_df

def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
    """Split data into train, validation, and test sets."""
    print("Splitting data into train/validation/test sets...")
    
    X = df['clean_text']
    y = df['sentiment']
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
    )
    
    print(f"Train set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_and_process_gaming_console_data(data_path: str) -> Tuple:
    """
    Complete pipeline to load and process gaming console sentiment data.
    Returns processed train/val/test splits and verification samples.
    """
    # Load raw Amazon dataset
    raw_df = load_amazon_dataset(data_path)
    
    # Apply strict gaming console filtering
    console_df = filter_gaming_console_reviews(raw_df)
    
    # Verify gaming console content
    verification_samples = verify_gaming_console_content(console_df)
    
    # Map ratings to sentiment
    console_df = map_ratings_to_sentiment(console_df)
    
    # Preprocess text
    console_df = preprocess_dataframe(console_df)
    
    # Balance dataset
    console_df = balance_dataset(console_df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(console_df)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, console_df, verification_samples
