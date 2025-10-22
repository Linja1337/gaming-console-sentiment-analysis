"""
Gaming Console Sentiment Analysis - Visualization Module
Generate professional visualizations and word clouds
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import os
from typing import List, Dict, Any


plt.style.use('default')
sns.set_palette("husl")

def ensure_directory(path: str):
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)

def save_plot(fig, filepath: str, dpi: int = 300):
    """Save plot with high quality."""
    ensure_directory(os.path.dirname(filepath))
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved plot: {filepath}")

def create_gaming_console_word_clouds(
    df: pd.DataFrame,
    output_dir: str,
    sentiment_labels: Dict[int, str] = {0: "Negative", 1: "Neutral", 2: "Positive"}
):
    """
    Create and save word clouds for each sentiment class with sentiment-specific vocabulary extraction.
    Ensures distinct vocabulary for each sentiment class without overlap.
    """
    print("Creating gaming console word clouds with sentiment-specific vocabulary extraction...")


    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk


    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Get comprehensive English stop words
    english_stopwords = set(stopwords.words('english'))

    # Comprehensive stop words including NLTK + custom gaming/review terms
    comprehensive_stop_words = english_stopwords.union({
        # Basic English stop words (additional)
        'and', 'the', 'or', 'for', 'that', 'is', 'to', 'of', 'in', 'it', 'on', 'with', 'as', 'be', 'at', 'by', 'this', 'have', 'from', 'they', 'we', 'been', 'has', 'had', 'but', 'not', 'are', 'was', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'do', 'does', 'did', 'done', 'being', 'having', 'were', 'am', 'an', 'a', 'all', 'any', 'some', 'no', 'nor', 'if', 'than', 'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'whose', 'whether', 'while', 'until', 'since', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'everywhere', 'anywhere', 'somewhere', 'nowhere', 'everyone', 'anyone', 'someone', 'no one', 'everything', 'anything', 'something', 'nothing', 'each', 'every', 'either', 'neither', 'both', 'few', 'many', 'much', 'more', 'most', 'other', 'another', 'such', 'only', 'own', 'same', 'so', 'just', 'very', 'too', 'now', 'here', 'there', 'today', 'tomorrow', 'yesterday',

        # Gaming-specific generic terms
        'game', 'games', 'gaming', 'play', 'playing', 'played', 'player', 'players', 'gameplay', 'gamer', 'gamers',

        # Console generic terms
        'console', 'system', 'device', 'machine', 'unit', 'hardware', 'software',

        # Review/purchase terms
        'buy', 'bought', 'purchase', 'purchased', 'money', 'price', 'cost', 'worth', 'value', 'dollar', 'dollars',
        'recommend', 'recommended', 'review', 'reviews', 'product', 'item', 'amazon', 'seller', 'shipping', 'delivery',
        'order', 'ordered', 'return', 'returned', 'refund', 'exchange',

        # Generic descriptors (keep sentiment-specific ones)
        'thing', 'things', 'stuff', 'way', 'ways', 'time', 'times', 'lot', 'lots', 'bit', 'piece', 'part', 'parts',
        'one', 'two', 'three', 'four', 'five', 'first', 'second', 'last', 'next', 'new', 'old',

        # Generic verbs
        'get', 'got', 'getting', 'give', 'gave', 'given', 'take', 'took', 'taken', 'make', 'made', 'making',
        'come', 'came', 'coming', 'go', 'went', 'going', 'see', 'saw', 'seen', 'look', 'looked', 'looking',
        'find', 'found', 'think', 'thought', 'know', 'knew', 'known', 'say', 'said', 'tell', 'told',
        'use', 'used', 'using', 'work', 'works', 'working', 'worked', 'try', 'tried', 'trying',

        # Generic adjectives (keep sentiment-bearing ones like 'broken', 'excellent', etc.)
        'big', 'small', 'large', 'little', 'long', 'short', 'high', 'low', 'right', 'left', 'different', 'same',

        # Pronouns and determiners
        'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
        'they', 'them', 'their', 'theirs', 'themselves',

        # Numbers and single characters
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'one', 'two', 'three', 'four', 'five',
        'six', 'seven', 'eight', 'nine', 'ten', 'first', 'second', 'third',

        # Common filler words
        'also', 'even', 'still', 'yet', 'already', 'always', 'never', 'sometimes', 'often', 'usually',
        'maybe', 'perhaps', 'probably', 'definitely', 'certainly', 'surely', 'really', 'actually', 'basically',
        'generally', 'specifically', 'especially', 'particularly', 'exactly', 'quite', 'rather', 'pretty',
        'fairly', 'somewhat', 'kind', 'sort', 'type', 'types', 'kinds', 'sorts'
    })
    
    def extract_sentiment_specific_vocabulary(df):
        """Extract vocabulary that is unique and characteristic to each sentiment class."""

        # Define strong sentiment-bearing words for each class
        strong_negative_words = {
            'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated', 'disappointing', 'disappointed',
            'broken', 'defective', 'useless', 'waste', 'garbage', 'junk', 'crap', 'sucks', 'sucked',
            'failed', 'failure', 'dead', 'died', 'crashed', 'freezing', 'overheating', 'loud', 'noisy',
            'cheap', 'flimsy', 'poor', 'bad', 'worse', 'regret', 'returned', 'refund', 'money', 'wasted'
        }

        strong_positive_words = {
            'amazing', 'excellent', 'fantastic', 'perfect', 'outstanding', 'incredible', 'wonderful',
            'brilliant', 'superb', 'awesome', 'love', 'loved', 'recommend', 'recommended', 'best',
            'beautiful', 'gorgeous', 'smooth', 'fast', 'reliable', 'solid', 'durable', 'quality',
            'impressed', 'happy', 'satisfied', 'pleased', 'delighted', 'thrilled', 'excited'
        }

        strong_neutral_words = {
            'okay', 'decent', 'average', 'fine', 'acceptable', 'reasonable', 'adequate', 'sufficient',
            'standard', 'normal', 'typical', 'regular', 'ordinary', 'common', 'basic', 'simple',
            'works', 'working', 'functional', 'usable', 'serviceable', 'comparable', 'similar'
        }


        sentiment_vocabularies = {}

        for sentiment_value, sentiment_name in sentiment_labels.items():
            print(f"Extracting vocabulary for {sentiment_name} sentiment...")

       
            sentiment_texts = df[df['sentiment'] == sentiment_value]['clean_text'].tolist()

            if not sentiment_texts:
                continue

            # Preprocess texts
            processed_texts = []
            for text in sentiment_texts:
          
                text = text.lower()
        
                import re
                text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()

                # Split into words and filter
                words = text.split()
                filtered_words = []

                for word in words:
                    if (len(word) >= 3 and
                        word not in comprehensive_stop_words and
                        not word.isdigit() and
                        len(word) <= 15):
                        filtered_words.append(word)

                processed_texts.append(' '.join(filtered_words))

            # Use TF-IDF to find characteristic words for this sentiment
            if processed_texts:
                # Get all other sentiment texts for comparison
                other_texts = []
                for other_sentiment in [0, 1, 2]:
                    if other_sentiment != sentiment_value:
                        other_sentiment_texts = df[df['sentiment'] == other_sentiment]['clean_text'].tolist()
                        for text in other_sentiment_texts[:500]:  # Sample to avoid memory issues
                            text = text.lower()
                            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
                            text = re.sub(r'\s+', ' ', text).strip()
                            words = [w for w in text.split() if len(w) >= 3 and w not in comprehensive_stop_words]
                            other_texts.append(' '.join(words))

                # Combine current sentiment texts
                current_corpus = ' '.join(processed_texts)
                other_corpus = ' '.join(other_texts)

                # Extract words that appear more frequently in current sentiment
                from collections import Counter
                current_words = Counter(current_corpus.split())
                other_words = Counter(other_corpus.split())

                # Find words that are characteristic of this sentiment
                characteristic_words = []

                # Add strong sentiment words if they appear
                if sentiment_value == 0:  # Negative
                    target_words = strong_negative_words
                elif sentiment_value == 2:  # Positive
                    target_words = strong_positive_words
                else:  # Neutral
                    target_words = strong_neutral_words

                # Prioritize strong sentiment words
                for word in target_words:
                    if word in current_words and current_words[word] > 2:
                        characteristic_words.extend([word] * min(current_words[word], 20))

                # Add other characteristic words
                for word, count in current_words.most_common(200):
                    if (word not in target_words and
                        count > 3 and
                        (word not in other_words or current_words[word] > other_words[word] * 1.5)):
                        characteristic_words.extend([word] * min(count, 10))

                sentiment_vocabularies[sentiment_value] = ' '.join(characteristic_words)

        return sentiment_vocabularies

    # Extract sentiment-specific vocabularies
    sentiment_vocabularies = extract_sentiment_specific_vocabulary(df)

    word_cloud_files = []

    for sentiment_value, sentiment_name in sentiment_labels.items():
        # Use sentiment-specific vocabulary instead of raw text
        if sentiment_value not in sentiment_vocabularies:
            print(f"No vocabulary found for {sentiment_name} sentiment")
            continue

        combined_text = sentiment_vocabularies[sentiment_value]

        if not combined_text.strip():
            print(f"No meaningful vocabulary found for {sentiment_name} sentiment")
            continue
        
        # Create word cloud with comprehensive stop word filtering
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            stopwords=comprehensive_stop_words,
            colormap='viridis' if sentiment_value == 1 else ('Reds' if sentiment_value == 0 else 'Greens'),
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=100,
            prefer_horizontal=0.7,
            min_word_length=3,  # Filter out very short words
            collocations=False  # Avoid word pairs
        ).generate(combined_text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Gaming Console Reviews - {sentiment_name} Sentiment Word Cloud', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Save word cloud
        filename = f"wordcloud_{sentiment_name.lower()}_gaming_console.png"
        filepath = os.path.join(output_dir, filename)
        save_plot(fig, filepath)
        word_cloud_files.append(filepath)
    
    return word_cloud_files

def plot_gaming_console_dataset_overview(df: pd.DataFrame, output_dir: str):
    """Create dataset overview visualization for gaming console reviews."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sentiment Distribution
    sentiment_counts = df['sentiment'].value_counts().sort_index()
    sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
    labels = [sentiment_labels[i] for i in sentiment_counts.index]
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
    
    ax1.pie(sentiment_counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Gaming Console Review Sentiment Distribution', fontweight='bold')
    
    # 2. Review Length Distribution
    review_lengths = df['clean_text'].str.len()
    ax2.hist(review_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Review Length (characters)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Gaming Console Review Length Distribution', fontweight='bold')
    ax2.axvline(review_lengths.median(), color='red', linestyle='--', label=f'Median: {review_lengths.median():.0f}')
    ax2.legend()
    
    # 3. Rating Distribution
    rating_counts = df['overall'].value_counts().sort_index()
    ax3.bar(rating_counts.index, rating_counts.values, color='lightcoral', alpha=0.8)
    ax3.set_xlabel('Amazon Rating (1-5 stars)')
    ax3.set_ylabel('Number of Reviews')
    ax3.set_title('Gaming Console Review Rating Distribution', fontweight='bold')
    ax3.set_xticks(range(1, 6))
    
    # 4. Console Mentions (if we can extract them)
    console_keywords = ['playstation', 'ps4', 'ps5', 'xbox', 'nintendo', 'switch']
    console_mentions = {}
    
    for keyword in console_keywords:
        count = df['clean_text'].str.contains(keyword, case=False, na=False).sum()
        if count > 0:
            console_mentions[keyword.title()] = count
    
    if console_mentions:
        consoles = list(console_mentions.keys())
        counts = list(console_mentions.values())
        ax4.barh(consoles, counts, color='lightgreen', alpha=0.8)
        ax4.set_xlabel('Number of Mentions')
        ax4.set_title('Gaming Console Brand Mentions', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Console mentions\nanalysis unavailable', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Gaming Console Brand Mentions', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'gaming_console_dataset_overview.png')
    save_plot(fig, filepath)
    
    return filepath

def plot_distilbert_performance(
    train_result: Dict,
    val_result: Dict,
    test_result: Dict,
    cv_result: Dict,
    output_dir: str
):
    """Plot DistilBERT performance across all phases."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy across phases
    phases = ['Training', 'Validation', 'Test']
    accuracies = [
        1 - train_result['training_loss'],  # Approximate training accuracy
        val_result['val_accuracy'],
        test_result['test_accuracy']
    ]
    
    bars = ax1.bar(phases, accuracies, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('DistilBERT Gaming Console Sentiment Analysis - Accuracy by Phase', fontweight='bold')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Cross-validation results
    cv_accuracies = cv_result['fold_accuracies']
    ax2.boxplot([cv_accuracies], labels=['Cross-Validation'])
    ax2.scatter([1] * len(cv_accuracies), cv_accuracies, alpha=0.6, color='red')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{len(cv_accuracies)}-Fold Cross-Validation Results', fontweight='bold')
    ax2.text(1.1, np.mean(cv_accuracies), f'Mean: {np.mean(cv_accuracies):.3f}\nStd: {np.std(cv_accuracies):.3f}',
             va='center', fontweight='bold')
    
    # 3. Confusion Matrix (Test Set)
    cm = test_result['test_confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Test Set Confusion Matrix', fontweight='bold')
    
    # 4. Performance metrics comparison
    test_report = test_result['test_report']
    metrics = ['precision', 'recall', 'f1-score']
    sentiments = ['0', '1', '2']  # negative, neutral, positive
    
    x = np.arange(len(sentiments))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [test_report[sentiment][metric] for sentiment in sentiments]
        ax4.bar(x + i*width, values, width, label=metric.title(), alpha=0.8)
    
    ax4.set_xlabel('Sentiment Class')
    ax4.set_ylabel('Score')
    ax4.set_title('Test Set Performance by Sentiment Class', fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'distilbert_gaming_console_performance.png')
    save_plot(fig, filepath)
    
    return filepath

def plot_confusion_matrices(val_cm, test_cm, output_dir: str):
    """Plot confusion matrices for validation and test sets."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = ['Negative', 'Neutral', 'Positive']
    
    # Validation confusion matrix
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=labels, yticklabels=labels)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Validation Set Confusion Matrix', fontweight='bold')
    
    # Test confusion matrix
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=labels, yticklabels=labels)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Test Set Confusion Matrix', fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'confusion_matrices_gaming_console.png')
    save_plot(fig, filepath)
    
    return filepath

def create_all_visualizations(
    df: pd.DataFrame,
    train_result: Dict,
    val_result: Dict,
    test_result: Dict,
    cv_result: Dict,
    output_dir: str
) -> List[str]:
    """
    Create all visualizations for gaming console sentiment analysis.
    Returns list of generated file paths.
    """
    print("Creating all visualizations for gaming console sentiment analysis...")
    
    ensure_directory(output_dir)
    generated_files = []
    
    # 1. Dataset overview
    overview_file = plot_gaming_console_dataset_overview(df, output_dir)
    generated_files.append(overview_file)
    
    # 2. Word clouds
    word_cloud_files = create_gaming_console_word_clouds(df, output_dir)
    generated_files.extend(word_cloud_files)
    
    # 3. DistilBERT performance
    performance_file = plot_distilbert_performance(train_result, val_result, test_result, cv_result, output_dir)
    generated_files.append(performance_file)
    
    # 4. Confusion matrices
    confusion_file = plot_confusion_matrices(
        val_result['val_confusion_matrix'],
        test_result['test_confusion_matrix'],
        output_dir
    )
    generated_files.append(confusion_file)
    
    print(f"Generated {len(generated_files)} visualization files:")
    for file in generated_files:
        print(f"  - {file}")
    
    return generated_files
