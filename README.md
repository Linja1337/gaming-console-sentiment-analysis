# Gaming Console Sentiment Analysis with DistilBERT

Fine-tuned transformer model for 3-class sentiment classification on Amazon gaming console reviews, achieving 70.2% test accuracy with robust cross-validation performance.

---

## Overview

End-to-end NLP pipeline implementing DistilBERT fine-tuning for sentiment analysis of gaming console hardware reviews. Processed 497,577 Amazon reviews, applied strict domain filtering to extract 13,587 gaming console-specific samples, and trained a production-ready classification model.

**Key Results:**
- 70.2% test accuracy on 3-class sentiment classification (Negative/Neutral/Positive)
- 5-fold cross-validation: 70.7% ± 1.3% accuracy
- Strong negative detection: 79.8% recall (critical for quality assurance)
- Strong positive detection: 83.8% recall (valuable for marketing insights)
- Balanced dataset: 13,587 samples across PlayStation, Xbox, and Nintendo ecosystems

---

## Technical Implementation

### Model Architecture
- **Base Model:** distilbert-base-uncased (Hugging Face Transformers)
- **Fine-tuning:** Custom PyTorch dataset class with stratified train/val/test split
- **Training:** AdamW optimizer, learning rate 2e-5, early stopping, 3 epochs
- **Hardware:** CUDA acceleration (NVIDIA RTX 4060)

### Data Pipeline
1. **Data Ingestion:** Loaded 497,577 reviews from Amazon Video Games dataset
2. **Domain Filtering:** Applied strict gaming console keyword filtering (7.84% retention rate)
3. **Preprocessing:** Text cleaning, sentiment mapping (5-star to 3-class), class balancing
4. **Splitting:** Stratified 70/10/20 train/validation/test split

### Key Features
- Domain-specific filtering with hardware context validation
- Comprehensive text preprocessing (HTML removal, URL cleaning, punctuation handling)
- Class balancing to prevent bias (maximum 5,000 samples per class)
- Cross-validation for robust performance estimation
- Extensive evaluation metrics (precision, recall, F1-score, confusion matrices)

---

## Performance Metrics

### Model Performance

| Phase | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|-------|----------|-------------------|----------------|------------------|
| Validation | 70.4% | 69.2% | 70.4% | 69.2% |
| Test | 70.2% | 69.1% | 70.2% | 69.1% |
| Cross-Validation | 70.7% ± 1.3% | - | - | 69.9% ± 1.3% |

### Per-Class Performance (Test Set)

| Sentiment | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Negative | 70.3% | 79.8% | 74.7% | 897 |
| Neutral | 59.1% | 43.1% | 49.9% | 821 |
| Positive | 76.2% | 83.8% | 79.8% | 1,000 |

**Analysis:** High recall for negative sentiment (79.8%) enables effective identification of customer complaints and hardware issues. Positive sentiment detection (83.8% recall) provides reliable signals for satisfied customers. Lower performance on neutral class (49.9% F1) reflects the inherent difficulty of detecting moderate opinions.

---

## Business Value

### Applications
- **Quality Assurance:** Automated detection of hardware failures, overheating, and performance issues
- **Product Development:** Identification of features driving positive/negative sentiment
- **Competitive Intelligence:** Brand-specific sentiment analysis across PlayStation, Xbox, Nintendo
- **Marketing Optimization:** Data-driven messaging based on customer satisfaction drivers

### Dataset Insights
- Gaming console market represented across multiple generations (PS2-PS5, Xbox 360-Series, Wii-Switch)
- Balanced sentiment distribution prevents positive bias common in e-commerce reviews
- Average review length: 1,716 characters (sufficient context for sentiment analysis)

---

## Technical Stack

**Core Technologies:**
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- pandas, numpy
- scikit-learn
- NLTK

**Model & Training:**
- DistilBERT (66M parameters)
- Custom PyTorch Dataset implementation
- Hugging Face Trainer API with early stopping
- Stratified K-fold cross-validation

**Analysis & Visualization:**
- matplotlib, seaborn
- Confusion matrices
- Word clouds (sentiment-specific vocabulary)
- Classification reports

---

## Project Structure

```
gaming_console_sentiment_analysis/
│
├── src/
│   ├── data_preprocessing.py      # Data loading, filtering, cleaning, splitting
│   ├── model_training.py          # DistilBERT training, evaluation, cross-validation
│   ├── visualization.py           # Plots, word clouds, performance metrics
│   └── main.py                    # Main execution pipeline
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── Gaming_Console_Sentiment_Analysis_Report.md  # Comprehensive analysis report
```

---

## Dataset Methodology

### Filtering Criteria
**Required Keywords (Primary):**
- PlayStation: ps2, ps3, ps4, ps5, playstation
- Xbox: xbox, xbox 360, xbox one, xbox series
- Nintendo: nintendo switch, wii, wii u

**Required Keywords (Hardware Context):**
- Technical terms: console, system, hardware, controller, hdmi, ssd, storage

**Exclusion Terms:**
- Non-gaming devices: alexa, kindle, tablet, phone
- Software-only content: pc game, board game, card game

### Final Dataset Composition
- Total samples: 13,587
- Negative (1-2 stars): 4,483 (33.0%)
- Neutral (3 stars): 4,104 (30.2%)
- Positive (4-5 stars): 5,000 (36.8%)
- Train/Val/Test: 9,510 / 1,359 / 2,718

---

## Running the Pipeline

### Installation
```bash
pip install -r requirements.txt
```

### Execution
```bash
python src/main.py
```

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- Amazon Video Games dataset (Video_Games_5.json.gz)

---

## Key Takeaways

This project demonstrates end-to-end NLP pipeline development with modern transformer architectures. The implementation showcases:
- LLM fine-tuning with Hugging Face Transformers
- Large-scale data processing (500K+ reviews)
- Domain-specific filtering and data quality assurance
- Statistical rigor through cross-validation
- Business-oriented problem framing and actionable insights
- Professional code organization and documentation
