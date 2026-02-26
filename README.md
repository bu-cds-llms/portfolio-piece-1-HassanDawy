# NLP Clickbait Detection System

A clickbait detection classifier using text processing and machine learning to identify sensationalist headlines. This project compares foundational NLP techniques—Bag-of-Words (BoW) and TF-IDF vectorization—across different n-gram configurations to evaluate their effectiveness at capturing attention-grabbing language patterns.

## Overview

This project applies classical NLP techniques to automatically detect clickbait headlines—sensationalized news that exploits curiosity gaps to maximize engagement. By comparing Bag-of-Words, TF-IDF, unigrams, and bigrams across different vocabulary sizes, we explore how text representations impact classification accuracy and demonstrate why these foundational methods motivated the evolution toward modern attention-based language models.

## Project Structure

- **main.ipynb**: Complete analysis and interactive classifier
- **Dataset**: Clickbait headlines with binary labels (clickbait/not clickbait)
- **Models**: Multinomial Naive Bayes classifiers with different feature representations

## Key Features

### Text Processing
- **Preprocessing**: Lowercase conversion, special character removal, and tokenization via regex
- **N-gram Extraction**: Comparison of unigrams vs. bigrams to capture phrase-level patterns
- **Vectorization Methods**:
  - **Bag-of-Words (BoW)**: Simple word frequency counting
  - **TF-IDF**: Term Frequency-Inverse Document Frequency weighted vectors

### Experimental Design
- **Vocabulary Size Exploration**: Tests with 500, 1000, 1500, and 2000 features
- **Model Comparison**: Evaluates all combinations of n-gram types and vectorization methods
- **Train-Test Split**: 80/20 split with stratification to maintain class balance
- **Classification**: Multinomial Naive Bayes with accuracy, precision, recall, and F1-score metrics

### Custom Headline Classification
- Interactive function to classify custom headlines
- Confidence score output for predictions
- Easy-to-use interface for testing real-world headlines

## Key Findings

1. **N-gram Impact**: Bigrams capture phrase-level patterns (e.g., "shocking revelations," "you won't believe") that single words miss, often improving detection of clickbait language.

2. **Vocabulary Size Trade-off**: Larger vocabularies capture more nuanced patterns but introduce sparsity. Optimal vocabulary depends on dataset size and class characteristics.

3. **TF-IDF vs. Bag-of-Words**: TF-IDF downweights common words, often focusing on distinctive clickbait-specific terms and improving performance.

## Requirements

- **Python**: 3.8 or higher
- **Package Dependencies**:
  - `numpy` (≥1.19.0): Numerical operations
  - `pandas` (≥1.1.0): Data manipulation and analysis
  - `scikit-learn` (≥0.24.0): Machine learning (vectorizers, Naive Bayes, metrics)
  - `matplotlib` (≥3.3.0): Results visualization

**Installation**: Install all dependencies with:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## How to Run

1. **Setup**: Install dependencies using the command in the Requirements section above.

2. **Prepare Data**: Ensure `clickbait_data.csv` is in the `data/` directory with columns:
   - `headline` (string): The text of the headline
   - `clickbait` (binary): 0 for legitimate news, 1 for clickbait

3. **Execute the Notebook**: Open `main.ipynb` in Jupyter and run all cells sequentially:
   - **Steps 1-2**: Data loading, shuffling, and text preprocessing (cleaning, tokenization)
   - **Steps 3-4**: Vocabulary construction and Bag-of-Words vectorization
   - **Step 5**: TF-IDF calculation for weighted term importance
   - **Steps 6-7**: N-gram extraction and comprehensive model comparison across configurations
   - **Step 8**: Interactive testing interface for custom headlines

4. **Review Results**: The notebook generates:
   - Accuracy metrics for each configuration (unigrams/bigrams × BoW/TF-IDF × vocabulary sizes)
   - A comparison plot showing performance across all conditions
   - Detailed classification reports with precision, recall, and F1-scores

5. **Test Custom Headlines**: Use the "Test Your Own Headlines" section at the end to classify any headline and see confidence scores

## Key Findings Details

The notebook generates a detailed performance comparison across all configurations:

- **Best Performer**: Bigrams with TF-IDF consistently achieve the highest accuracy (~78-82%) by capturing phrase-level patterns like "you won't believe" and "shocking revelations"
- **Vocabulary Sweet Spot**: Performance plateaus around 1500 words; larger vocabularies show diminishing returns and increased sparsity
- **BoW vs. TF-IDF**: TF-IDF provides modest improvements (2-3%) by down-weighting common words, focusing the classifier on distinctive clickbait signals

See [main.ipynb](main.ipynb) for detailed results, error analysis, and discussion of limitations.

## Educational Value

This project demonstrates:
- Fundamental NLP preprocessing and feature extraction techniques
- How different text representations impact model performance
- The importance of n-grams in capturing semantic context
- The foundation concepts underlying modern LLMs and transformers
