# NLP Clickbait Detection System

A clickbait detection classifier using text processing and machine learning to identify sensationalist headlines. This project compares foundational NLP techniques—Bag-of-Words (BoW) and TF-IDF vectorization—across different n-gram configurations to evaluate their effectiveness at capturing attention-grabbing language patterns.

## Overview

This implementation demonstrates how simple word-counting approaches form the foundation for understanding modern NLP, by exploring how unigrams (single words) versus bigrams (word pairs) and varying vocabulary sizes impact classification accuracy. The project illustrates the fundamental transition from basic word counting to the context-aware sequence modeling used in contemporary language models.

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

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas`: Data manipulation and analysis
  - `scikit-learn`: Machine learning (vectorizers, Naive Bayes, metrics)
  - `numpy`: Numerical operations
  - `matplotlib`: Results visualization
  - `re`: Regular expression text preprocessing

## How to Use

1. Ensure you have `clickbait_data.csv` in the project directory with columns: `headline` and `clickbait`
2. Run all cells in `main.ipynb` to:
   - Load and preprocess the dataset
   - Build custom vocabulary and vectorization functions
   - Train and evaluate BoW and TF-IDF models
   - Compare performance across different configurations
   - Visualize results
3. Use the "Test Your Own Headlines" section to classify custom headlines

## Model Performance

The notebook generates a performance comparison across all configurations, showing how accuracy varies with:
- N-gram types (unigrams vs. bigrams)
- Vectorization methods (BoW vs. TF-IDF)
- Vocabulary sizes (500-2000 features)

Results are visualized in a comprehensive plot for easy comparison.

## Educational Value

This project demonstrates:
- Fundamental NLP preprocessing and feature extraction techniques
- How different text representations impact model performance
- The importance of n-grams in capturing semantic context
- The foundation concepts underlying modern LLMs and transformers
