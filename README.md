# Machine-Learning
# Video Game Comments Classification

## Problem Description

The objective of this project is to classify video game comments as positive or negative using machine learning techniques. This automatic classification can be useful for developing more effective marketing strategies and making decisions based on user feedback.

## Dataset

The dataset used in this project contains user comments on various video games, classified as positive or negative. The main column of interest is `Contenido`, which contains the text comments, and `Recomendado_binario`, which indicates whether the comment is positive (`1`) or negative (`0`).

### Dataset Description

- **Dataset Name**: dataset_reviews_cleaned.csv
- **Columns**:
  - `Contenido`: User comments.
  - `Recomendado_binario`: Binary indicator (1 = Positive, 0 = Negative).
- **Source**: This dataset is private and not publicly available.

## Solution Adopted

The solution implemented consists of several key steps:

1. **Data Preprocessing**:
   - Converting all texts to lowercase.
   - Removing stopwords and applying lemmatization.
   - Detecting the language to apply specific preprocessing for English and Spanish.

2. **Vectorization**:
   - Converting preprocessed text into numerical vectors using TF-IDF.

3. **Modeling**:
   - Training various machine learning models, including:
     - Logistic Regression
     - Random Forest
     - SVM
   - Hyperparameter tuning using `GridSearchCV`.
   - Model ensemble with a Voting Classifier.

4. **Evaluation**:
   - Evaluating models using cross-validation.
   - Final evaluation of the best model on the test set.

## Directory Structure

The repository is organized as follows:

