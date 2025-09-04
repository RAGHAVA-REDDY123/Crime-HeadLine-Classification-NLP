# Crime-HeadLine-Classification-NLP & Deep Learning

## Project Overview

This project focuses on classifying crime-related headlines into binary classes using Natural Language Processing (NLP) techniques and various Machine Learning and Deep Learning models.
The objective was to explore how preprocessing and different modeling approaches affect performance.

## ⚙️ Tech Stack

Python

Libraries: NLTK, Scikit-learn, TensorFlow/Keras, Pandas, NumPy

Models: Logistic Regression, Random Forest, LSTM, BiLSTM

## 🛠️ Workflow

Data Preprocessing

Lowercasing text

Tokenization

Removing stopwords & punctuation

Padding sequences

## Feature Engineering

Text to sequences using Tokenizer

Fixed input length with pad_sequences

## Modeling

Logistic Regression → 67% accuracy

Random Forest → 76% accuracy

BiLSTM → 85% accuracy 🎯

## Evaluation Metrics

Accuracy

F1-score

Confusion Matrix

## 📊 Results
Model	Accuracy
Logistic Regression	67%
Random Forest	76%
BiLSTM	85%

## 🚀 Future Improvements

Experiment with Transformers (BERT, DistilBERT)

Hyperparameter tuning for LSTM/BiLSTM

Deployment as a Flask/Streamlit web app
