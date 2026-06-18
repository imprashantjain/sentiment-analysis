# Sentiment Analysis using TensorFlow and LSTM

## Overview

This project is an end-to-end Sentiment Analysis application built using TensorFlow and Keras. The model is trained on the IMDB Movie Reviews dataset and predicts whether a movie review is positive or negative.

The project demonstrates key Natural Language Processing (NLP) concepts including:

* Tokenization
* Padding
* Word Embeddings
* LSTM Networks
* Binary Classification
* Model Deployment

---

## Dataset

Dataset: IMDB Movie Reviews

* 25,000 training reviews
* 25,000 testing reviews
* Binary labels:

  * Positive (1)
  * Negative (0)

---

## Model Architecture

Input Review

↓

Padding (250 Tokens)

↓

Embedding Layer (10000 Vocabulary, 16 Dimensions)

↓

LSTM Layer (32 Units)

↓

Dense Layer (Sigmoid Activation)

↓

Sentiment Prediction

### Model Summary

* Embedding Parameters: 160,000
* LSTM Parameters: 6,272
* Dense Parameters: 33
* Total Parameters: 166,305

---

## Results

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 91.88% |
| Validation Accuracy | 86.40% |
| Validation Loss     | 0.3287 |

---

## Project Structure

sentiment-analysis/

├── app.py

├── sentiment_model.keras

├── requirements.txt

├── README.md

├── src/

│   ├── train.py

│   ├── predict.py

│   ├── embedding_demo.py

│   ├── padding_demo.py

│   └── model_architecture.py

└── notebooks/

---

## Installation

Clone the repository:

git clone https://github.com/your-username/sentiment-analysis.git

cd sentiment-analysis

Install dependencies:

pip install -r requirements.txt

---

## Training

Run:

python src/train.py

The model will train and save:

sentiment_model.keras

---

## Prediction

Run:

python src/predict.py

Example:

Input:
"This movie was absolutely fantastic"

Output:
Positive Sentiment

---

## Skills Demonstrated

* Deep Learning
* Natural Language Processing
* TensorFlow
* Keras
* LSTM Networks
* Text Preprocessing
* Model Deployment
* Git & GitHub

---

## Future Improvements

* Bidirectional LSTM
* GRU Networks
* Attention Mechanism
* Transformer Models
* BERT Fine-Tuning
* Streamlit Deployment

---

## Author

Prashant Jain

Aspiring AI Engineer passionate about Machine Learning, Deep Learning, NLP, and Generative AI.
