# 1.News Article Sentiment Analysis with Visualization

## Overview
This project demonstrates a basic sentiment analysis pipeline on a sample dataset of news articles. It includes:
1. **Pre-trained Sentiment Analysis Model**: Using `cardiffnlp/twitter-roberta-base-sentiment-latest` to classify news articles.
2. **Interactive Prediction**: Allows users to enter custom news articles and receive real-time sentiment predictions.
3. **Data Visualization**: Displays a bar plot of sentiment distribution and a line plot of confidence scores for live predictions.

## Visualization
The project includes two main visualizations:

Sentiment Distribution Bar Plot:

A bar chart showing the counts of each sentiment category (Positive, Negative, Neutral) in the dataset.

Confidence Score Line Plot:
A line plot visualizing the confidence scores of real-time predictions made on user input.

These visualizations help analyze sentiment trends in news articles and observe confidence levels in live sentiment predictions.

## Required Libraries

To run this project, install the following libraries:

-pip install pandas matplotlib seaborn transformers


# 2.ChatBot Using Retrieval-Augmented Generation (RAG)

## Overview
The Machine Learning AI Chatbot leverages a Retrieval-Augmented Generation (RAG) framework to provide precise answers to machine learning-related queries. It uses DistilBERT (a distilled version of BERT) for question answering and FAISS for efficient similarity search over document chunks. The chatbot processes large datasets, retrieves contextually relevant information, and generates accurate answers in real-time.

## Pre-trained models used:
**DistilBERT**: A distilled version of BERT used for question answering.



**Sentence Transformer**: Used for generating embeddings of document chunks

## Required Libraries
Install the required libraries using:

```bash
pip install pymupdf sentence-transformers faiss-cpu transformers
