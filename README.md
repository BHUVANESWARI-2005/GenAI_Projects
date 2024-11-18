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

-pip install pymupdf sentence-transformers faiss-cpu transformers

# 3.Image Captioning with Vision-Language Model(BLIP)
## Project Overview
This project demonstrates how to build an Image Captioning App using the BLIP (Bootstrapped Language-Image Pre-training) model and Streamlit. Users can upload an image, and the app generates a descriptive caption for the image using the Salesforce BLIP model for conditional image captioning.

## How it works
1.The uploaded image is processed using the BLIPProcessor.

2.The processed image is passed to the BLIPForConditionalGeneration model to generate a caption.

3.The caption is decoded and displayed in the Streamlit app.
## Required Libraries
Install the required libraries using:

-pip install streamlit transformers torch pillow


