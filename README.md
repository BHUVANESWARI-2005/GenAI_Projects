# 1.News Article Sentiment Analysis with Visualization

## Overview
This project demonstrates a basic sentiment analysis pipeline on a sample dataset of news articles. It includes:
1. **Pre-trained Sentiment Analysis Model**: Using `cardiffnlp/twitter-roberta-base-sentiment-latest` to classify news articles.
  
2. **Interactive Prediction**: Allows users to enter custom news articles and receive real-time sentiment predictions.
   
3. **Data Visualization**: Displays a bar plot of sentiment distribution and a line plot of confidence scores for live predictions.

## Visualization
The project includes two main visualizations:

**Sentiment Distribution Bar Plot:**

A bar chart showing the counts of each sentiment category (Positive, Negative, Neutral) in the dataset.

**Confidence Score Line Plot:**
A line plot visualizing the confidence scores of real-time predictions made on user input.

These visualizations help analyze sentiment trends in news articles and observe confidence levels in live sentiment predictions.

## Required Libraries

To run this project, install the following libraries:
```bash
pip install pandas matplotlib seaborn transformers
```

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
```

# 3.Image Captioning with Vision-Language Model(BLIP)
## Project Overview
This project demonstrates how to build an Image Captioning App using the BLIP (Bootstrapped Language-Image Pre-training) model and Streamlit. Users can upload an image, and the app generates a descriptive caption for the image using the Salesforce BLIP model for conditional image captioning.

## How it works
1.The uploaded image is processed using the BLIPProcessor.

2.The processed image is passed to the BLIPForConditionalGeneration model to generate a caption.

3.The caption is decoded and displayed in the Streamlit app.

## Required Libraries
Install the required libraries using:
```bash
pip install streamlit transformers torch pillow
```

# 4.Text Summarization Using BART

## Project Overview
This project is a Text Summarizer Application that extracts textual content from PDF files and generates a concise summary using state-of-the-art Transformer models. It utilizes Streamlit for an intuitive web-based interface, allowing users to upload PDF documents and receive a textual summary with just a click. The underlying model is based on BART (Bidirectional and Auto-Regressive Transformer), a robust summarization model provided by the Hugging Face Transformers library.

## How it works

**1.Upload the PDF:** Users upload a .pdf file through the Streamlit interface. The application accepts multi-page PDFs and processes them seamlessly.

**2.Text Extraction:** The app uses PyMuPDF to extract text from all pages of the uploaded PDF, focusing on textual content and ignoring non-text elements like images.

**3.Text Summarization:** The extracted text is passed to the BART model (via Hugging Face Transformers), which generates a concise summary. Parameters like maximum and minimum lengths ensure the summary is informative and well-structured.

**4.Display Results:** The app previews the extracted text and displays the summary in a clean interface. Users can review, copy, and use the summary directly.


## Required Libraries
Install the required libraries using:
```bash
pip install streamlit transformers PyMuPDF Pillow
```

# 5.Question Answering System with Dense Retrival and BERT

This project leverages dense retrieval to enable a question-answering system that extracts information from PDF documents. By converting both the document content and user queries into dense vector embeddings using BERT, the system can retrieve the most relevant document segments based on semantic similarity rather than exact keyword matches. The retrieval process is accelerated using FAISS, a powerful similarity search library. This approach ensures efficient and contextually accurate answers, even in large and unstructured text data like PDFs.

## How it works

**1.Dense Retrieval with BERT Embeddings:** The system converts both the PDF content and user queries into dense vector embeddings using BERT. This allows for capturing semantic meaning rather than relying on exact keyword matches, ensuring more relevant and contextually accurate results.

**2.FAISS Index for Efficient Search:** Once the document is transformed into embeddings, a FAISS index is built to store and quickly retrieve the most relevant document segments. FAISS enables fast similarity searches by calculating the distance between query and document embeddings.

**3.Contextual Answer Extraction:** After retrieving the most relevant document snippet, the BERT Question Answering model is used to extract the precise answer from the text. The system uses the start and end logits to pinpoint the answer span within the retrieved content.

**4.Query-Document Matching:** The system performs dense retrieval by comparing the embedding of the user’s question with the document embeddings in the FAISS index. This results in a more intelligent retrieval process, enabling the chatbot to provide accurate, contextually-relevant answers based on the document content.

## Required Libraries

Install the required libraries using:
```bash
pip install torch transformers faiss-cpu PyPDF2
```

# 6.Text-to-Speech System Using Quantized Model

This project demonstrates a simple Text-to-Speech (TTS) system that converts input text into an audio file using a pre-trained quantized TTS model. The model generates natural-sounding speech by synthesizing mel-spectrograms and converting them into audio waveforms. The quantized model used in this project is the Tacotron2-DDC model from the TTS library.It utilizes the Tacotron2-DDC architecture for high-quality speech synthesis. Quantization is a technique applied to deep learning models to reduce their size and speed up inference without significantly sacrificing performance. 

## How it works

**1.Input Text:** Provide a text string to the TTS system.

**2.Model Loading:** The pre-trained Tacotron2-DDC model is loaded and configured for faster processing.

**3.Speech Synthesis:** The input text is processed to produce an audio file in .wav format.

**4.Output**: The synthesized audio file is saved to the specified file path.

The system is designed to be efficient and easy to use, making it ideal for quick speech synthesis tasks.

## Required Libraries

Install the required libraries using:
```bash
pip install TTS transformers soundfile torch
```

# 7.Multilingual Sentiment Analysis Using XLM-RoBERTa

## Project Overview

This project implements a multilingual sentiment analysis system using the XLM-RoBERTa model, a cross-lingual version of the popular RoBERTa architecture. The model is pre-trained on a variety of languages, making it capable of analyzing sentiment across multiple languages without needing separate models for each language.

The project leverages the transformers library by Hugging Face, which provides an easy-to-use interface for loading pre-trained models and performing NLP tasks like sentiment analysis. The system analyzes text inputs in different languages and classifies the sentiment as positive, neutral, or negative.

## How it works

**1.Preprocessing**: The input text is tokenized using the appropriate tokenizer for the XLM-RoBERTa model.

**2.Model Loading**: The pre-trained XLM-RoBERTa model, fine-tuned for multilingual sentiment analysis, is loaded using the Hugging Face library.

**3.Inference**: The model processes the tokenized text and performs sentiment classification (positive, neutral, or negative).

**4.Sentiment Output**: The model returns the predicted sentiment label and a confidence score for each input text.

**5.Multilingual Support**: The model handles texts in various languages including English, Tamil, Spanish, French, German, Japanese, Hindi and more.

## Required Libraries

Install the required libraries using:

```bash
pip install transformers torch
```
