# News Article Sentiment Analysis with Visualization

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
```bash
pip install pandas matplotlib seaborn transformers
