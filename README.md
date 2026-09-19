# Movie Review Sentiment Analysis

This project analyzes Amazon product reviews and predicts whether a review is **Positive** or **Negative** using Machine Learning.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes

## Dataset

The dataset contains **25,000 Amazon product reviews** with ratings from 1 to 5.

- Ratings 1–3 → Negative
- Ratings 4–5 → Positive

## Machine Learning

TF-IDF is used to convert review text into numerical features, and Multinomial Naive Bayes is used for sentiment classification.

The model achieved **79.90% test accuracy**.

## Web Application

The project is deployed using Streamlit and allows users to enter a review and receive a Positive or Negative sentiment prediction.

## Live Demo

https://movie-review-sentiment-analysis-iwr3ewocjhovdo3cadyvdk.streamlit.app/
