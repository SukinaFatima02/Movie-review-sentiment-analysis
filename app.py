import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to:",
    ["Home", "Review Prediction", "About Project"]
)


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv("AmazonReview.csv")


# ==================================================
# REMOVE MISSING VALUES
# ==================================================

df = df.dropna(subset=["Review", "Sentiment"])


# ==================================================
# CREATE BINARY SENTIMENT
# ==================================================
# Ratings 1, 2, 3 = Negative
# Ratings 4, 5 = Positive
#
# 0 = Negative
# 1 = Positive
# ==================================================

df["SentimentLabel"] = df["Sentiment"].apply(
    lambda x: 0 if x <= 3 else 1
)


# ==================================================
# DATA FOR MACHINE LEARNING
# ==================================================

X = df["Review"]
y = df["SentimentLabel"]


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# CREATE MACHINE LEARNING MODEL
# ==================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=5000,
            stop_words="english"
        )
    ),
    (
        "classifier",
        MultinomialNB()
    )
])


# ==================================================
# TRAIN MODEL
# ==================================================

model.fit(X_train, y_train)


# ==================================================
# TEST MODEL
# ==================================================

y_pred = model.predict(X_test)

accuracy_score_value = accuracy_score(
    y_test,
    y_pred
)


# ==================================================
# REVIEW COUNTS
# ==================================================

# Original CSV contains 25,000 records
total_reviews = 25000

negative_reviews = (df["Sentiment"] <= 3).sum()
positive_reviews = (df["Sentiment"] >= 4).sum()


# ==================================================
# HOME PAGE
# ==================================================

if page == "Home":

    st.title("🎬 Movie Review Sentiment Analysis")

    st.write(
        "Analyze Amazon product reviews and predict whether the review "
        "is Positive or Negative using Machine Learning."
    )

    # -----------------------------
    # Dataset Overview
    # -----------------------------

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Reviews",
        total_reviews
    )

    col2.metric(
        "Positive Reviews",
        positive_reviews
    )

    col3.metric(
        "Negative Reviews",
        negative_reviews
    )

    col4.metric(
        "Test Accuracy",
        f"{accuracy_score_value * 100:.2f}%"
    )


    # -----------------------------
    # Review Data
    # -----------------------------

    st.subheader("📋 Review Data")

    st.write(
        "Sample reviews from the Amazon dataset:"
    )

    st.dataframe(
        df[["Review", "Sentiment"]].head(10),
        use_container_width=True
    )


    # -----------------------------
    # Sentiment Distribution
    # -----------------------------

    st.subheader("📈 Sentiment Distribution")

    sentiment_summary = pd.Series({
        "Negative": negative_reviews,
        "Positive": positive_reviews
    })

    st.bar_chart(
        sentiment_summary
    )


    # -----------------------------
    # Sentiment Summary
    # -----------------------------

    st.subheader("📊 Sentiment Summary")

    col1, col2 = st.columns(2)

    col1.metric(
        "Negative Reviews",
        negative_reviews
    )

    col2.metric(
        "Positive Reviews",
        positive_reviews
    )


# ==================================================
# REVIEW PREDICTION PAGE
# ==================================================

if page == "Review Prediction":

    st.title("🔍 Review Sentiment Prediction")

    st.info(
        "Enter a movie or product review below to predict whether "
        "it is Positive or Negative."
    )


    review = st.text_area(
        "Write your review here:",
        placeholder=(
            "Example: This movie was absolutely fantastic! "
            "I really enjoyed it."
        )
    )


    if st.button("Predict Sentiment"):

        if review.strip() == "":
            
            st.warning(
                "Please enter a review."
            )

        else:

            st.write("**Your Review:**")

            st.write(review)


            # Predict sentiment
            prediction = int(
                model.predict([review])[0]
            )


            # -----------------------------
            # Display Result
            # -----------------------------

            if prediction == 1:

                st.success(
                    "✅ Positive Review"
                )

                st.write(
                    "The review appears to express a positive sentiment."
                )

            else:

                st.error(
                    "❌ Negative Review"
                )

                st.write(
                    "The review appears to express a negative sentiment."
                )


# ==================================================
# ABOUT PROJECT PAGE
# ==================================================

if page == "About Project":

    st.title("📌 About This Project")


    st.write(
        "This project analyzes Amazon product reviews using "
        "Natural Language Processing and Machine Learning techniques."
    )


    st.write(
        "The original dataset contains sentiment ratings from 1 to 5."
    )


    st.write(
        "For sentiment classification, ratings 1–3 are treated "
        "as Negative and ratings 4–5 are treated as Positive."
    )


    # -----------------------------
    # Technologies
    # -----------------------------

    st.subheader("🛠️ Technologies Used")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• Pandas")
    st.write("• Scikit-learn")
    st.write("• TF-IDF")
    st.write("• Naive Bayes")
    st.write("• Natural Language Processing")


    # -----------------------------
    # Machine Learning
    # -----------------------------

    st.subheader("🤖 Machine Learning Method")

    st.write(
        "TF-IDF is used to convert review text into numerical "
        "features. Multinomial Naive Bayes is then used to "
        "classify the review as Positive or Negative."
    )


    # -----------------------------
    # Model Evaluation
    # -----------------------------

    st.subheader("📊 Model Evaluation")

    st.write(
        "The dataset is divided into 80% training data and "
        "20% testing data."
    )

    st.write(
        f"Test Accuracy: {accuracy_score_value * 100:.2f}%"
    )


    # -----------------------------
    # Dataset
    # -----------------------------

    st.subheader("📚 Dataset")

    st.write(
        "The dataset contains 25,000 Amazon product review records."
    )

    st.write(
        "Ratings 1, 2 and 3 are classified as Negative, while "
        "ratings 4 and 5 are classified as Positive."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.write(
    "🎬 Movie Review Sentiment Analysis | Machine Learning Project"
)

st.write(
    "Built using Python, Streamlit, Pandas and Naive Bayes"
)