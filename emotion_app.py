EMOJI_DICT = {
    'anger': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'joy': '😊',
    'love': '❤️',
    'sadness': '😢',
    'surprise': '😲'
}

import pandas as pd
import streamlit as st
# Load dataset
def load_data(path):
    data = pd.read_csv(path, sep=';', header=None, names=['text', 'emotion'])
    return data

# Load the training data
data = load_data("data/train.txt")

# Show basic info
print("Sample data:")
print(data.head())
print("\nLabel distribution:")
print(data['emotion'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Split into X (text) and y (emotion)
X = data['text']
y = data['emotion']

# Convert text into numerical features
vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)

# Split into train and test (you can skip if using full train.txt)
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel performance:\n")
print(classification_report(y_test, y_pred))

@st.cache_resource

def train_emotion_model():
    data = pd.read_csv("data/train.txt", sep=';', header=None, names=['text', 'emotion'])
    X = data['text']
    y = data['emotion']
    vectorizer = CountVectorizer()
    X_vect = vectorizer.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vect, y)
    return model, vectorizer

model, vectorizer = train_emotion_model()
st.markdown("<h1 style='text-align: center; color: #ff8da1;'> Welcome to the Emotion Detector </h1>", unsafe_allow_html=True)

st.sidebar.title("🌸 Welcome!")
st.sidebar.write("This is a cute little app that detects emotions from the text you type 🧠💬")
st.sidebar.write("Try typing how you feel!")

st.sidebar.image("image.jpeg", caption="...", use_container_width=True)

st.title("🧠 Emotion Detector from Text")
st.write("Type a sentence below to find out what emotion it contains:")

user_input = st.text_area("Your text here 💬", height=100)



if st.button("Detect Emotion"):
    if user_input.strip() != "":
        input_vect = vectorizer.transform([user_input])
        prediction = model.predict(input_vect)[0]

        st.markdown(f"### Predicted Emotion: {EMOJI_DICT.get(prediction, '')} **{prediction.upper()}**")

        # Show probabilities
        probs = model.predict_proba(input_vect)[0]
        labels = model.classes_

        # Bar chart
        st.subheader("Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(labels, probs, color='skyblue')
        ax.set_ylabel("Probability")
        ax.set_xlabel("Emotion")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("Please enter some text to analyze.")


