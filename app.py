import streamlit as st
import pickle

# Load the saved model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Prediction function without any text preprocessing
def predict_sentiment(text):
    vector = vectorizer.transform([text])  # Use raw text directly
    prediction = model.predict(vector)
    return "😊 Positive" if prediction[0] == 1 else "😞 Negative"

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Analyzer", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Analyzer")

st.write("Type a movie review below and the app will predict the sentiment!")

# Text input
user_input = st.text_area("Enter a movie review:")

# Predict button
if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a valid movie review.")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {result}")