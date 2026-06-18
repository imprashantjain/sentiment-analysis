import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("sentiment_model.keras")

word_index = tf.keras.datasets.imdb.get_word_index()

st.title("🎬 Movie Review Sentiment Analysis")

review = st.text_area("Enter Movie Review")

if st.button("Predict"):

    tokens = []

    for word in review.lower().split():
        if word in word_index:
            tokens.append(word_index[word] + 3)

    padded = pad_sequences(
        [tokens],
        maxlen=250
    )

    prediction = model.predict(padded)

    score = prediction[0][0]

    st.write(f"Confidence Score: {score:.4f}")

    if score > 0.5:
        st.success("Positive Review 🫠")
    else:
        st.error("Negative Review 🥲 ")