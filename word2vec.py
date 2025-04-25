from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import spacy
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz

# Initialize Flask app
app = Flask(__name__)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load and preprocess dataset
df = pd.read_excel("dataset.xlsx").dropna()

def clean_text(text):
    doc = nlp(str(text).lower())
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

df["clean_question"] = df["Question"].apply(clean_text)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Train Word2Vec on training questions
sentences = train_df["clean_question"].tolist()
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Function to get vector representation of a sentence
def get_sentence_vector(tokens):
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(w2v_model.vector_size)

# Main matching function
def get_answer(user_input, dataset):
    cleaned_input = clean_text(user_input)
    input_vector = get_sentence_vector(cleaned_input).reshape(1, -1)

    max_score = 0
    best_match = None

    for _, row in dataset.iterrows():
        candidate_vector = get_sentence_vector(row["clean_question"]).reshape(1, -1)
        score = cosine_similarity(input_vector, candidate_vector)[0][0]
        if score > max_score:
            max_score = score
            best_match = row

    if max_score < 0.1:
        return None, "🙁 Sorry, I can't help with the question.", max_score

    return best_match["Question"], best_match["Answer"], max_score

# Flask endpoint
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("text")
    username = request.form.get("user_name", "user")

    if not user_input:
        return jsonify({"response_type": "ephemeral", "text": "❗ Please enter a question."})

    matched_q, matched_a, score = get_answer(user_input, train_df)

    return jsonify({
        "response_type": "in_channel",
        "text": (
            f"🧠 **Question by <@{username}>:** {user_input}\n\n"
            f"🔍 **Closest Match:** {matched_q or 'No good match'}\n"
            f"💬 **Answer:** {matched_a}"
        )
    })

# Evaluation function
def evaluate_model():
    correct = 0
    total = len(test_df)

    for _, row in test_df.iterrows():
        true_answer = row["Answer"]
        _, predicted_answer, _ = get_answer(row["Question"], train_df)
        if predicted_answer != "🙁 Sorry, I can't help with the question.":
            if fuzz.token_set_ratio(true_answer.lower(), predicted_answer.lower()) >= 75:
                correct += 1

    accuracy = correct / total
    print(f"\n📊 Evaluation Accuracy: {correct}/{total} = {accuracy:.2f}")

# Entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        evaluate_model()
        while True:
            q = input("\nAsk your question (or type 'exit'): ")
            if q.lower() == "exit":
                break
            matched_q, matched_a, score = get_answer(q, train_df)
            print(f"\n🔍 Closest Match: {matched_q or 'N/A'}")
            print(f"💬 Answer: {matched_a}")
            print(f"📈 Score: {score:.2f}")
    else:
        app.run(port=5000)
