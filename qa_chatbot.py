 
# 🧠 Offline Q&A Chatbot Using NLP (No API Required)

# 📦 Install dependencies (uncomment if running for the first time)
# !pip install pandas scikit-learn nltk openpyxl

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 📁 Load Excel File
df = pd.read_excel("dataset.xlsx")
df = df.dropna()

# ✨ Clean and Preprocess Questions
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_question'] = df['Question'].apply(clean_text)

# 🔢 Vectorize with TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_question'])

# 🤖 Define Answer Retrieval Function
def get_answer(user_input, top_k=1):
    user_input_clean = clean_text(user_input)
    user_vec = vectorizer.transform([user_input_clean])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[::-1][:top_k]
    best_match_index = top_indices[0]

    return {
        "Your Question": user_input,
        "Matched Question": df.iloc[best_match_index]['Question'],
        "Answer": df.iloc[best_match_index]['Answer'],
        "Confidence Score": cosine_similarities[best_match_index]
    }

# 💬 Test the Chatbot
if __name__ == "__main__":
    while True:
        user_q = input("\nAsk your question (or type 'exit'): ")
        if user_q.lower() == 'exit':
            break
        result = get_answer(user_q)
        print(f"\nMatched Question: {result['Matched Question']}")
        print(f"Answer: {result['Answer']}")
        print(f"Confidence Score: {result['Confidence Score']:.2f}")
