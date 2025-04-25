from flask import Flask, request, jsonify
import pandas as pd
import google.generativeai as genai

app = Flask(__name__)

# 1. Load Q&A Data
def load_qa_data(filename="dataset.xlsx"):
    try:
        df = pd.read_excel(filename).dropna()
        Q_COL = 'Question' if 'Question' in df else 'question'
        A_COL = 'Answer' if 'Answer' in df else 'answer'
        return list(zip(df[Q_COL], df[A_COL]))
    except Exception as e:
        return []

# 2. Format Q&A Context
def format_qa_context(qa_pairs):
    return "\n\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)

# 3. Configure Gemini
def setup_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-pro")
    except Exception:
        return None

# 4. Ask Question
def ask_gemini(model, qa_context, user_question):
    prompt = f"""
    You are a helpful assistant. 
    Answer ONLY from the provided knowledge base. 
    Do not make up answers.
    If the answer isn't in the knowledge base, say: "I cannot answer this."

    Knowledge Base:
    {qa_context}

    User Question: {user_question}
    Answer:
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else "I cannot answer this."
    except Exception as e:
        return f"Error: {e}"

@app.route("/ask", methods=["POST"])
def ask():
    question = request.get_json().get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    answer = ask_gemini(model, qa_context, question)
    return jsonify({"question": question, "answer": answer})

if __name__ == "__main__":
    qa_pairs = load_qa_data()
    if not qa_pairs:
        exit("Failed to load data.")
    qa_context = format_qa_context(qa_pairs)

    gemini_key = "YOUR_GEMINI_API_KEY" # Replace this
    model = setup_gemini(gemini_key)
    if not model:
        exit("Failed to setup Gemini.")

    app.run(debug=True, port=5000)
