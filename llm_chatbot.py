from flask import Flask, request, jsonify
import pandas as pd
import google.generativeai as genai
from typing import List, Tuple

# 1. Load and Preprocess Data
def load_qa_data(filename="dataset.xlsx") -> List[Tuple[str, str]]:
    """
    Loads question-answer data from an Excel file and preprocesses it.

    Args:
        filename (str): The name of the Excel file.  Defaults to "dataset.xlsx".

    Returns:
        List[Tuple[str, str]]: A list of (question, answer) tuples.  Returns an empty list on error.
    """
    try:
        df = pd.read_excel(filename).dropna()  # Drop rows with any missing values
        #  Handle potential column name variations (case-insensitive and with/without spaces)
        question_cols = ['Question', 'question']
        answer_cols = ['Answer', 'answer']

        question_col = None
        for col in question_cols:
            if col in df.columns:
                question_col = col;
                break
        if not question_col:
            raise ValueError(f"No question column found. Expected one of {question_cols}")

        answer_col = None
        for col in answer_cols:
            if col in df.columns:
                answer_col = col
                break
        if not answer_col:
            raise ValueError(f"No answer column found. Expected one of {answer_cols}")
        
        qa_pairs = list(zip(df[question_col], df[answer_col]))
        return qa_pairs
    except FileNotFoundError:
        print(f"Error: File not found at {filename}.  Please ensure the file exists.")
        return []
    except ValueError as e:
        print(f"Error: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def format_qa_context(qa_pairs: List[Tuple[str, str]]) -> str:
    """
    Formats the question-answer pairs into a context string for the Gemini prompt.

    Args:
        qa_pairs (List[Tuple[str, str]]): A list of (question, answer) tuples.

    Returns:
        str: A formatted context string.
    """
    context = ""
    for q, a in qa_pairs:
        context += f"Q: {q}\nA: {a}\n\n"
    return context

# 2. Configure Gemini
def setup_gemini(api_key: str) -> genai.GenerativeModel:
    """
    Configures the Gemini model with the provided API key.

    Args:
        api_key (str): Your Gemini API key.

    Returns:
        genai.GenerativeModel: The configured Gemini model.  Returns None on error.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro")
        return model
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return None

# 3. Flask App
app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Handles user questions by querying the Gemini model with a crafted prompt.
    """
    data = request.get_json()
    user_question = data.get("question")

    if not user_question:
        return jsonify({"error": "Please provide a 'question' field in the request."}), 400

    # 4.  Contextual Prompt Engineering (CRITICAL)
    preamble = "You are a helpful assistant.  Answer the user's question concisely and accurately using only the information provided in the following knowledge base. Do not make up answers or provide information that is not in the knowledge base.  If the question cannot be answered from the knowledge base, respond with \"I cannot answer the question from the provided knowledge base.\""
    
    prompt = f"{preamble}\n\nKnowledge Base:\n{qa_context}\n\nUser Question: {user_question}\nAnswer:"

    try:
        response = model.generate_content(prompt)
        #  Improved response handling.  Check for None and handle it.
        if response and response.text:
            answer = response.text.strip()
        else:
            answer = "I cannot answer the question from the provided knowledge base."
        
        return jsonify({
            "question": user_question,
            "answer": answer
        })
    except Exception as e:
        error_message = f"Error generating response: {e}"
        print(error_message)  # Log the error for debugging
        return jsonify({"error": error_message}), 500

if __name__ == "__main__":
    # 5.  Initialization (Load data and model)
    qa_pairs = load_qa_data()
    if not qa_pairs:
        print("Failed to load Q&A data. Exiting.")
        exit(1)  #  Exit with an error code
    
    qa_context = format_qa_context(qa_pairs)
    
    #  Replace "YOUR_GEMINI_API_KEY" with your actual API key.  Better to get from environment variable.
    gemini_api_key = "YOUR_GEMINI_API_KEY" #  DO NOT HARDCODE IN REAL APP
    model = setup_gemini(gemini_api_key)
    if not model:
        print("Failed to configure Gemini. Exiting.")
        exit(1)  # Exit with an error code
    
    app.run(debug=True, port=5000)
