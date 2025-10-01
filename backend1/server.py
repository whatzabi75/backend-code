import os
from dotenv import load_dotenv  # must load before importing modules that read env
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from emotion_detection import emotion_detector  # adjust import to your function
from stock_analyzer import stock_analyzer  # import the stock analyzer function

from rag_upload import process_pdf, answer_question  # import the RAG deployment function

app = Flask(__name__)
CORS(app)

#------------------------------------
# Emotion Detection Route
#------------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    emotion = emotion_detector(text)  # your existing Python function
    return jsonify(emotion)

#------------------------------------
# Stock Analyzer Route
#------------------------------------
@app.route("/stock-analyzer", methods=["POST"])
def analyze_stock():
    data = request.get_json()
    symbol = data.get("symbol", "").upper()
    range_ = data.get("range", "6mo")
    
    if not symbol:
        return jsonify({"error": "No stock symbol provided"}), 400

    analysis = stock_analyzer(symbol, range_)  # call the stock analyzer function
    return jsonify(analysis)

#------------------------------------
# RAG upload Route
#------------------------------------
@app.route("/rag-upload", methods=["POST"])
def rag_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files["file"]
    print(f"[INFO] Upload received: {file.filename}")

    if not file.filename.endswith(".pdf"):
        print(f"[WARN] Rejected non-PDF file: {file.filename}")
        return jsonify({"error": "Only PDF files are allowed"}), 400
    
    # process_pdf should parse PDF, chunk, embed, and store in FAISS
    try:
       success = process_pdf(file)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    if success:
        return jsonify({"status": "ok", "message": "PDF processed and model trained"})
    else:
        return jsonify({"status": "failed", "message": "PDF could not be processed"}), 500

#------------------------------------
# RAG Chat with uploaded file Route
#------------------------------------
@app.route("/rag-chat", methods=["POST"])
def rag_chat():
    data = request.get_json()
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    answer = answer_question(question)  # retrieves from FAISS + calls LLM
    return jsonify({"status": "ok", "answer": answer})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)