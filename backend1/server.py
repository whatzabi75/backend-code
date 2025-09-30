import os
from dotenv import load_dotenv  # must load before importing modules that read env
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
from emotion_detection import emotion_detector  # adjust import to your function
from stock_analyzer import stock_analyzer  # import the stock analyzer function

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
    
    if not symbol:
        return jsonify({"error": "No stock symbol provided"}), 400

    analysis = stock_analyzer(symbol)  # call the stock analyzer function
    return jsonify(analysis)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)