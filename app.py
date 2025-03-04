from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)  # Allow requests from Blogspot

# Load Grammar Correction Model
grammar_corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")

@app.route("/check_grammar", methods=["POST"])
def check_grammar():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        corrected_text = grammar_corrector(text, max_length=256)[0]['generated_text']
        return jsonify({"corrected_text": corrected_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
