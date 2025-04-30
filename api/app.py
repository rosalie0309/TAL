from flask import Flask, request, jsonify, render_template
from bm25_model import search_bm25_sbert_rerank

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Utilisation de SBERT pour reranker les résultats BM25
    results = search_bm25_sbert_rerank(query_text=query)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
