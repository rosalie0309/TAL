import pickle
import pandas as pd
from nltk.tokenize import word_tokenize

# Charger le modèle BM25 et les documents
with open("bm25_model.pkl", "rb") as f:
    bm25 = pickle.load(f)

documents_df = pd.read_csv("documents_cleaned.csv")

synonym_dict = {
    "covid": ["covid-19", "coronavirus", "sars-cov-2", "sars cov 2", "ncov", "2019-ncov"],
    "origin": ["source", "emergence", "evolution", "beginning", "birth"],
    "pandemic": ["outbreak", "epidemic", "global health crisis"],
    "vaccine": ["vaccination", "immunization", "injection"],
    "transmission": ["spread", "contagion", "infection path", "propagation"],
    "face mask": ["mask", "surgical mask", "protective mask", "n95"],
    "lockdown": ["shutdown", "quarantine", "confinement", "isolation"],
    "hospital": ["clinic", "medical center", "health facility"],
    "test": ["diagnostic", "screening", "pcr", "swab"],
    "treatment": ["therapy", "medication", "drug", "remedy", "intervention"],
    "symptom": ["sign", "clinical sign", "manifestation"],
    "respiratory": ["breathing", "lung", "pulmonary", "airway"],
    "healthcare": ["medical care", "health system", "hospital system"],
    "death": ["mortality", "fatality", "loss of life"],
    "prevention": ["protection", "avoidance", "barrier", "control measures"],
    "mutation": ["variant", "strain", "genetic change", "genomic shift"],
    "children": ["kids", "infants", "minors", "young people"],
    "elderly": ["older adults", "seniors", "aged population", "geriatrics"],
    "vulnerable": ["at-risk", "immunocompromised", "fragile", "susceptible"],
    "public health": ["population health", "community health", "health policy"]
}



## Fonction pour l'utiliser 
def enrich_with_synonyms(text, synonym_dict):
    text = text.lower()
    for key, synonyms in synonym_dict.items():
        for synonym in synonyms:
            if synonym in text:
                text = text.replace(synonym, key)
    return text


def search_bm25(query_text, top_k=10):
    
    # On applique le même prétraitement qu'on a appliqué sur nos données lors de l'entrainement du model 
    query_text = enrich_with_synonyms(query_text, synonym_dict)
    tokenized_query = word_tokenize(query_text.lower())

    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    top_docs = documents_df.iloc[top_indices].copy()
    top_docs["score"] = [scores[i] for i in top_indices]
    top_docs["title"] = top_docs["text"].apply(lambda t: t.split(". ")[0] if ". " in t else t[:100])
    top_docs["abstract"] = top_docs["text"].apply(lambda t: t if ". " not in t else ". ".join(t.split(". ")[1:]))

    return top_docs[["doc_id", "title", "abstract", "score"]].to_dict(orient="records")


from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# Charger SBERT une seule fois
model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

def search_bm25_sbert_rerank(query_text, top_k=10, bm25_k=20):
    # Étape 1 - BM25
    bm25_results = search_bm25(query_text, top_k=bm25_k)
    if not bm25_results:
        return []

    # Étape 2 - SBERT embeddings
    query_embedding = model_sbert.encode(query_text, convert_to_tensor=True)
    doc_texts = [doc["title"] + ". " + doc["abstract"] for doc in bm25_results]
    doc_embeddings = model_sbert.encode(doc_texts, convert_to_tensor=True)

    # Étape 3 - Similarités cosinus
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings).flatten()

    # Étape 4 - Réordonnancement
    reranked_indices = np.argsort(cosine_scores.cpu().numpy())[::-1]

    # Étape 5 - Sélection finale
    reranked_results = []
    for i in reranked_indices[:top_k]:
        doc = bm25_results[i].copy()
        doc["semantic_score"] = float(cosine_scores[i])
        reranked_results.append(doc)

    return reranked_results

