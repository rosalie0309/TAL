## COVID-19 Search Engine – Projet TAL

Ce projet a été réalisé dans le cadre du cours de Traitement Automatique des Langues (TAL), 
et avait pour objectif de concevoir un moteur de recherche documentaire autour des articles liés à la pandémie de COVID‑19.

### 1. Objectif du projet 
L’objectif était de comparer plusieurs modèles classiques et avancés de recherche d'information afin d’identifier celui qui offrait les meilleurs résultats en termes de pertinence, tout en tenant compte des contraintes de ressources (notamment l’absence de GPU)

### 2. Méthodes explorées 
Nous avons progressivement mis en œuvre plusieurs approches de recherche d’information :

- Recherche booléenne (avec opérateurs AND/OR)

- Recherche binaire (présence ou absence de mots dans le document)

- Modèle sac de mots (Bag-of-Words)

- TF (Term Frequency) et TF-IDF (Term Frequency-Inverse Document Frequency)

- Recherche vectorielle avec cosine similarity appliquée sur les vecteurs TF ou TF-IDF

- BM25 (Best Matching 25), un modèle probabiliste plus robuste

- SBERT (Sentence-BERT) pour un réordonnancement sémantique des résultats (semantic reranking): méthode non achevevée car très couteuse en temps de traitement 

### 3. Fonctionnement du moteur final 
Après évaluation, c’est le modèle BM25 qui a fourni les meilleurs résultats globaux, alliant pertinence et efficacité.
Nous avons enrichi la requête utilisateur avec un dictionnaire de synonymes pour améliorer la couverture sémantique.

### 4. Donées utilisées 
Le corpus utilisé provient de CORD-19 (COVID-19 Open Research Dataset), une collection d’articles scientifiques liés à la pandémie. Plus précisément, nous avons exploité le round 1 du jeu TREC-COVID via la bibliothèque ir_datasets, qui fournit une interface normalisée pour accéder aux documents, aux requêtes et aux jugements de pertinence.

```
import ir_datasets
# Chargement du jeu CORD-19 / TREC-COVID Round 1
dataset = ir_datasets.load("cord19/trec-covid/round1")

```

Chaque document contient :

- doc_id : identifiant unique

- title : titre de l’article


- abstract : résumé de l’article

