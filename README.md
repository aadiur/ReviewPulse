# 🧠 ReviewPulse — AI-Powered Product Review Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io)
[![NLP](https://img.shields.io/badge/NLP-VADER%20%7C%20DistilBERT%20%7C%20LDA-blueviolet)]()

> From customer reviews to actionable product intelligence.

---

## 📌 Problem Statement

Customer reviews contain critical qualitative signals, but manually reading through thousands of entries is impractical. Traditional sentiment tools provide only a single coarse classification (Positive/Negative/Neutral). **ReviewPulse** analyzes reviews at multiple levels of granularity to explain *why* customers feel the way they do, which specific product features are praised or criticized, and what underlying themes emerge over time.

---

## ✨ Implemented Core Features

| Feature | Implementation | Description |
|---|---|---|
| **Dual Sentiment Analysis** | VADER + DistilBERT | Lexicon baseline combined with transformer classification for transparent, explainable scoring. |
| **Model Agreement & Uncertainty** | Disagreement Engine | Automatically flags reviews where VADER and the transformer diverge or when confidence is below threshold. |
| **Aspect-Based Sentiment (ABSA)** | Context Window Scoring | Extracts mentions across 12 product dimensions and isolates the sentiment of the specific aspect. |
| **Emotion Classification** | DistilRoBERTa & Fallback | Identifies emotional tones (joy, sadness, anger, fear, surprise, neutral, disgust). |
| **Topic Modelling** | Scikit-Learn LDA (Built-in) | Discovers latent topics and synthesizes human-readable topic labels from constituent keywords. Optional BERTopic integration. |
| **Keyword & Keyphrase Extraction** | TF-IDF & KeyBERT | Extracts corpus-level and review-level keywords with adaptive document frequency. |
| **Semantic Search** | Sentence Embeddings | Natural-language query search powered by dense vector representations and cosine similarity. |
| **Review Clustering** | KMeans + PCA 2D | Groups semantically similar feedback and maps clusters into 2D space for interactive visualization. |
| **Product Intelligence Engine** | Automated Synthesis | Calculates aggregated sentiment distributions, most praised/criticized aspects, and trends without fabricating metrics. |
| **Interactive Dashboard** | Streamlit (7 Pages) | Comprehensive web interface with Plotly data visualizations. |
| **Robust Data Ingestion** | Dynamic Auto-Detection | Detects review, rating, and date columns across multiple naming conventions; gracefully handles missing values and duplicates. |
| **Supervised Model Evaluation** | Scikit-Learn Metrics | Computes accuracy, precision, recall, F1, and confusion matrices whenever ground-truth ratings are available. |

---

## 🏗️ Architecture

```
reviewpulse/
│
├── app/
│   ├── streamlit_app.py          # Streamlit entry point & state management
│   └── pages/
│       ├── overview.py           # Executive KPIs, sentiment donut, top aspects & topics
│       ├── sentiment.py          # VADER vs. Transformer comparison & agreement analysis
│       ├── aspects.py            # Aspect frequency, praised vs. criticized breakdown
│       ├── topics.py             # Topic explorer cards & 2D PCA cluster scatter plot
│       ├── semantic_search.py    # Natural-language query interface with similarity ranks
│       ├── reviews.py            # Multi-dimensional filtering and CSV export
│       └── insights.py           # Automated product intelligence summary
│
├── src/
│   ├── data/loader.py            # CSV loading, alias resolution & validation
│   ├── preprocessing/cleaner.py  # Text cleaning, Unicode & HTML normalization
│   ├── sentiment/analyzer.py     # VADER + DistilBERT pipeline & uncertainty logic
│   ├── aspects/extractor.py      # Aspect extraction & context sentiment scoring
│   ├── emotions/detector.py      # Emotion classification with heuristic fallback
│   ├── topics/modeler.py         # Built-in LDA topic modeling & optional BERTopic
│   ├── keywords/extractor.py     # Adaptive TF-IDF & KeyBERT keyphrase extractors
│   ├── clustering/embedder.py    # SentenceTransformer dense embeddings & KMeans
│   ├── search/semantic_search.py # Vector similarity search engine
│   └── insights/intelligence.py  # Data-grounded executive insight generator
│
├── configs/settings.py           # Central configuration: model names, thresholds, aliases
├── tests/test_pipeline.py        # Automated test suite (37 unit & integration tests)
├── data/
│   ├── raw/                      # Input datasets (includes sample_reviews.csv)
│   └── processed/                # Pipeline CSV outputs (.gitkeep)
├── notebooks/
│   └── reviewpulse_demo.ipynb    # Interactive Jupyter exploration notebook
├── requirements.txt              # Production Python dependencies
├── run_pipeline.py               # Standalone command-line pipeline runner
├── streamlit_app.py              # Root launcher forwarding to app/
├── ATTRIBUTION.md                # Project origin and third-party license notice
└── README.md
```

---

## 🔄 End-to-End NLP Pipeline

```
Raw Customer Reviews
    │
    ▼ [1. Data Validation & Loader]
    Column Alias Detection → Deduplication → Null Pruning → ID Assignment
    │
    ▼ [2. Text Preprocessing]
    Unicode Normalization (NFKD) → HTML Stripping → URL Removal → Whitespace Trimming
    (Preserves raw review, cleaned text, and model-input representation)
    │
    ▼ [3. Dual-Layer Sentiment & Uncertainty]
    VADER Lexicon Scoring ────┐
                              ├─► Disagreement & Low-Confidence Uncertainty Flagging
    DistilBERT Transformer ───┘
    │
    ▼ [4. Aspect-Based Sentiment Analysis]
    Product Dimension Pattern Matching → Local Context Isolation → Aspect Polarity Assignment
    │
    ▼ [5. Emotion Detection]
    DistilRoBERTa Emotion Classifier (Joy, Anger, Sadness, Fear, Surprise, Neutral, Disgust)
    │
    ▼ [6. Semantic Topic Discovery & Keywords]
    Built-in Scikit-Learn LDA (Human-readable labels from top terms) + TF-IDF n-grams
    │
    ▼ [7. Embeddings, Clustering & Semantic Index]
    Dense Sentence Embeddings (all-MiniLM-L6-v2) → KMeans Clustering → PCA 2D Map → Cosine Similarity Index
    │
    ▼ [8. Product Intelligence & Dashboard]
    Deterministic Synthesis → Interactive Streamlit Interface
```

---

## 🛠️ Technology Stack

| Domain | Libraries & Tools |
|---|---|
| **Data Processing** | `pandas`, `numpy` |
| **Classical NLP** | `nltk`, `vaderSentiment` |
| **Machine Learning** | `scikit-learn` (LDA, KMeans, PCA, TF-IDF, Metrics) |
| **Neural NLP** | `transformers` (DistilBERT, DistilRoBERTa), `sentence-transformers` (`all-MiniLM-L6-v2`), `torch` |
| **Keyword Discovery** | `keybert`, `scikit-learn` |
| **Interactive Dashboard** | `streamlit`, `plotly` |
| **Testing** | `pytest` |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ReviewPulse.git
cd ReviewPulse
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Topic Modeling Notice:** ReviewPulse comes with a robust, lightweight **scikit-learn LDA** topic model enabled by default. To optionally use **BERTopic**, install it explicitly (`pip install bertopic>=0.16.0`). ReviewPulse automatically detects BERTopic if present and falls back gracefully to LDA if not.

---

## 🚀 Execution Guide

### Option 1: Interactive Streamlit Dashboard

Run from the repository root:
```bash
streamlit run streamlit_app.py
```
*(Or alternatively: `streamlit run app/streamlit_app.py`)*

Open your browser at `http://localhost:8501`. From the sidebar:
1. Upload your own CSV file, supply a local file path, or select the built-in sample dataset.
2. Toggle transformer sentiment or emotion detection as needed.
3. Click **▶ Run Analysis**.
4. Navigate through the 7 insight pages:
   - **Overview**: Summary metrics and high-level charts.
   - **Sentiment Intelligence**: Side-by-side VADER vs. Transformer analysis, agreement metrics, and confidence distributions.
   - **Aspect Intelligence**: Praised vs. criticized features, net aspect sentiment scores.
   - **Topic Explorer**: Discovered topics, representative customer excerpts, and 2D cluster scatter plot.
   - **Semantic Search**: Real-time query search across reviews using semantic embedding similarity.
   - **Review Explorer**: Multi-filter review table (filter by sentiment, emotion, rating, aspect, topic, or uncertainty) with CSV export.
   - **Product Intelligence**: Actionable business summary and automated observations.

### Option 2: Command-Line Pipeline Runner

To analyze reviews from the command line and save enriched CSVs:

```bash
# Standard run using sample dataset
python run_pipeline.py --input data/raw/sample_reviews.csv

# Fast mode (VADER baseline only, skipping heavy models)
python run_pipeline.py --input data/raw/sample_reviews.csv --no-transformer --no-emotions

# Sample a subset of reviews for quick testing
python run_pipeline.py --input data/raw/sample_reviews.csv --sample 100
```

Outputs are written to `data/processed/`:
- `reviews_processed.csv`: Enriched reviews with sentiment, aspect tags, topic IDs, and cluster labels.
- `aspects.csv`: Long-format table of all extracted aspect mentions and local sentiments.
- `topics.csv`: Discovered topics, keyword representations, and frequencies.
- `keywords.csv`: Top corpus keyphrases.

---

## 📁 Supported Data Formats

ReviewPulse does not require rigid column names. It automatically recognizes common column aliases:

| Field | Recognized Column Headers (case-insensitive) |
|---|---|
| **Review Content** | `review`, `text`, `review_text`, `content`, `body`, `comment` |
| **Rating (Optional)** | `rating`, `score`, `stars`, `overall` |
| **Timestamp (Optional)**| `date`, `time`, `timestamp`, `created_at`, `review_date` |

*Note: Ratings and timestamps are optional. If absent, ReviewPulse computes all sentiment, aspect, topic, and semantic features without error.*

---

## 🤖 Model Specifications & Hardware Compatibility

All models selected for ReviewPulse are compact, distilled architectures designed for **CPU-compatible execution** without requiring a dedicated GPU:

| Component | Model Name | Download Size | Execution |
|---|---|---|---|
| **Sentiment** | `distilbert-base-uncased-finetuned-sst-2-english` | ~67 MB | CPU |
| **Emotion** | `j-hartmann/emotion-english-distilroberta-base` | ~82 MB | CPU |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB | CPU |
| **Topic Modeling**| Scikit-Learn LDA (Built-in) / BERTopic (Optional) | Local | CPU |

Models are automatically retrieved from Hugging Face on first execution and cached locally. Model identifiers can be customized via environment variables:
```bash
set REVIEWPULSE_SENTIMENT_MODEL=distilbert-base-uncased-finetuned-sst-2-english
set REVIEWPULSE_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 🧪 Testing & Verification

ReviewPulse includes an automated test suite verifying data ingestion, edge cases, preprocessing rules, sentiment scoring, aspect extraction, keywords, and pipeline integration.

Run the test suite:
```bash
pytest tests/ -v
```

All 37 test cases validate:
- Missing column and malformed CSV error handling
- Deduplication and empty review filtering
- Unicode normalization and HTML/URL stripping
- VADER positive, negative, and neutral boundary scoring
- Seed aspect extraction and local context isolation
- Deterministic metric computation (no fabricated statistics)
- End-to-end pipeline execution smoke test

---

## ⚠️ Known Technical Limitations

1. **Aspect Matching**: Aspect extraction is based on targeted category lexicon matching with contextual window sentiment scoring. It does not perform full constituency dependency parsing on rare or domain-specific slang.
2. **Review Length**: Transformer sentiment analysis truncates reviews exceeding 512 tokens.
3. **Temporal Insights**: Time trends require valid timestamp data in the input CSV; when absent, temporal visualizations are omitted.
4. **Topic Modeling Scale**: BERTopic requires substantial dataset size (≥ 50 reviews) and external dependencies. The built-in LDA model is used as the reliable default.

---

## 🔮 Future Roadmap (Optional Extensions)

The following architectural extensions are planned for future versions:
- [ ] **SHAP/LIME Explainability**: Visual token-level attribution for transformer sentiment classifications.
- [ ] **FastAPI Service**: A REST API endpoint (`POST /analyze`) for real-time external integration.
- [ ] **SQLite Persistence**: Local database support for tracking review datasets across sessions.
- [ ] **LLM Grounded Synthesis**: Optional LLM prompt adapter for natural language report generation strictly constrained to computed metrics.

---

## 📜 Attribution & License Notice

This project was developed as a substantial derivative expansion of the open-source starter project `NLP-Review-Analyser`. 

The original repository did not include an explicit open-source license. Please consult [ATTRIBUTION.md](ATTRIBUTION.md) for full attribution, list of retained concepts, newly developed components, and third-party dependency licenses.
