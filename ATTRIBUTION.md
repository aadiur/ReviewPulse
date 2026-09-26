# Attribution & Origin Notice

This project, **ReviewPulse — AI-Powered Product Review Intelligence Platform**, is a substantially extended derivative work developed from an existing starter repository.

---

## 1. Original Repository Information

| Field | Detail |
|---|---|
| **Original Project Title** | NLP Review Analyser |
| **Original Repository / Source** | `NLP-Review-Analyser` |
| **Initial Implementation Scope** | Basic VADER sentiment scoring script, initial BERTopic invocation, and a minimal 39-line Streamlit display script. |
| **Original License** | **Unspecified / Not explicitly licensed.** The original repository contained no `LICENSE` file and no copyright or license declaration in its source files or README. |

> [!IMPORTANT]
> Because the original repository did not include an explicit open-source license, this repository does **not** claim an arbitrary license (such as MIT or Apache-2.0) on behalf of the original author. Users wishing to redistribute or use this software in proprietary settings should be aware of this origin.

---

## 2. Retained Concepts & Patterns

The following conceptual foundations from the original starter repository were preserved and built upon:

- Utilization of **VADER** lexicon-based scoring as an interpretable baseline for customer review sentiment.
- The concept of unsupervised **topic discovery** over review text.
- The use of **Streamlit** and **Plotly** for interactive data exploration.
- The primary domain problem: transforming qualitative product reviews into structured metrics.

---

## 3. Substantial Modifications & New Architectural Components

The repository was comprehensively redesigned, refactored, and expanded. The following modules and subsystems are entirely new or rewritten from scratch:

| Component | Description | Status |
|---|---|---|
| **Multi-Tier Architecture** | Clean directory layout separating presentation (`app/`), core logic (`src/`), data (`data/`), tests (`tests/`), and configuration (`configs/`). | **New** |
| **Data Ingestion Engine** (`src/data/loader.py`) | Automatic multi-alias column detection (review text, rating, timestamp), duplicate detection, empty row pruning, robust error handling without hardcoded paths. | **New** |
| **Text Preprocessing** (`src/preprocessing/cleaner.py`) | Unicode normalization (NFKD), HTML tag stripping, URL removal, whitespace compaction, preserving raw text alongside cleaned variants. | **New** |
| **Dual Sentiment & Uncertainty** (`src/sentiment/analyzer.py`) | Dual-model architecture combining VADER with a Hugging Face transformer model, disagreement detection, and confidence-based uncertainty flagging. | **New** |
| **Aspect-Based Sentiment** (`src/aspects/extractor.py`) | Pattern-driven aspect extraction across 12 product dimensions with localized context-window sentiment classification and net aspect scoring. | **New** |
| **Emotion Classification** (`src/emotions/detector.py`) | 7-class emotion detection using a distilled transformer with a heuristic fallback. | **New** |
| **Topic Modelling** (`src/topics/modeler.py`) | Production-ready scikit-learn Latent Dirichlet Allocation (LDA) providing guaranteed execution without heavy binary dependencies, plus optional BERTopic integration. Generates human-readable topic labels from top keywords. | **Substantially Rewritten** |
| **Keyword Extraction** (`src/keywords/extractor.py`) | Corpus-level and per-review n-gram TF-IDF extraction with adaptive minimum document frequency, plus optional KeyBERT. | **New** |
| **Embeddings & Clustering** (`src/clustering/embedder.py`) | Document embedding generation via `sentence-transformers` (with SVD/TF-IDF fallback), KMeans clustering, and PCA 2D coordinates for visual cluster inspection. | **New** |
| **Semantic Search** (`src/search/semantic_search.py`) | Natural-language semantic search across reviews using cosine similarity over embedding vectors. | **New** |
| **Product Intelligence Engine** (`src/insights/intelligence.py`) | Deterministic aggregation of sentiment distribution, most praised vs. criticized aspects, emotion distribution, and temporal trends. Never fabricates statistics. | **New** |
| **7-Page Streamlit App** (`app/`) | Complete multi-page interactive web application: Overview, Sentiment Intelligence, Aspect Intelligence, Topic Explorer, Semantic Search, Review Explorer, and Product Insights. | **Substantially Rewritten** |
| **Test Suite** (`tests/`) | Comprehensive automated test suite covering all data loading, preprocessing, sentiment, aspect, keyword, and insight modules. | **New** |
| **CLI Pipeline Runner** (`run_pipeline.py`) | Standalone CLI entrypoint with configurable flags for batch execution. | **New** |

---

## 4. Hardcoded Personal Path Elimination

The original repository relied on hardcoded personal paths:
```python
DATA_PATH = "C:\\Users\\Dell\\Downloads\\Reviews.csv"
```
This dependency has been eliminated. ReviewPulse dynamically accepts uploaded CSV files via the Streamlit interface, command-line arguments (`--input`), or configurable paths defined in `configs/settings.py`.

---

## 5. Third-Party Open Source Software

ReviewPulse utilizes the following open-source packages under their respective licenses:

| Package | Purpose | Upstream License |
|---|---|---|
| **pandas** | Tabular data manipulation | BSD 3-Clause |
| **numpy** | Numerical operations & vector math | BSD 3-Clause |
| **scikit-learn** | Machine learning (LDA, KMeans, PCA, TF-IDF) | BSD 3-Clause |
| **nltk** | Natural language toolkit | Apache 2.0 |
| **vaderSentiment** | Rule-based sentiment analysis | MIT |
| **torch** | Deep learning framework runtime | Modified BSD |
| **transformers** | Pretrained neural language models | Apache 2.0 |
| **sentence-transformers** | Dense sentence embedding generation | Apache 2.0 |
| **keybert** | Keyword extraction via embeddings | MIT |
| **streamlit** | Web dashboard framework | Apache 2.0 |
| **plotly** | Interactive data visualization charts | MIT |
| **pytest** | Automated test runner | MIT |

---

*Last revised: 2026*
