# Zepto Capstone Project

This repository contains all three modules of the Zepto Data & AI Platform capstone: a data-engineering pipeline, an analytics/modeling pipeline, and a GenAI support assistant. They are meant to read as one story — a data pipeline feeds clean structured data, an analytics pipeline shows how Zepto would profile and predict outcomes end to end, and a support assistant shows how Zepto would put a grounded GenAI service in front of its own policies.

Each module has its own `requirements.txt` (one per module, not a single consolidated file), since they depend on different, largely non-overlapping libraries.

## Repository layout

```
Zepto_Capstone/
├── data_pipeline/       # Module 1 — scrape, clean, store, query (25 marks)
├── analytics/           # Module 2 — Titanic EDA + modeling pipeline (50 marks)
├── support_assistant/   # Module 3 — RAG support assistant (25 marks)
└── README.md            # this file
```

---

## Module 1 — Data Pipeline (`/data_pipeline`)

**What it does:** scrapes book listings from books.toscrape.com (5 pages of the "All products" catalogue, 100 books), cleans and types the fields, converts price to INR using a fixed rate, loads everything into a normalized two-table SQLite database, and runs SQL queries against it alongside equivalent pandas operations.

**Setup and run:**
```bash
cd data_pipeline
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python scraper.py
```
Running `scraper.py` performs the whole pipeline end to end: scrape → clean → convert → build `books.db` → run and print the SQL queries → print the `pd.read_sql` / `pd.merge` comparison. No API key or account is needed.

**Design decisions:**
- Currency conversion uses the fixed project-defined rate **1 GBP = 105.50 INR** (not a live rate — no lookup or date reference needed).
- Missing numeric fields (price, rating) are filled with the column median; rows with missing text fields (title, availability, category) are dropped, since there's no sensible way to impute free text.
- Schema: `categories(category_id PK, category_name)` and `books(book_id PK, title, price_gbp, price_inr, rating, in_stock, category_id FK)` — a standard one-to-many relationship.
- Five SQL queries cover SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN, and a JOIN across both tables; the JOIN result is also reproduced with `pd.merge` on the equivalent in-memory DataFrames to confirm both approaches agree.

See `data_pipeline/README.md` for further detail.

---

## Module 2 — Analytics Pipeline (`/analytics`)

**What it does:** loads the Titanic dataset once, profiles and cleans it, tells a visual data story across several charts, then builds and evaluates a full classification pipeline (Logistic Regression, Decision Tree, Random Forest) plus a regression side-task predicting fare — all from the same cleaned data.

**Setup and run:**
```bash
cd analytics
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Open `01_eda.ipynb` in Jupyter/VS Code and run it top to bottom first — this is the **only** place the raw dataset is loaded (`sns.load_dataset('titanic')`, needs internet the first time only), and it saves `titanic.csv` as an offline fallback. Then run `02_modeling.ipynb`, which reads that same `titanic.csv` and needs no further internet access.

**Design decisions:**
- Missing values are handled per-column by a percentage threshold: under 5% missing → drop those rows (`embarked`); 5–30% → impute (`age`, median); over 30% → drop the column (`deck`, ~77% missing, too sparse to impute reliably).
- The correlation matrix is restricted to exactly the 6 numeric columns the task specifies (`survived`, `pclass`, `age`, `sibsp`, `parch`, `fare`), explicitly excluding the derived flags `adult_male` and `alone`.
- All preprocessing (imputing, encoding, scaling) is done inside a scikit-learn `Pipeline`/`ColumnTransformer`, fit only on the training split, to structurally prevent test-set leakage.
- Three classifiers are compared on accuracy/precision/recall/F1/AUC; class imbalance is handled three ways (baseline, `class_weight='balanced'`, SMOTE on the training fold only) and compared; the Random Forest is tuned with `GridSearchCV` and reports its OOB score.
- The final recommendation is the Random Forest (best accuracy and F1 on the test set); full reasoning and all written interpretations are in `analytics/README.md` and the notebooks' markdown cells.
- The complete fitted pipeline (preprocessing + model together) is saved with `joblib.dump`, and reloading it is verified against raw, unprocessed input.

See `analytics/README.md` for the full model comparison table and every written interpretation.

---

## Module 3 — Support Assistant (`/support_assistant`)

**What it does:** a small RAG service that answers questions about Zepto's own policies (delivery, returns, membership, etc.), grounded in 8 policy documents embedded locally and retrieved with ChromaDB, orchestrated through a LangGraph intent router, with a schema-validated JSON response and a FastAPI wrapper.

**Setup and run:**
```bash
cd support_assistant
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```
The first request triggers ingestion (downloads the `all-MiniLM-L6-v2` embedding model once, then embeds the 8 documents into a persistent ChromaDB collection). Test it at `http://localhost:7860/docs`, or with:
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d "{\"query\": \"How long does delivery take?\"}"
```
A `Dockerfile` is also included and has been tested with `docker build` + `docker run`.

**Design decisions:**
- Every LLM call is gated behind the `MOCK_LLM` environment variable. Left at its default (unset), the whole service is deterministic and fully offline — no signup, no API key, no network call to any LLM provider — which is the required graded baseline.
- `classify_intent` routes each query to `policy_question` or `general_question` using a keyword heuristic in mock mode (no LLM call either way).
- `retrieve_and_answer` always performs real retrieval (embed query → ChromaDB top-3 by cosine similarity) in both modes, since that needs no API key; only the final answer-generation step branches on `MOCK_LLM` — a canned template in mock mode, or a real LLM call (Groq free tier) using the structured prompt in `prompts.py` if `MOCK_LLM=0` is explicitly set.
- The JSON output (`answer`/`sources`/`confidence`) is enforced with a Pydantic model; the optional real-LLM path retries up to 2 times with a corrective instruction if the model's output fails validation.
- Full architecture walkthrough and real captured example-call transcripts (from an actual local run, `MOCK_LLM` at its default) are in `support_assistant/README.md`.

---

## Git workflow

This repository's history includes a feature branch (`feature/data-pipeline`) created off `main`, committed to more than once, and merged back into `main` — visible via `git log --graph --all`.
