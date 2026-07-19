# Pokemon Primary Type Classifier

An end-to-end data science pipeline: pull Pokemon data from the free,
key-less [PokeAPI](https://pokeapi.co/docs/v2), clean it, train a
classifier to predict a Pokemon's primary type from its base stats,
and serve predictions through an interactive Streamlit dashboard.

## Live app
[Add your Streamlit Cloud URL here once deployed]

## Project structure
```
├── app.py                  # Streamlit dashboard (5 sections, see below)
├── fetch_data.py           # Task 1: pulls data from PokeAPI -> data/raw_data.csv
├── eda_and_clean.py        # Task 2: EDA + cleaning -> data/clean_data.csv, notebooks/*.png
├── train_model.py          # Task 3: trains + compares models -> model.pkl, scaler.pkl, label_encoder.pkl, model_metrics.json
├── requirements.txt
├── NOTES.md                # Task 2 cleaning decisions, with reasoning
├── MODEL_NOTES.md          # Task 3 model comparison and choice, with reasoning
├── data/
│   ├── raw_data.csv
│   └── clean_data.csv
├── notebooks/              # EDA plots (class balance, correlation heatmap, etc.)
├── model.pkl
├── scaler.pkl
├── label_encoder.pkl
└── model_metrics.json
```

## Data source
[PokeAPI](https://pokeapi.co/docs/v2) — free, no API key or registration
required. `fetch_data.py` paginates through the `/pokemon` list endpoint,
then calls each Pokemon's detail endpoint to retrieve stats and types.

## Target and features
- **Target**: `primary_type` (18 classes: Fire, Water, Grass, etc.)
- **Features**: `height`, `weight`, `base_experience`, `hp`, `attack`,
  `defense`, `special_attack`, `special_defense`, `speed`
  (stats-only, to avoid leaking type information into the model —
  see `MODEL_NOTES.md` for why)

## How to run this locally
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

# Optional: regenerate the data/model from scratch (already included in repo)
python fetch_data.py
python eda_and_clean.py
python train_model.py

# Run the dashboard
streamlit run app.py
```
Then open the local URL it prints (usually `http://localhost:8501`).

## Pipeline summary
1. **Data acquisition** (`fetch_data.py`): paginated PokeAPI pull, ~1,351 Pokemon including alternate forms.
2. **EDA & cleaning** (`eda_and_clean.py`, `NOTES.md`): removed alternate forms (mega/gmax/regional, id > 10000), median-imputed `base_experience`, labeled missing `secondary_type` as "none", one-hot encoded categoricals. Result: 1,025 rows.
3. **Modeling** (`train_model.py`, `MODEL_NOTES.md`): Logistic Regression vs. Random Forest, evaluated with macro-averaged precision/recall/F1 due to class imbalance. Random Forest selected (macro F1 = 0.138 vs. 0.135).
4. **Dashboard** (`app.py`): 5 sections — overview, data preview, interactive EDA, model performance, live prediction — with sidebar navigation and cached data/model loading.
5. **Deployment**: Streamlit Community Cloud.

## Known limitation
Model accuracy (~21.5%) is modest — expected, since Pokemon typing is a
design/lore decision that isn't fully determined by base stats alone
(e.g. Fire and Fighting types can have very similar stat spreads). The
project's goal was a complete, honestly-evaluated pipeline rather than
a high accuracy score.
