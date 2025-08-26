# CODECRAFT_DS_03 — Decision Tree Classifier (Bank Marketing)

**Goal:** Build a decision tree classifier to predict whether a customer subscribes (y) using the UCI Bank Marketing dataset.  
**Sample dataset:** https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

## Files
- `task03_bank_marketing_dt.py` — trains and evaluates a DecisionTreeClassifier.
- `requirements.txt`

## How to Use
1. Download the dataset (e.g., `bank-full.csv` or `bank.csv`) and place it here as `bank.csv`.
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python task03_bank_marketing_dt.py
   ```
4. Model metrics and a simple feature-importance chart will be saved in `./outputs/`.

## Note
- Ignore the unrelated "dogs vs cats" link; use only the UCI Bank Marketing dataset for this task.
