# SMS & Email Spam Detection — README

This guide explains how to **configure your environment (conda)**, **prepare data**, **train the models**, and **run predictions** for the multi‑stage spam system:
**Classify → Score Probability → Cluster**.

> Works on macOS / Linux / Windows (PowerShell). Python 3.10+ recommended.

---

## 1) Environment Setup (conda)

### 1.1 Create and activate a conda env
```bash
# Create an environment named 'spam-ml' with Python 3.10 (or 3.11)
conda create -n spam-ml python=3.10 -y
conda activate spam-ml
```

### 1.2 Install dependencies
> We use scikit‑learn for ML, pandas for data, joblib for model persistence.
```bash
# Use pip inside the conda env
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn joblib
```

**Optional (useful for notebooks/plots):**
```bash
pip install matplotlib jupyter
```

### 1.3 (Optional) Save / restore the environment
```bash
# Export exact package versions
conda env export --name spam-ml --no-builds > environment.yml

# Recreate later
conda env create -f environment.yml
conda activate spam-ml
```

---

## 2) Project Structure (expected)

```
project/
├─ Datasets/
│  ├─ spam.csv                      # UCI
│  ├─ spamassassin_spam_clean.csv   # SpamAssassin (spam-only)
│  ├─ spam_d.csv                    # Large labelled SMS dataset
│  ├─ spam_only_merged.csv          # (generated) spam-only merge for clustering
├─ merge.py
├─ SMS_Classifier.py
├─ SMS_Prob.py
├─ cluster_model.py
└─ README.md
```

> If your dataset files are located elsewhere, update paths in the scripts (search for `Datasets/` and adjust).

---

## 3) Data Preparation

### 3.1 Build the spam-only corpus for clustering
`merge.py` reads **UCI** and **SpamAssassin**, normalises columns/encoding, and writes **spam_only_merged.csv**.

```bash
# From the project root
python merge.py
```

**What it does:**
- Reads `Datasets/spam.csv` (latin‑1), keeps `['v1','v2']`, renames `v2→text`, filters `v1=='spam'` → `Spam_cleaned.csv`
- Reads `Datasets/spamassassin_spam_clean.csv` (`text` column)
- Concatenates both spam sources → `Datasets/spam_only_merged.csv`

**Recommended enhancement (optional):**
Deduplicate and drop empty rows inside `merge.py` before saving:
```python
df = df.dropna(subset=['text']).drop_duplicates(subset=['text'])
```

---

## 4) Training

### 4.1 Train the SMS classifier (spam vs ham)
`SMS_Classifier.py` loads `Datasets/spam_d.csv`, performs TF‑IDF, then trains **Logistic Regression**.

```bash
python SMS_Classifier.py
```

**Outputs (typical):**
- Classification metrics (Accuracy/Precision/Recall/F1 + Confusion Matrix)
- Persisted artifacts via **joblib** (vectorizer + model)

### 4.2 Train the spam probability regressor
`SMS_Prob.py` uses the same TF‑IDF features and trains **Ridge** regression to predict a continuous **spam score**.

```bash
python SMS_Prob.py
```
**Outputs:** RMSE, MAE, R² and a persisted regression model bundle (e.g., `sms_ridge.joblib`).

### 4.3 Train the spam clustering model
`cluster_model.py` loads `Datasets/spam_only_merged.csv`, cleans text, vectorises, and fits **K‑Means**.
```bash
python cluster_model.py
```
**Outputs:**
- Silhouette score
- Saved bundle (e.g., `spam_kmeans_bundle.joblib`) containing vectorizer + kmeans
- Top terms per cluster to help name categories

---

## 5) Inference / Prediction

You can call the prediction helpers from Python **or** from the command line.

### 5.1 Python usage

```python
# Example: classify and score a single SMS, then (if spam) cluster it
from SMS_Classifier import predict_sms        # returns 'spam' or 'ham' (and/or prob depending on your function)
from SMS_Prob import predict_message_reg      # returns a float score 0..1
# If your cluster file exposes a prediction function:
# from cluster_model import predict_cluster

msg = "Congratulations! You won a $500 gift card. Click http://scam.link to claim now"
pred_label = predict_sms(msg)
score = predict_message_reg(msg)

print("Label:", pred_label)
print("Score:", round(score, 3))

if pred_label.lower() == "spam":
    # If you expose a 'predict_cluster' function in cluster_model.py:
    # cluster_id, cluster_name = predict_cluster(msg)
    # print("Cluster:", cluster_id, cluster_name)
    pass
```

### 5.2 One‑liners from the shell

> Replace the model and vectorizer names with the exact filenames produced by your scripts if needed.

**Classify:**
```bash
python - <<'PY'
from SMS_Classifier import predict_sms
print(predict_sms("Free entry in 2 a weekly competition to win cash! Txt WIN to 80086"))
PY
```

**Score probability:**
```bash
python - <<'PY'
from SMS_Prob import predict_message_reg
print(predict_message_reg("Low premium life insurance quotes available today"))
PY
```

**Cluster (if prediction helper is defined):**
```bash
python - <<'PY'
# from cluster_model import predict_cluster
# print(predict_cluster("Verify your bank account by following this secure link"))
print("Add predict_cluster(...) in cluster_model.py to enable CLI clustering.")
PY
```

---

## 6) Reproducibility & Tips

- **Random seeds:** For deterministic results, set `random_state` in `train_test_split`, `LogisticRegression`, and `KMeans`.
- **Vocabulary size:** `TfidfVectorizer(max_features=15000)` is a good balance for speed vs nuance.
- **Encodings:** Use `encoding='latin-1'` for `spam.csv` and `utf-8` for others.
- **Artifacts:** Keep `*.joblib` files under `models/` (recommended). Example:
```
models/
├─ vectorizer.joblib
├─ sms_logreg.joblib
├─ sms_ridge.joblib
└─ spam_kmeans_bundle.joblib
```
- **Thresholding:** Use the regression score or classifier `predict_proba` to set business‑specific thresholds for quarantine/allow.

---

## 7) Export README to PDF (optional)

If you have **pandoc** installed:
```bash
# macOS (brew) or Windows (choco) install pandoc if needed
# brew install pandoc
# choco install pandoc

pandoc README.md -o README.pdf
```

---

## 8) Troubleshooting

- **ConvergenceWarning (LogisticRegression):**
  - Increase `max_iter=1000` or standardise TF‑IDF options.
- **Memory issues (TF‑IDF):**
  - Reduce `max_features`, increase `min_df`, or sample data.
- **Weird characters / � :**
  - Recheck `encoding=` when reading CSVs.
- **Cluster instability:**
  - Set `n_init=50` or higher; review preprocessing regex.
- **ImportError for predict functions:**
  - Ensure `predict_sms(...)` and `predict_message_reg(...)` are defined and return a value in your scripts.
