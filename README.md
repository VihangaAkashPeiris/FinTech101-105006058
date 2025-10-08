# SMS Spam Detection System

A complete SMS spam detection pipeline with three parts: classification, probability scoring, and spam category clustering. The project provides a single entry point for end users while keeping each stage modular for training and evaluation.

## Highlights

- **Classifier** (`SMS_Classifier.py`) trains a TF‑IDF + Logistic Regression model and saves it to `spam_model.joblib` for reuse.
- **Probability scorer** (`SMS_Prob.py`) fits a Ridge regression on TF‑IDF features to produce a continuous spam likelihood in the range 0 to 1.
- **Clustering model** (`cluster_model.py`) groups spam into topic categories with K‑Means and exposes a friendly interactive console for end‑to‑end use: class label, probabilities for Ham and Spam, and a cluster category if the message is spam.
- **Data merge utility** (`merge.py`) builds the dedicated spam‑only dataset for clustering from the source CSV files.

> First run trains and saves the models. Later runs load the saved `.joblib` files and start immediately.

---

## 1. Project structure

```
.
├─ Datasets/
│  ├─ spam.csv                    # UCI SMS Spam Collection
│  ├─ spam_d.csv                  # Main training set for the classifier
│  ├─ spamassassin_spam_clean.csv # SpamAssassin (spam only)
│  ├─ Spam_cleaned.csv            # Output from merge.py (spam-only from UCI)
│  └─ spam_only_merged.csv        # Output from merge.py (SpamAssassin + UCI spam)
├─ SMS_Classifier.py              # Train + predict (Spam vs Ham)
├─ SMS_Prob.py                    # Probability scoring on the SMS pipeline
├─ cluster_model.py               # End‑to‑end interactive app + K‑Means clustering
├─ merge.py                       # Build spam-only dataset for clustering
├─ spam_model.joblib              # Saved classifier bundle (created on first run)
└─ spam_kmeans_bundle.joblib      # Saved clustering bundle (created on first run)
```

> Paths in the scripts assume the `Datasets/` folder exists in the project root.

---

## 2. Requirements

- Python 3.11
- Packages
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `joblib`

### Create the environment

```bash
# Create and activate a clean environment
conda create -n sms-spam python=3.11 -y
conda activate sms-spam

# Install dependencies
pip install --upgrade pip
pip install scikit-learn pandas numpy joblib
```

If you prefer `requirements.txt`, create one with these lines:

```
scikit-learn
pandas
numpy
joblib
```

Then install with:

```bash
pip install -r requirements.txt
```

---

## 3. Datasets

Place the CSV files under `Datasets/` exactly as shown in the structure above.

- `spam_d.csv` is the **main training file** for `SMS_Classifier.py`.
- `spam.csv` (UCI) and `spamassassin_spam_clean.csv` are used by `merge.py` to build the spam‑only corpus for clustering.
- `merge.py` outputs:
  - `Spam_cleaned.csv` (UCI spam only)
  - `spam_only_merged.csv` (SpamAssassin spam + UCI spam)

### Build the clustering dataset

```bash
python merge.py
```

This will create or refresh `Datasets/Spam_cleaned.csv` and `Datasets/spam_only_merged.csv`.

---

## 4. Training

You can train stage by stage, or simply run the final app which will train on the first run and load on later runs.

### Option A. Train step by step

1) **Train the classifier** (creates `spam_model.joblib`)

```bash
python SMS_Classifier.py
```
This script:
- reads `Datasets/spam_d.csv`
- splits into train and test using a fixed random state
- vectorizes text with TF‑IDF
- trains a Logistic Regression classifier
- prints evaluation metrics
- saves the trained objects to `spam_model.joblib`

2) **Train the probability scorer**

```bash
python SMS_Prob.py
```
This script imports the trained vectorizer and classifier, fits a Ridge regression for a continuous spam score, and exposes helper functions for later use.

3) **Train the clustering model** (creates `spam_kmeans_bundle.joblib`)

```bash
python cluster_model.py
```
On first run, this script vectorizes the spam‑only dataset and trains K‑Means. It stores a bundle with the vectorizer, K‑Means model, cluster keywords, and a category map for human‑readable labels.

### Option B. One‑shot via the final app

```bash
python cluster_model.py
```
- If no `.joblib` files exist, the script trains and saves them.
- If they exist, the script loads them and starts the interactive console immediately.

---

## 5. Usage

### Interactive console (recommended)

Run the final app:

```bash
python cluster_model.py
```

Sample session:

```
Enter message: Congratulations You have won a brand new iPhone Click the link to claim now
Result: Spam
Spam probability: 0.92
Ham probability: 0.08
Category: Prize promo / giveaway
Top cluster keywords: win, claim, click, prize, iphone
```

Your exact labels and keywords may differ depending on the data and the category map defined inside `cluster_model.py`.

### Programmatic use

```python
from SMS_Classifier import train_sms_model, predict_sms
from SMS_Prob import predict_message_reg

# Train or load the classifier
vectorizer, clf, *_ = train_sms_model()

# Predict class
msg = "Your package is on hold. Pay customs fee now to release it."
message, label = predict_sms(vectorizer, clf, msg)

# Probability score
labels, score = predict_message_reg(msg)   # score is between 0 and 1
```

---

## 6. Reproducibility and configuration

- Train‑test split uses a fixed random state.
- TF‑IDF options and model hyperparameters are defined in the scripts. You can adjust them at the top of each file.
- The clustering `category_map` can be edited in `cluster_model.py` to rename clusters once you review the top keywords.

---

## 7. Evaluation

The training scripts print standard metrics to the console. Typical examples include:

```python
from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
```

Add these lines during training if you want labeled output. The classifier script already includes similar prints.

---

## 8. Model files and version control

- `spam_model.joblib`
- `spam_kmeans_bundle.joblib`

These files can grow large. Do not commit them to Git by default. Add a `.gitignore` entry like:

```
*.joblib
```

If you must version large binaries, use Git LFS.

---

## 9. Troubleshooting

- **File not found**: Ensure the `Datasets/` folder exists and CSV file names match exactly.
- **Unicode or parsing errors** on CSV load: use `encoding="latin-1"` as shown in the scripts.
- **Package mismatch**: reinstall dependencies inside the clean environment shown above.
- **Large files block a Git push**: either ignore `.joblib` files or configure Git LFS before pushing.

---

## 10. Acknowledgements

- UCI SMS Spam Collection dataset
- SpamAssassin public corpora

Use these datasets in line with their respective licenses.

---

## 11. License

Include your license of choice here. For coursework, clarify the terms of use if this repository is public.
