# SMS Spam Detection System

A complete SMS spam detection pipeline with three modules: classification, probability scoring, and spam clustering. The system classifies an input message as spam or ham, reports spam and ham probabilities, and, if spam, assigns a category using K‑Means clustering. The final user entry point is `cluster_model.py`.

---

## Overview

The project is designed to be simple to run and easy to mark. Training happens in a clear order. Saved models are reused on later runs.

**Execution and Training Order**

```
1) SMS_Classifier.py   → creates spam_model.joblib
2) SMS_Prob.py         → optional probability model (can be skipped)
3) cluster_model.py    → final system, creates spam_kmeans_bundle.joblib
(merge.py is used to prepare spam-only datasets for clustering)
```

Important notes:
- `cluster_model.py` does not create `spam_model.joblib`. You must run `SMS_Classifier.py` first to train and save the classifier.
- Running `SMS_Prob.py` is optional. If run, it trains the probability model used to report a continuous spam likelihood.

---

## 1. Environment Setup

### Requirements

- Python 3.11
- Libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`

### Create and activate a clean environment

```bash
conda create -n sms-spam python=3.11 -y
conda activate sms-spam

pip install --upgrade pip
pip install scikit-learn pandas numpy joblib
```

Alternatively, with `requirements.txt`:

```
scikit-learn
pandas
numpy
joblib
```

```bash
pip install -r requirements.txt
```

---

## 2. Project Structure

```
.
├─ Datasets/
│  ├─ spam.csv                    # UCI SMS Spam Collection
│  ├─ spam_d.csv                  # Main training data for SMS classifier
│  ├─ spamassassin_spam_clean.csv # SpamAssassin dataset (spam only)
│  ├─ Spam_cleaned.csv            # Output from merge.py (UCI spam only)
│  └─ spam_only_merged.csv        # Output from merge.py (merged spam datasets)
│
├─ SMS_Classifier.py              # Trains classifier and saves spam_model.joblib
├─ SMS_Prob.py                    # Optional probability model
├─ cluster_model.py               # Final interactive system + K‑Means clustering
├─ merge.py                       # Builds spam-only datasets for clustering
│
├─ spam_model.joblib              # Saved classifier (created by SMS_Classifier.py)
└─ spam_kmeans_bundle.joblib      # Saved clustering bundle (created by cluster_model.py)
```

Place all CSV files inside `Datasets/` exactly as shown.

---

## 3. Dataset Preparation

`merge.py` creates the spam-only corpus used for clustering by combining two sources.

- `Datasets/spam.csv` (UCI)
- `Datasets/spamassassin_spam_clean.csv` (SpamAssassin spam only)

Run:

```bash
python merge.py
```

Outputs in `Datasets/`:
- `Spam_cleaned.csv`  (UCI spam only)
- `spam_only_merged.csv`  (merged spam for clustering)

---

## 4. Training

You must train the classifier first. The probability model is optional. The clustering bundle is created by the final system.

### 4.1 Train the SMS classifier (required)

```bash
python SMS_Classifier.py
```
This script:
- reads `Datasets/spam_d.csv`
- splits into train and test with a fixed random state
- vectorizes with TF‑IDF
- trains a Logistic Regression classifier
- prints evaluation metrics
- saves `spam_model.joblib`

### 4.2 Train the probability model (optional)

```bash
python SMS_Prob.py
```
This script:
- loads the trained classifier and vectorizer
- fits a Ridge regression to produce a continuous spam likelihood between 0 and 1
- exposes helper functions for later use

You can skip this step and proceed directly to the final system if probability scoring is not required for marking.

### 4.3 Create the clustering bundle and run the final system

```bash
python cluster_model.py
```
This script:
- reads `Datasets/spam_only_merged.csv`
- vectorizes spam text and trains K‑Means on first run
- saves `spam_kmeans_bundle.joblib`
- provides an interactive console to classify an input message and, if spam, assign a cluster category

`cluster_model.py` requires that `spam_model.joblib` already exists from step 4.1.

---

## 5. Using the Final System

Run:

```bash
python cluster_model.py
```

Example session:

```
Enter message: Congratulations! You have won a brand new iPhone. Click here to claim your prize.

Result: Spam
Spam probability: 0.94
Ham probability: 0.06
Category: Promotion / Giveaway
```

The exact category label depends on the learned clusters and the category map in `cluster_model.py`.

---

## 6. Evaluation

During training, the classifier script prints standard metrics.

Example:

```python
from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
```

For clustering quality, you may report the silhouette score printed in the console during the first run.

---

## 7. Saved Models and Version Control

Saved artifacts:
- `spam_model.joblib`  (classifier)
- `spam_kmeans_bundle.joblib`  (clustering)

Add the following to `.gitignore` to avoid large files in commits:

```
*.joblib
```

If large model files must be tracked, use Git LFS.

---

## 8. Troubleshooting

| Issue | Likely cause | Fix |
|------|--------------|-----|
| `FileNotFoundError` on CSV | Wrong path or missing files | Place all CSV files under `Datasets/` |
| Encoding error when reading CSV | Non‑UTF8 characters | Load with `encoding="latin-1"` |
| `cluster_model.py` fails to load classifier | `spam_model.joblib` not created yet | Run `SMS_Classifier.py` first |
| Git push rejected for `.joblib` | File size over GitHub limit | Add to `.gitignore` or configure Git LFS |

---

## 9. Acknowledgements

- UCI SMS Spam Collection
- SpamAssassin Public Corpus
- scikit-learn, pandas, numpy, joblib

---

## 10. Notes for Assessment

- Steps are reproducible from a clean environment using the commands above.
- The training order is explicit. The final system requires a trained classifier.
- Outputs are easy to verify during marking: classification, probability, and category.
