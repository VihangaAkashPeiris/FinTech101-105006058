import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

#  Load engineered dataset 
#    Must contain: date, target, and the feature columns you created earlier
PATH = "Taskc7/DF_with_features_and_target.csv"
DF = pd.read_csv(PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

#  Define feature sets
#    Price block: simple returns, moving averages, momentum, short-window volatility, and RSI
price_features = ["ret_1d", "ma_5", "ma_10", "mom_5", "vol_5", "RSI_14"]
#    Sentiment blocks (shifted so they refer to yesterday’s info)
vader_feats   = ["vader_mean_shifted", "vader_change_shifted"]
finbert_feats = ["finbert_mean_shifted", "finbert_change_shifted"]

#    Convenient presetts so We can compare configurations one at a time
feat_sets = {
    "PriceOnly": price_features,
    "Price+VADER": price_features + vader_feats,
    "Price+FinBERT": price_features + finbert_feats,
    "Price+VADER+FinBERT": price_features + vader_feats + finbert_feats,
}

#  Choose ONE set to run now:
#     Start simple, then add sentiment once the baseline is stable
SET_NAME =  "Price+VADER"# Could be changed according to feat_sets
FEATURES = feat_sets[SET_NAME]

#  Build X, y
#    y is the binary label: 1 if next day is up, else 0
X = DF[FEATURES].copy()
y = DF["target"].astype(int).copy()

#  Chronological split (80/20)
#    No shuffling here. Train on the past, test on the future.
def time_split(X, y, split_ratio=0.8):
    n = len(X); cut = int(n * split_ratio)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

X_train, X_test, y_train, y_test = time_split(X, y, split_ratio=0.8)

#  Scale features (fit on TRAIN only)
#   fit on train, apply the same transform to test
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

#  Train Logistic Regression
lr = LogisticRegression(max_iter=1000, solver="lbfgs")  
lr.fit(X_train_s, y_train)

#  Evaluate
#    y_pred are class labels, y_prob are probabilities for the positive class (up)
y_pred = lr.predict(X_test_s)
y_prob = lr.predict_proba(X_test_s)[:, 1]  # probability of class 1 (UP)

#    Basic metrics for classification
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred)

print("\n============================================== Logistic Regression ================================================")
print(f"Feature set: {SET_NAME}")
print(f"Features   : {FEATURES}")
print(f"Accuracy   : {acc:.3f}")
print(f"Precision  : {prec:.3f}")
print(f"Recall     : {rec:.3f}")
print(f"F1         : {f1:.3f}")
print("\n=====================================================================================================================")
print("Confusion Matrix [TN FP; FN TP]:")
print(cm)
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0, digits=3))
print("\n=====================================================================================================================")

#  Save quick outputs 
#    One-row summary for easy aggregation across runs
summary = pd.DataFrame([{
    "FeatureSet": SET_NAME,
    "Model": "LogReg",
    "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
}])
summary_path = f"Taskc7_evidence/results_LogReg_{SET_NAME}.csv"
summary.to_csv(summary_path, index=False)

#    Coefficients tell direction and relative strength in log-odds space
coef_path = f"Taskc7_evidence/logreg_coefs_{SET_NAME}.csv"
coefs = pd.Series(lr.coef_[0], index=FEATURES).sort_values(ascending=False)
coefs.to_csv(coef_path, header=["coefficient"])

print(f"\nSaved metrics  -> {summary_path}")
print(f"Saved LR coefs -> {coef_path}")
print(coefs.head(10))
print("\n=====================================================================================================================")