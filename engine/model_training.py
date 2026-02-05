import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
df = pd.read_csv("data_log_final.csv")
cols = [ 
    "difficulty_before",
    "skill",
    "task_id",
    "answer_type",
    "time_taken",
    "is_correct",
    ]
df = df[cols].copy()
df["is_correct"] = df["is_correct"].astype(str).str.strip().str.lower()
df["is_correct"] = df["is_correct"].map({ "1": 1, "true": 1, "0": 0, "false": 0 })
df["difficulty_before"] = pd.to_numeric(df["difficulty_before"], errors="coerce")
df["time_taken"] = pd.to_numeric(df["time_taken"], errors="coerce")
df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")
df = df.dropna()
df = df[(df["time_taken"] > 0) & (df["time_taken"] < 120)]
df = df[(df["difficulty_before"] >= 0) & (df["difficulty_before"] <= 100)] 
df = df[df["is_correct"].isin([0, 1])]
df["task_subfamily"] = df["task_id"].str.split("-").str[1]
x = df[[ "difficulty_before", "skill", "task_subfamily", "answer_type" ]]
y_correct = df["is_correct"].astype(int)
y_time = np.log1p(df["time_taken"].astype(float))
x = pd.get_dummies(x, columns=["skill","answer_type","task_subfamily"], dtype=int)
for col in x.columns:
    if col.startswith("skill_"):
        x[f"{col}_difficulty"] = x[col] * x["difficulty_before"]
feature_cols = x.columns.tolist()
print("Num features:", len(feature_cols))
x_train, x_test, y_train, y_test = train_test_split(x, y_correct, random_state=42, test_size=0.2, stratify=y_correct)
y_train_t = y_time.loc[x_train.index]
y_test_t = y_time.loc[x_test.index]
model = LogisticRegression(max_iter=5000, class_weight = "balanced")
model.fit(x_train, y_train)
pred = model.predict(x_test)
probs = model.predict_proba(x_test)[:, 1]
time_model = Ridge(alpha = 1.0)
time_model.fit(x_train, y_train_t)
pred_time = time_model.predict(x_test)
pred_time_real = np.expm1(pred_time)
y_test_real = np.expm1(y_test_t)
with open("correct_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("time_model.pkl", "wb") as f:
    pickle.dump(time_model, f)
with open("feature_cols.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred, zero_division=0))
print("Prob range", probs.min(), probs.max())
print("Expected Time Model")
print("Real-time MAE (seconds):", mean_absolute_error(y_test_real, pred_time_real))
print("Real-time RMSE (seconds):", np.sqrt(mean_squared_error(y_test_real, pred_time_real)))
print("Pred time range:", pred_time_real.min(), pred_time_real.max())
print(df["time_taken"].describe())
print(df["time_taken"].head(20).tolist())
print("Saved: correct_model.pkl, time_model.pkl, feature_cols.pkl")
coef = pd.Series(model.coef_[0], index=x.columns)
print(coef.filter(like="difficulty").sort_values())
    
    
    
