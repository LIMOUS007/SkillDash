import pandas as pd

df0 = pd.read_csv("data_log1.csv")
print("Raw rows:", len(df0))
print(df0.info())

cols = [
    "difficulty_before",
    "difficulty_after",
    "skill",
    "task_id",
    "answer_type",
    "time_taken",
    "is_correct",
]
df = df0[cols].copy()

print("After selecting cols:", len(df))
df["is_correct"] = df["is_correct"].astype(str).str.strip().str.lower()
df["is_correct"] = df["is_correct"].map({
    "1": 1, "1.0": 1, "true": 1, "yes": 1,
    "0": 0, "0.0": 0, "false": 0, "no": 0
})
df["difficulty_before"] = pd.to_numeric(df["difficulty_before"], errors="coerce")
df["difficulty_after"] = pd.to_numeric(df["difficulty_after"], errors="coerce")
df["time_taken"] = pd.to_numeric(df["time_taken"], errors="coerce")
df["is_correct"] = pd.to_numeric(df["is_correct"], errors="coerce")

print("Rows with any NaN:", df.isna().any(axis=1).sum())

df = df.dropna()
print("After dropna:", len(df))

print("Rows time_taken <=0:", (df["time_taken"] <= 0).sum())
print("Rows time_taken >=120:", (df["time_taken"] >= 120).sum())

df = df[(df["time_taken"] > 0) & (df["time_taken"] < 120)]
print("After time filter:", len(df))

df = df[df["is_correct"].isin([0, 1])]
print("After is_correct filter:", len(df))
