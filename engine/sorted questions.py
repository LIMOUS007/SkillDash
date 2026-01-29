import pandas as pd

df = pd.read_csv("data_log_new.csv")

# extract subfamily from task_id
df["task_subfamily"] = df["task_id"].str.split("-").str[1]

# keep only what you want to inspect
cols = [
    "skill",
    "task_subfamily",
    "task_id",
    "difficulty_before",
    "question",
    "answer",
    "time_taken",
    "is_correct"
]

df_view = df[cols].copy()
df_sorted = df_view.sort_values(
    by=["skill", "task_subfamily", "difficulty_before"],
    ascending=[True, True, True]
)
squares = df_sorted[df_sorted["task_subfamily"] == "COMP"]

for _, row in squares.iterrows():
    print("Difficulty:", row["difficulty_before"])
    print("Q:", row["question"])
    print("A:", row["answer"])
    print("Correct:", row["is_correct"])
    print("-" * 40)

