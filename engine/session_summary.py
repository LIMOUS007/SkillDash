import csv
import os
from datetime import datetime
def log_session_summary(user_id, session_id, total_attempts, total_correct, start_time_iso, end_time_iso):
    csv_file = os.path.join(os.getcwd(), "session_summary.csv")
    fields = [
        "timestamp",
        "user_id",
        "session_id",
        "start_time",
        "end_time",
        "total_attempts",
        "total_correct",
        "accuracy"
    ]
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        acc = (total_correct / total_attempts) if total_attempts > 0 else 0.0
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "session_id": session_id,
            "start_time": start_time_iso,
            "end_time": end_time_iso,
            "total_attempts": total_attempts,
            "total_correct": total_correct,
            "accuracy": round(acc, 4)
        })
