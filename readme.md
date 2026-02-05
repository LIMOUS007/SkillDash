# Skilldash V1 — Adaptive Reasoning Trainer  

This project implements an **adaptive reasoning system** that dynamically adjusts *difficulty levels* based on user performance while continuously collecting data for future machine learning.

It supports three cognitive skills:  

- Numerical reasoning  
- Logical reasoning  
- Pattern recognition  

Each skill contains multiple **procedural task generators** that create unlimited questions at different difficulty levels.

---

## How it Works (High Level)

1. A task is generated procedurally from one of several subfamilies.  
2. A pretrained ML model predicts:  
   - Probability the user will answer correctly (`p_correct`)  
   - Expected response time (`expected_time`)  
3. The user answers the question.  
4. Difficulty is updated using a rule-based controller that uses correctness, time, and ML predictions.  
5. Every attempt and task feature is logged for analysis and future learning.  

This creates a closed-loop **adaptive tutoring system with data collection for Task ML.**

> In V1, ML models only predict performance — they do **not** choose or modify tasks.  
> Adaptation happens via difficulty updates, not task parameters.

---

## What’s Implemented in V1  

- Procedural task generation (no fixed question bank)  
- Independent adaptive difficulty for each skill  
- Detailed user attempt logging (`data_log_final.csv`)  
- Task feature logging (`task_feature_ml.csv`)  
- Session summaries (`session_summary.csv`)  
- ML models for:
  - predicting correctness  
  - predicting response time  

---

## Data Collection  

Every session automatically records:  

- User responses, timing, and correctness  
- Difficulty transitions per skill  
- Task-level structural features  

New data can be committed after running sessions so models can be retrained over time.
Commit new versions of `data_log_final.csv` and `task_feature_ml.csv` after running meaningful sessions.
---

## Future Work (V2): Task ML  

Instead of only adjusting scalar difficulty, V2 will aim to learn how to **adapt task parameters directly**, such as:  

- Number of objects in logic problems  
- Depth of implication chains  
- Noise in comparison tasks  
- Step size in arithmetic progressions  
- Multipliers in multiplicative patterns  

V1 exists primarily to **generate high-quality training data** for this next phase.

---

## How to Run  

```bash
pip install -r requirements.txt
python session.py
