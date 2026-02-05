# Skilldash V1 — Adaptive Reasoning Trainer

This project implements an **adaptive reasoning system** that dynamically adjusts task difficulty based on user performance using machine learning.

It supports three cognitive skills:

- **Numerical reasoning**
- **Logical reasoning**
- **Pattern recognition**

Each skill contains multiple procedural task generators that create unlimited questions at different difficulty levels.

---

## How it Works (High Level)

1. A task is generated procedurally.
2. A pretrained ML model predicts:
   - Probability the user will answer correctly (`p_correct`)
   - Expected response time (`expected_time`)
3. The user answers.
4. Difficulty is updated using a principled ML-based rule.
5. Everything is logged for future learning.

This forms a closed-loop **adaptive tutoring system**.

---

## What’s Implemented in V1

✅ Procedural task generation  
✅ Adaptive difficulty per skill  
✅ User performance logging  
✅ Session summaries  
✅ ML models for:
- predicting correctness  
- predicting response time  

🚧 **Future Work (V2): Task ML**
- Learn how to adjust *task parameters themselves* rather than just difficulty level.

---

## How to Run

```bash
pip install -r requirements.txt
python engine/session.py
