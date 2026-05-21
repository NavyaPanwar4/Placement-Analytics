# 🎓 Placement Analytics

A machine learning–powered web app that analyzes campus recruitment data and predicts student placement outcomes.

**[🚀 Live Demo →](https://placementai.streamlit.app/)**

---

## Overview

Placement Analytics is an interactive dashboard built with Streamlit that helps visualize placement trends and predict whether a student is likely to get placed — based on academic scores, degree type, work experience, and MBA specialization.

The app is trained on real campus recruitment data and exposes a clean UI with three sections:

- **Dashboard** — High-level KPIs, placement rates, gender breakdowns, and score comparisons
- **Placement Predictor** — Enter a student profile and get an instant ML-based placement prediction with a probability gauge
- **Detailed Analysis** — Score distributions, salary analysis, and placement rates by stream and degree type

---

## Features

- Interactive charts powered by Plotly
- ML model with accuracy and ROC-AUC metrics shown in-app
- Personalized suggestions based on student input (weak scores, lack of work experience, etc.)
- Supports both light and dark themes natively
- Fully deployed on Streamlit Cloud

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit, Plotly |
| ML | scikit-learn (model + scaler + label encoders) |
| Data | Pandas, NumPy |
| Deployment | Streamlit Cloud |
| Notebook | Jupyter |

---

## Project Structure

```
Placement-Analytics/
├── app/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── label_encoders.pkl
│   └── model_metadata.json
├── data/
│   └── Placement_Data_Full_Class.csv
├── notebook/
│   └── (EDA & model training)
├── app.py
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/NavyaPanwar4/Placement-Analytics.git
cd Placement-Analytics
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

- **Source:** [benroshan on Kaggle](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)
- 215 students, 14 features including academic scores, degree type, work experience, and salary

---

## Model

The best-performing classifier is selected automatically during training and saved to `app/model.pkl`. Key metrics are stored in `app/model_metadata.json` and displayed live in the app sidebar and predictor page.

---

## Live App

👉 [placementai.streamlit.app](https://placementai.streamlit.app/)

---

## Author

**Navya Panwar** — [GitHub](https://github.com/NavyaPanwar4)
