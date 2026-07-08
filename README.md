---
title: Nsap Scheme Proj
emoji: ⚡
colorFrom: blue
colorTo: indigo

sdk: streamlit 

app_file: app.py
pinned: false
---
# 🏛️ NSAP Scheme Eligibility Predictor

An AI-driven predictive analytics backend built using **IBM Watson AutoAI** to automate and optimize public welfare delivery under the Government of India's **National Social Assistance Programme (NSAP)**.

---

## 🎯 Project Overview
Manual verification of social security and welfare applications is a time-consuming, error-prone process that delays vital aid to eligible citizens. This project implements a high-precision multi-class classification engine trained on verified regional and socio-economic data from the official **AI_KOSH dataset** to instantaneously map applicant profiles to the correct welfare sub-scheme.

### Core Target Schemes:
* **IGNOAPS** (Indira Gandhi National Old Age Pension Scheme)
* **IGNWPS** (Indira Gandhi National Widow Pension Scheme)
* **IGNDPS** (Indira Gandhi National Disability Pension Scheme)

---

## 🧠 Machine Learning Architecture & Pipeline

The predictive model was trained and evaluated using automated machine learning pipelines inside **IBM Cloud Pak for Data**. 

### 1. Model Selection Flow
The automated learning process systematically evaluated two distinct mathematical model branches before routing them through progressive fine-tuning stages:
* **Data Prep & Processing:** Ingested raw datasets, split training/testing holdouts, and normalized regional input metrics.
* **Algorithm Branching:** Compared an optimized **Snap Decision Tree Classifier** branch against a standard Decision Tree baseline.
* **Enhancements:** Iterated through Hyperparameter Optimization (HPO) and Feature Engineering (FE) loops.

### 2. Pipeline Leaderboard Performance
The generated classification pipelines were ranked dynamically by their optimized cross-validation accuracy score:

| Rank | Pipeline Name | Base Algorithm | Enhancements Applied | Cross-Validation Accuracy | Build Time |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **1 ★** | **Pipeline 4** | **Snap Decision Tree Classifier** | **HPO-1, FE, HPO-2** | **98.9%** | **00:00:28** |
| 2 | Pipeline 3 | Snap Decision Tree Classifier | HPO-1, FE | 97.8% | 00:00:23 |
| 3 | Pipeline 8 | Decision Tree Classifier | HPO-1, FE, HPO-2 | 96.7% | 00:00:32 |
| 4 | Pipeline 7 | Decision Tree Classifier | HPO-1, FE | 96.7% | 00:00:28 |

> **Operational Choice:** **Pipeline 4** was deployed to production due to its exceptional $98.9\%$ accuracy achieved within a nominal 28-second execution footprint.

---

## ⚙️ Technology Used
* **IBM Watson AutoAI** – For automated preprocessing, feature engineering, and pipeline selection.
* **Snap Decision Tree Classifier** – Core machine learning model chosen for multi-class classification.
* **IBM Cloud (Watson Machine Learning)** – Provides secure enterprise infrastructure hosting the live `schemep0` deployment endpoint.
* **Python Engine (Requests)** – For structured JSON payload formatting and server-side HTTPS token authentication.
* **Streamlit Framework** – Powering a clean, highly responsive, multi-column administrative input dashboard.

---

## ✨ Novelty and Uniqueness
* **Automated Feature Engineering:** Dynamically maps overlapping socio-economic indicators into predictive matrices without human bias.
* **Sub-Second Execution Velocity:** Generates feature branches and outputs definitive welfare decisions within milliseconds of endpoint invocation.
* **Transient Cryptographic Tokenization:** Isolates high-privilege architecture keys completely from the client side by handling security tokens server-side.
* **Probabilistic Scoring:** Delivers true multi-class distribution scoring to give administrators immediate insights into prediction confidence percentages.

---

## 🚀 Future Scope
1. **Centralized Government Integration:** Link directly with unified national data hubs (like the Jan Samarth Portal) to pull and verify regional demographics via live APIs.
2. **Multilingual Voice Interfaces:** Integrate speech-to-text engines to support regional languages, making the predictive assistant highly inclusive for grassroots workers.
3. **Predictive Vulnerability Mapping:** Analyze historic trend anomalies over time to forecast future welfare demands and set up proactive enrollment camps.
4. **Explainable AI (XAI):** Integrate SHAP/LIME frameworks to output exact plain-text justifications for every prediction to support transparent auditing.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
