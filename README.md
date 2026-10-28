# ScamSense AI — Hybrid Threat & Fraud Detection Engine

An end-to-end cyber security detection dashboard combining classical machine learning (TF-IDF + Logistic Regression) with Generative AI (Gemini Flash) for deep explainability and psychological scam analysis.

## Features
- **Real-Time Single Scanner:** Instant probability scoring & GenAI explainability.
- **Explainable AI (XAI):** Identifies triggers (Urgency, Greed, OTP demand) and outputs tailored safety advice.
- **Batch CSV Scanner:** Bulk processing with downloadable threat audit reports.

## Setup & Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
