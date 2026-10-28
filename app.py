import json
import os
import pickle
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

st.set_page_config(page_title="ScamSense AI", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_ml_assets():
    with open("spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    ml_model, tfidf = load_ml_assets()
except Exception as e:
    st.error("ML Model ફાઇલો નથી મળી! ખાતરી કરો કે spam_model.pkl અને tfidf_vectorizer.pkl એક જ ફોલ્ડરમાં છે.")
    st.stop()

def analyze_with_genai(message: str, api_key: str):
    if not api_key:
        return {
            "risk_level": "Unknown",
            "tactics": ["API Key Not Provided"],
            "suspicious_phrases": [],
            "advice": "સાઇડબારમાં Gemini API Key ઉમેરો જેથી સંપૂર્ણ ડીપ એનાલિસિસ મળે."
        }

    client = genai.Client(api_key=api_key)
    prompt = f"""
    Analyze this message for fraud, phishing, or financial scams:
    Message: "{message}"

    Return strictly a valid JSON object matching this schema:
    {{
        "risk_level": "High" | "Medium" | "Low",
        "tactics": ["Urgency", "Impersonation", "Greed", "Malicious Link", etc.],
        "suspicious_phrases": ["phrase 1", "phrase 2"],
        "advice": "1-2 lines of direct action guidance for the user"
    }}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "risk_level": "Error",
            "tactics": [],
            "suspicious_phrases": [],
            "advice": f"API Error: {str(e)}"
        }

st.title("🛡️ ScamSense AI — Threat & Fraud Detection Engine")
st.caption("A dual-layer security engine: Rapid ML classification + Generative AI psychological reasoning.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key_input = st.text_input("Gemini API Key", type="password", help="aistudio.google.com પરથી ફ્રી API Key મેળવો")
    st.markdown("---")
    st.markdown("**Architecture Flow:**")
    st.markdown("1. **TF-IDF + Logistic Reg**: First-pass speed filter.")
    st.markdown("2. **Gemini 2.5 Flash**: Zero-shot psychological vulnerability parsing & advice.")

tabs = st.tabs(["🔍 Single Message Scanner", "📁 Batch CSV Scanner"])

with tabs[0]:
    user_msg = st.text_area("તપાસવા માટેનો SMS, WhatsApp કે Email મેસેજ અહીં પેસ્ટ કરો:", height=120)
    
    if st.button("Analyze Threat", type="primary"):
        if not user_msg.strip():
            st.warning("કૃપા કરીને પહેલા કોઈ મેસેજ દાખલ કરો.")
        else:
            vec = tfidf.transform([user_msg])
            prob = ml_model.predict_proba(vec)[0][1]
            ml_pred = "Spam/Scam" if prob > 0.5 else "Clean/Ham"

            with st.spinner("AI એનાલિસિસ ચાલી રહ્યું છે..."):
                genai_res = analyze_with_genai(user_msg, api_key_input)

            col1, col2, col3 = st.columns(3)
            col1.metric("ML Threat Probability", f"{prob*100:.1f}%")
            col2.metric("ML Heuristic Verdict", ml_pred)
            col3.metric("GenAI Threat Rating", genai_res.get("risk_level", "N/A"))

            st.subheader("🧐 Explainable Threat Insights")
            if genai_res.get("suspicious_phrases"):
                st.write("**શંકાસ્પદ શબ્દો (Triggers):**", ", ".join([f"`{p}`" for p in genai_res["suspicious_phrases"]]))
            if genai_res.get("tactics"):
                st.write("**વપરાયેલી યુક્તિઓ (Tactics):**", ", ".join(genai_res["tactics"]))
            
            st.info(f"💡 **સુરક્ષા સલાહ:** {genai_res.get('advice')}")

with tabs[1]:
    st.write("એકસાથે ઘણા મેસેજ સ્કેન કરવા માટે CSV ફાઇલ અપલોડ કરો (જેમાં `message` નામની કોલમ હોવી જોઈએ).")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        if "message" not in df_upload.columns:
            st.error("CSV ફાઇલમાં 'message' નામની કોલમ હોવી જરૂરી છે.")
        else:
            if st.button("Start Batch Processing"):
                with st.spinner("બેચ સ્કેનિંગ ચાલુ છે..."):
                    cleaned_messages = df_upload["message"].fillna("").astype(str)
                    vectors = tfidf.transform(cleaned_messages)
                    probabilities = ml_model.predict_proba(vectors)[:, 1]
                    df_upload["Scam_Probability"] = (probabilities * 100).round(2)
                    df_upload["Verdict"] = ["High Risk" if p > 0.5 else "Low Risk" for p in probabilities]

                st.success("સ્કેનિંગ પૂર્ણ થયું!")
                st.dataframe(df_upload.head(10))

                csv_data = df_upload.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Audit Report (CSV)",
                    data=csv_data,
                    file_name="scamsense_audit_report.csv",
                    mime="text/csv"
                )
