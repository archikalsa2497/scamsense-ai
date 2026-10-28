import streamlit as st
import pandas as pd
import joblib
import os
import google.generativeai as genai

st.set_page_config(page_title="ScamSense AI", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_ml_components():
    model_path = "spam_model.pkl"
    vec_path = "tfidf_vectorizer.pkl"
    if os.path.exists(model_path) and os.path.exists(vec_path):
        model = joblib.load(model_path)
        vec = joblib.load(vec_path)
        return model, vec
    return None, None

ml_model, tfidf = load_ml_components()

def analyze_with_gemini(api_key, text):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Analyze this message for social engineering, phishing, or scam tactics:
        "{text}"
        
        Provide the response in the following exact format:
        THREAT_LEVEL: [Low/Medium/High/Critical]
        TACTICS: [Comma separated tactics, e.g., Urgency, Financial Lure, OTP Request, None]
        ADVICE: [One actionable sentence for the user]
        """
        response = model.generate_content(prompt)
        lines = response.text.strip().split('\n')
        result = {"threat": "Unknown", "tactics": [], "advice": "No specific advice generated."}
        for line in lines:
            if line.startswith("THREAT_LEVEL:"):
                result["threat"] = line.replace("THREAT_LEVEL:", "").strip()
            elif line.startswith("TACTICS:"):
                result["tactics"] = [t.strip() for t in line.replace("TACTICS:", "").strip().split(",")]
            elif line.startswith("ADVICE:"):
                result["advice"] = line.replace("ADVICE:", "").strip()
        return result
    except Exception as e:
        return {"threat": "Error", "tactics": ["API Error"], "advice": str(e)}

st.title("🛡️ ScamSense AI — Threat & Fraud Detection Engine")
st.markdown("A dual-layer security engine: Rapid ML classification + Generative AI psychological reasoning.")

with st.sidebar:
    st.header("⚙️ Settings")
    user_api_key = st.text_input("Gemini API Key", type="password", help="Get a free API key from aistudio.google.com")
    st.markdown("---")
    st.markdown("### Architecture Flow:")
    st.markdown("1. **TF-IDF + Logistic Reg:** First-pass speed filter.")
    st.markdown("2. **Gemini 2.5 Flash:** Zero-shot psychological vulnerability parsing & advice.")

tabs = st.tabs(["🔍 Single Message Scanner", "📁 Batch CSV Scanner"])

with tabs[0]:
    msg_input = st.text_area("Paste SMS, WhatsApp, or Email message here to analyze:", height=120)
    
    if st.button("Analyze Threat", type="primary"):
        if not msg_input.strip():
            st.warning("Please enter a message to analyze.")
        elif ml_model is None or tfidf is None:
            st.error("ML model files are missing. Please verify repository files.")
        else:
            vec = tfidf.transform([msg_input])
            prob = ml_model.predict_proba(vec)[0][1]
            verdict = "Spam/Scam" if prob > 0.5 else "Clean/Ham"
            
            col1, col2, col3 = st.columns(3)
            col1.metric("ML Threat Probability", f"{prob*100:.1f}%")
            col2.metric("ML Heuristic Verdict", verdict)
            
            genai_res = None
            if user_api_key:
                with st.spinner("Analyzing psychological vectors with Gemini 2.5 Flash..."):
                    genai_res = analyze_with_gemini(user_api_key, msg_input)
                    col3.metric("GenAI Threat Rating", genai_res.get("threat", "Unknown"))
            else:
                col3.metric("GenAI Threat Rating", "Unknown")
            
            st.markdown("---")
            st.subheader("🧐 Explainable Threat Insights")
            
            if genai_res and genai_res.get("threat") != "Error":
                tactics = ", ".join(genai_res.get("tactics", []))
                st.write(f"**Identified Tactics:** {tactics if tactics else 'None'}")
                st.info(f"💡 **Security Recommendation:** {genai_res.get('advice')}")
            elif not user_api_key:
                st.write("**Identified Tactics:** API Key Not Provided")
                st.info("💡 **Security Tip:** Add your Gemini API Key in the sidebar for complete in-depth threat analysis.")
            else:
                st.error(f"GenAI Analysis failed: {genai_res.get('advice')}")

with tabs[1]:
    st.write("Upload a CSV file containing a `message` column for bulk threat scanning.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        if "message" not in df_upload.columns:
            st.error("The CSV file must contain a 'message' column.")
        else:
            if st.button("Start Batch Processing"):
                with st.spinner("Processing batch audit..."):
                    cleaned_messages = df_upload["message"].fillna("").astype(str)
                    vectors = tfidf.transform(cleaned_messages)
                    probabilities = ml_model.predict_proba(vectors)[:, 1]
                    df_upload["Scam_Probability"] = (probabilities * 100).round(2)
                    df_upload["Verdict"] = ["High Risk" if p > 0.5 else "Low Risk" for p in probabilities]
                    
                    st.success("Batch processing completed successfully!")
                    st.dataframe(df_upload.head(10))
                    
                    csv_data = df_upload.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Audit Report (CSV)",
                        data=csv_data,
                        file_name="scamsense_audit_report.csv",
                        mime="text/csv"
                    )
