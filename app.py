
import streamlit as st
import joblib
import re
from pathlib import Path
from html import escape

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ComplainSense AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# THEME-AWARE CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #103B57, #087E8B);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 34px;
}

.hero p {
    color: #E1F3F5 !important;
    font-size: 16px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
}

/* Result card */
.result-card {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    padding: 22px;
    border-radius: 12px;
    border: 1px solid var(--border-color);
    margin-top: 15px;
}

.result-card h4,
.result-card p {
    color: var(--text-color) !important;
}

/* Footer */
.footer {
    text-align: center;
    color: var(--text-color);
    opacity: 0.75;
    padding: 25px;
}

/* Text area */
.stTextArea textarea {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: var(--secondary-background-color);
}

/* Buttons */
.stButton button {
    border-radius: 8px;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #2185D5;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD FINAL AI MODEL
# ==========================================

@st.cache_resource
def load_model():

    model_path = (
        Path(__file__).parent
        / "models"
        / "complainsense_final.pkl"
    )

    artifacts = joblib.load(model_path)

    return artifacts


try:
    artifacts = load_model()

    model = artifacts["model"]
    tfidf = artifacts["tfidf"]
    spelling_map = artifacts["spelling_map"]
    phrase_map = artifacts["phrase_map"]

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s{}]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def correct_spelling(text):

    for wrong, right in phrase_map.items():

        text = re.sub(
            rf"\b{re.escape(wrong)}\b",
            right,
            text
        )

    words = text.split()

    corrected_words = [
        spelling_map.get(word, word)
        for word in words
    ]

    return " ".join(corrected_words)


# ==========================================
# INTENT PREDICTION
# ==========================================

def predict_intent(message):

    cleaned = clean_text(message)
    corrected = correct_spelling(cleaned)

    vector = tfidf.transform([corrected])

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities) * 100

    return prediction, confidence


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🤖 ComplainSense AI")

    st.caption("Customer Support Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "About"]
    )

    st.divider()

    st.success("AI Model: Active")

    st.caption("27 Intent Categories")

    st.caption("Powered by Machine Learning")


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>ComplainSense AI</h1>
        <p>Intelligent Customer Complaint Analysis System</p>
        <p>Understand customer messages through AI-powered intent classification.</p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Model", "Active")

    with col2:
        st.metric("Intent Classes", "27")

    with col3:
        st.metric("Test Accuracy", "99.74%")

    st.write("")

    st.subheader("📝 Customer Message Analyzer")

    st.write(
        "Enter a customer message below to identify its intent."
    )

    customer_message = st.text_area(
        "Customer Message",
        placeholder="Example: I want to track my order...",
        height=150
    )

    analyze = st.button(
        "🔍 Analyze Customer Message",
        type="primary",
        use_container_width=True
    )

    if analyze:

        if customer_message.strip():

            with st.spinner("Analyzing customer message..."):

                predicted_intent, confidence = predict_intent(
                    customer_message
                )

            st.divider()

            st.subheader("📊 Analysis Results")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Intent",
                    predicted_intent.replace("_", " ").title()
                )

            with col2:

                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%"
                )

            st.progress(
                min(confidence / 100, 1.0)
            )

            if confidence < 50:

                st.warning(
                    "Low confidence: Human review is recommended."
                )

            elif confidence < 70:

                st.info(
                    "Moderate confidence: Please review the prediction."
                )

            else:

                st.success(
                    "The model identified the customer's intent with good confidence."
                )

            safe_message = escape(customer_message)
            safe_intent = escape(str(predicted_intent))

            st.markdown(
                f"""
                <div class="result-card">
                    <h4>Customer Message</h4>
                    <p>{safe_message}</p>
                    <h4>Detected Intent</h4>
                    <p>{safe_intent}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning("Please enter a customer message first.")


# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "About":

    st.title("About ComplainSense AI")

    st.write("""
    ComplainSense AI is an intelligent customer support
    classification system.

    It uses Natural Language Processing and Machine Learning
    to classify customer messages into 27 different intents.

    **Technology Stack**

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - TF-IDF
    - Logistic Regression
    """)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    ComplainSense AI | Intelligent Customer Support System
</div>
""", unsafe_allow_html=True)
