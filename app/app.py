"""
Financial Complaint Risk Intelligence System - Dashboard
Author: Sanjeev Kumar | IIT Bombay
Banking-style complaint routing and priority triage demo
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────
st.set_page_config(
    page_title="Complaint Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, 'models')
REPORTS = os.path.join(BASE, 'reports')

# ── Load Models ───────────────────────────
@st.cache_resource
def load_models():
    models = {}
    try:
        models['product_clf'] = joblib.load(f'{MODELS}/product_classifier_champion.pkl')
    except: pass
    try:
        models['triage_lr']   = joblib.load(f'{MODELS}/triage_lr.pkl')
    except: pass
    try:
        models['triage_xgb']  = joblib.load(f'{MODELS}/triage_xgb.pkl')
    except: pass
    try:
        models['triage_tfidf']= joblib.load(f'{MODELS}/triage_tfidf.pkl')
    except: pass
    try:
        models['le']          = joblib.load(f'{MODELS}/label_encoder.pkl')
    except: pass
    return models

@st.cache_data
def load_results():
    try:
        with open(f'{REPORTS}/triage_results.json') as f:
            return json.load(f)
    except:
        return {}

models  = load_models()
results = load_results()

# ── Sidebar ───────────────────────────────
st.sidebar.title("🏦 Complaint Risk Intelligence")
st.sidebar.markdown("**Author:** Sanjeev Kumar | IIT Bombay")
page = st.sidebar.radio("Navigate", [
    "📊 Executive Overview",
    "🔀 Complaint Router",
    "🚨 Priority Review Queue",
    "📈 Model Performance",
    "⚠️ Limitations"
])

# ────────────────────────────────────────
# PAGE 1: EXECUTIVE OVERVIEW
# ────────────────────────────────────────
if page == "📊 Executive Overview":
    st.title("📊 Financial Complaint Risk Intelligence System")
    st.markdown("**Banking-style NLP complaint routing and priority triage | Portfolio Project | CFPB Data**")
    st.info("This is a portfolio/demo project using CFPB public data. Not a production underwriting system.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Complaints", "200,000", help="CFPB dataset loaded")
    with col2:
        st.metric("With Narratives", "48,815 (24.4%)", help="Used for NLP modeling")
    with col3:
        st.metric("Priority Review Rate", "1.07%", help="Untimely response complaints")
    with col4:
        st.metric("Product Categories", "7", help="Grouped from 20 raw categories")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Product Distribution")
        product_data = {
            'Product': ['Credit Reporting', 'Debt Collection', 'Credit Card',
                        'Bank Account', 'Loans', 'Mortgage', 'Money Transfer'],
            'Count': [32435, 5411, 3148, 2453, 1968, 1850, 1547]
        }
        df_prod = pd.DataFrame(product_data)
        fig = px.bar(df_prod, x='Count', y='Product', orientation='h',
                     color='Count', color_continuous_scale='Blues')
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Project Architecture")
        st.markdown("""
        **Task 1: Complaint Routing**
        - Input: Complaint narrative text
        - Output: Product category (7 classes)
        - Champion: TF-IDF + Logistic Regression
        - Test Macro-F1: **0.7652**

        **Task 2: Priority Triage**
        - Input: Complaint narrative text
        - Output: Risk score (priority review probability)
        - Metric: PR-AUC, Recall@K, Lift@K
        - NOT accuracy! (1.07% positive rate)

        **Timeline:**
        - Train: 2015–2025 | Val: 2025 | Test: 2025–2026
        """)

    st.markdown("---")
    st.subheader("Portfolio Story")
    st.info("""
    **Project 1:** Credit Card Fraud Detection → Transaction-level real-time risk
    **Project 2:** Credit Risk Scorecard → Customer-level default risk (PD × LGD × EAD)
    **Project 3:** Complaint Risk Intelligence → Customer/regulatory/operational risk

    Together = Complete Banking Risk Intelligence Portfolio 🏆
    """)

# ────────────────────────────────────────
# PAGE 2: COMPLAINT ROUTER
# ────────────────────────────────────────
elif page == "🔀 Complaint Router":
    st.title("🔀 Complaint Product Router")
    st.markdown("Paste a complaint narrative to predict product category.")
    st.warning("Illustrative demo — model trained on CFPB data 2015–2025")

    text_input = st.text_area(
        "Complaint Narrative",
        placeholder="e.g. I have incorrect information on my credit report from Equifax...",
        height=150
    )

    if st.button("🔍 Predict Product", type="primary"):
        if text_input.strip():
            if 'product_clf' in models:
                proba = models['product_clf'].predict_proba([text_input])[0]
                classes = models['product_clf'].classes_
                pred_class = classes[np.argmax(proba)]
                pred_conf  = np.max(proba)

                col1, col2 = st.columns(2)
                with col1:
                    color = "🟢" if pred_conf > 0.7 else "🟡" if pred_conf > 0.4 else "🔴"
                    st.metric("Predicted Product", f"{color} {pred_class}")
                    st.metric("Confidence", f"{pred_conf:.1%}")

                with col2:
                    df_proba = pd.DataFrame({
                        'Product': classes,
                        'Probability': proba
                    }).sort_values('Probability', ascending=True)
                    fig = px.bar(df_proba, x='Probability', y='Product',
                                 orientation='h', color='Probability',
                                 color_continuous_scale='Blues')
                    fig.update_layout(height=300, showlegend=False,
                                      title="Top Product Probabilities")
                    st.plotly_chart(fig, use_container_width=True)

                # Risk score
                if 'triage_lr' in models:
                    risk_prob = models['triage_lr'].predict_proba([text_input])[0][1]
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Priority Risk Score", f"{risk_prob:.1%}")
                    with col2:
                        if risk_prob > 0.15:
                            st.error("🚨 HIGH RISK — Urgent Review!")
                        elif risk_prob > 0.05:
                            st.warning("⚠️ MEDIUM RISK — Monitor")
                        else:
                            st.success("✅ LOW RISK — Standard Queue")
                    with col3:
                        st.metric("Baseline Rate", "1.07%")
            else:
                st.error("Models not loaded! Run notebooks 02 and 05 first.")
        else:
            st.warning("Please enter complaint text!")

    st.markdown("---")
    st.subheader("Sample Complaints")
    samples = {
        "Credit Reporting": "I have incorrect information on my credit report. Equifax is reporting an account that doesn't belong to me.",
        "Debt Collection": "A debt collector keeps calling me about a debt I don't owe. They are harassing me multiple times a day.",
        "Credit Card": "There is an unauthorized charge on my credit card. I never made this purchase and want it reversed immediately.",
        "Mortgage": "My mortgage servicer has not applied my payments correctly and is claiming I am behind on my loan.",
    }
    for product, sample in samples.items():
        if st.button(f"Try: {product}"):
            st.text_area("Sample complaint:", sample, height=100)

# ────────────────────────────────────────
# PAGE 3: PRIORITY REVIEW QUEUE
# ────────────────────────────────────────
elif page == "🚨 Priority Review Queue":
    st.title("🚨 Priority Review Queue Simulator")
    st.markdown("Simulate complaint risk triage with threshold control.")

    threshold = st.slider("Risk Score Threshold", 0.01, 0.50, 0.10, 0.01)

    sample_complaints = [
        {"text": "My credit report has errors and they refuse to fix it despite multiple disputes.", "product": "Credit Reporting"},
        {"text": "Unauthorized charges on my credit card that the company refuses to reverse.", "product": "Credit Card"},
        {"text": "Debt collector is threatening me illegally. I do not owe this debt.", "product": "Debt Collection"},
        {"text": "My mortgage company lost my payment and is now charging late fees.", "product": "Mortgage"},
        {"text": "Student loan servicer applied payment to wrong account causing default.", "product": "Loans"},
        {"text": "Bank froze my account without notice and I cannot access my funds.", "product": "Bank Account"},
        {"text": "Wire transfer never arrived and company refuses to investigate.", "product": "Money Transfer"},
        {"text": "Simple question about my account balance and statement.", "product": "Bank Account"},
    ]

    if 'triage_lr' in models and 'product_clf' in models:
        rows = []
        for s in sample_complaints:
            risk  = models['triage_lr'].predict_proba([s['text']])[0][1]
            prod  = models['product_clf'].predict([s['text']])[0]
            rows.append({
                'Risk Score': risk,
                'Product': prod,
                'Snippet': s['text'][:80] + '...',
                'Priority': '🚨 High' if risk > threshold else ('⚠️ Medium' if risk > threshold/2 else '✅ Low'),
                'Action': 'Urgent Review' if risk > threshold else ('SLA Monitor' if risk > threshold/2 else 'Standard Queue')
            })

        df_queue = pd.DataFrame(rows).sort_values('Risk Score', ascending=False)
        df_queue['Risk Score'] = df_queue['Risk Score'].apply(lambda x: f"{x:.1%}")

        high_risk = (df_queue['Priority'].str.contains('High')).sum()
        st.metric("Complaints in High-Risk Queue", f"{high_risk}/{len(df_queue)}")

        st.dataframe(df_queue, use_container_width=True, height=400)
    else:
        st.error("Models not loaded! Run notebooks 02 and 05 first.")
        st.info("Sample queue preview:")
        st.table(pd.DataFrame([
            {"Priority": "🚨 High", "Risk Score": "0.82", "Product": "Credit Card", "Action": "Urgent Review"},
            {"Priority": "⚠️ Medium", "Risk Score": "0.31", "Product": "Mortgage", "Action": "SLA Monitor"},
            {"Priority": "✅ Low", "Risk Score": "0.04", "Product": "Bank Account", "Action": "Standard Queue"},
        ]))

# ────────────────────────────────────────
# PAGE 4: MODEL PERFORMANCE
# ────────────────────────────────────────
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    st.subheader("Product Classification Results")
    comp_data = {
        'Model': ['TF-IDF + LR', 'TF-IDF + SVM', 'Word2Vec + XGB',
                  'Frozen BERT CLS + LR', 'Frozen BERT Mean + LR', 'Fine-tuned DistilBERT'],
        'Val Macro-F1': [0.7696, 0.7803, 0.6971, 0.6507, 0.6703, 0.7968],
        'Test Macro-F1': [0.7652, 0.7591, 0.6383, 0.6716, 0.6737, 0.7722],
        'Decision': ['Champion ✅', 'Strong baseline', 'Underperformed ❌',
                     'Underperformed ❌', 'Underperformed ❌', 'Challenger']
    }
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Val F1', x=comp_data['Model'], y=comp_data['Val Macro-F1'], marker_color='steelblue'))
    fig.add_trace(go.Bar(name='Test F1', x=comp_data['Model'], y=comp_data['Test Macro-F1'], marker_color='coral'))
    fig.update_layout(barmode='group', title='Model Comparison', xaxis_tickangle=-30, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Insight: Why TF-IDF beats Frozen BERT")
    st.info("""
    Product categories are **keyword-heavy**:
    - "credit report", "Equifax" → Credit Reporting
    - "debt collector", "FDCPA" → Debt Collection
    - "mortgage", "foreclosure" → Mortgage

    TF-IDF preserves these exact keywords. Frozen BERT compresses everything into 768 numbers without learning these task-specific patterns.

    **Fine-tuned DistilBERT** learns the patterns (+0.007 lift) but the improvement is marginal — not enough to justify extra complexity.

    **This is senior DS thinking: don't blindly use BERT!**
    """)

    st.markdown("---")
    st.subheader("Triage Model Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Random baseline PR-AUC", "0.0107", help="1.07% positive rate")
    with col2:
        pr = results.get('triage_lr_prauc', 'Run notebook 05')
        st.metric("TF-IDF + LR PR-AUC", f"{pr:.4f}" if isinstance(pr, float) else pr)
    with col3:
        pr = results.get('triage_xgb_prauc', 'Run notebook 05')
        st.metric("XGBoost PR-AUC", f"{pr:.4f}" if isinstance(pr, float) else pr)

# ────────────────────────────────────────
# PAGE 5: LIMITATIONS
# ────────────────────────────────────────
elif page == "⚠️ Limitations":
    st.title("⚠️ Limitations & Leakage Controls")

    st.subheader("Leakage Controls")
    st.success("""
    **Features ALLOWED at prediction time:**
    - Complaint narrative text (available at submission)
    - Submission channel (Web/Phone/etc)
    - State (available at submission)
    - Date received (available at submission)
    """)
    st.error("""
    **Features EXCLUDED (post-resolution — leakage risk!):**
    - Company response to consumer
    - Timely response flag
    - Consumer disputed flag
    - Date sent to company
    - Any resolution outcomes
    """)

    st.markdown("---")
    st.subheader("Known Limitations")
    limitations = {
        "CFPB data is US-only": "Not directly applicable to India-specific banking context",
        "Narrative consent required": "Only 24.4% of complaints have narrative text — selection bias!",
        "Untimely response proxy": "1.07% positives. This is a proxy label, not ground-truth complaint severity",
        "Recent data censoring": "Complaints from last 30-60 days may have incomplete outcomes",
        "Not representative": "CFPB explicitly warns data is not a statistical sample of all consumers",
        "Marginal BERT lift": "Fine-tuned DistilBERT only +0.007 over TF-IDF — keywords dominate",
        "No reject inference": "We only see complaints submitted — survivorship bias possible",
        "LGD/EAD not applicable": "This is operational risk, not credit loss modeling",
    }
    for k, v in limitations.items():
        st.warning(f"**{k}:** {v}")

    st.markdown("---")
    st.subheader("Interview Story (90 seconds)")
    st.code("""
I built a Financial Complaint Risk Intelligence System using CFPB
complaint narratives. The system has two goals:

1. Route complaints to the right product queue
2. Prioritize complaints likely to breach response SLAs

I used chronological validation to simulate future complaint
performance. For product routing, I compared TF-IDF, SVM,
Word2Vec, frozen DistilBERT, and fine-tuned DistilBERT.

TF-IDF + LR achieved Test Macro-F1 = 0.765 and was selected
as champion. Fine-tuned DistilBERT gave only +0.007 lift —
not enough to justify the added complexity, latency, and
governance overhead.

For priority triage, I treated untimely response as a rare-event
proxy (1.07% positive) and evaluated using PR-AUC, Recall@K,
Precision@K, and Lift — NOT accuracy.

The key lesson: I did not blindly use BERT. I benchmarked
systematically and selected models based on generalization,
interpretability, and business value.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Portfolio Project | Not for production use")
st.sidebar.markdown("**github.com/17sanjeev**")
