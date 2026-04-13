"""
SMSGuard Dashboard  —  run with:  streamlit run src/dashboard.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)
from sklearn.model_selection import train_test_split

from preprocess import preprocess_dataframe
from feature_extraction import vectorize_messages
from utils import load_combined_data

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMSGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ────────────────────────────────────────────────────────────
SPAM_COLOR = "#EF4444"
HAM_COLOR  = "#22C55E"
BG_COLOR   = "#0F172A"
CARD_COLOR = "#1E293B"


# ═════════════════════════════════════════════════════════════════════════════
# Data & model loaders  (cached so they don't reload on every interaction)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    """Try processed CSV first; fall back to raw files."""
    processed = "data/processed/sms_cleaned.csv"
    if os.path.exists(processed):
        df = pd.read_csv(processed)
    else:
        uci    = "data/raw/SMSSpamCollection"
        kaggle = "data/raw/spam.csv"
        if not os.path.exists(uci) or not os.path.exists(kaggle):
            return None
        df = load_combined_data(uci, kaggle)
        df = preprocess_dataframe(df)
        df["label_binary"] = (df["label"] == "spam").astype(int)
        df["word_count"]   = df["message"].str.split().str.len()
        df["char_count"]   = df["message"].str.len()
    return df


@st.cache_resource(show_spinner="Loading model…")
def load_model_and_vectorizer():
    model_path = "models/trained_model.pkl"
    vec_path   = "models/vectorizer.pkl"
    if os.path.exists(model_path) and os.path.exists(vec_path):
        return joblib.load(model_path), joblib.load(vec_path)
    return None, None


@st.cache_data(show_spinner="Computing model metrics…")
def compute_metrics(_df):
    """Re-run train/test split and return metrics + curves."""
    X, vec = vectorize_messages(_df["cleaned"])
    y = _df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, list(model.classes_).index("spam")]

    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label="spam")
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])

    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam"),
        "recall":    recall_score(y_test, y_pred,    pos_label="spam"),
        "f1":        f1_score(y_test, y_pred,        pos_label="spam"),
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "cm": cm,
        "y_test": y_test.values, "y_pred": y_pred,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    .main { background-color: #0F172A; }
    .metric-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #F8FAFC; }
    .metric-label { font-size: 0.85rem; color: #94A3B8; margin-top: 4px; }
    .section-title {
        font-size: 1.1rem; font-weight: 600;
        color: #CBD5E1; margin-bottom: 12px;
    }
    div[data-testid="stMetric"] { background:#1E293B; border-radius:10px; padding:12px; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🛡️ SMSGuard")
    st.markdown("AI-powered SMS spam detection")
    st.divider()
    page = st.radio(
        "Navigation",
        ["📊 Data Overview", "🤖 Model Performance", "🔍 Live Predictor"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Built with Python · Scikit-learn · Streamlit")


# ═════════════════════════════════════════════════════════════════════════════
# Load everything
# ═════════════════════════════════════════════════════════════════════════════

df    = load_data()
model, vectorizer = load_model_and_vectorizer()

if df is None:
    st.error("⚠️ Dataset files not found. Make sure `data/raw/` contains the SMS datasets.")
    st.stop()

# Ensure cleaned column exists
if "cleaned" not in df.columns and "cleaned_message" in df.columns:
    df = df.rename(columns={"cleaned_message": "cleaned"})
elif "cleaned" not in df.columns:
    df = preprocess_dataframe(df)

if "word_count" not in df.columns:
    df["word_count"] = df["message"].str.split().str.len()
if "char_count" not in df.columns:
    df["char_count"] = df["message"].str.len()
if "label_binary" not in df.columns:
    df["label_binary"] = (df["label"] == "spam").astype(int)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Data Overview
# ═════════════════════════════════════════════════════════════════════════════

if page == "📊 Data Overview":
    st.title("📊 Data Overview")

    # KPI row
    total   = len(df)
    n_spam  = df["label_binary"].sum()
    n_ham   = total - n_spam
    spam_pct = n_spam / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Messages", f"{total:,}")
    c2.metric("Spam",  f"{n_spam:,}",  f"{spam_pct:.1f}%")
    c3.metric("Ham",   f"{n_ham:,}",   f"{100-spam_pct:.1f}%")
    c4.metric("Avg Word Count", f"{df['word_count'].mean():.1f}")

    st.divider()
    col_left, col_right = st.columns(2)

    # Donut — class distribution
    with col_left:
        st.markdown('<p class="section-title">Class Distribution</p>', unsafe_allow_html=True)
        fig_donut = px.pie(
            values=[n_spam, n_ham], names=["Spam", "Ham"],
            color_discrete_sequence=[SPAM_COLOR, HAM_COLOR],
            hole=0.55,
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1", margin=dict(t=10, b=10),
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Box plot — message length by label
    with col_right:
        st.markdown('<p class="section-title">Message Length by Class</p>', unsafe_allow_html=True)
        fig_box = px.box(
            df, x="label", y="char_count",
            color="label",
            color_discrete_map={"spam": SPAM_COLOR, "ham": HAM_COLOR},
            labels={"char_count": "Character Count", "label": ""},
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1", showlegend=False, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Word count distribution
    st.markdown('<p class="section-title">Word Count Distribution</p>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        df[df["word_count"] < 60], x="word_count",
        color="label", barmode="overlay", nbins=50,
        color_discrete_map={"spam": SPAM_COLOR, "ham": HAM_COLOR},
        labels={"word_count": "Word Count", "count": "Messages"},
        opacity=0.75,
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    metrics = compute_metrics(df)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics['accuracy']:.1%}")
    c2.metric("Precision", f"{metrics['precision']:.1%}")
    c3.metric("Recall",    f"{metrics['recall']:.1%}")
    c4.metric("F1 Score",  f"{metrics['f1']:.1%}")

    st.divider()
    col_left, col_right = st.columns(2)

    # Confusion matrix
    with col_left:
        st.markdown('<p class="section-title">Confusion Matrix</p>', unsafe_allow_html=True)
        cm = metrics["cm"]
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Spam", "Ham"], y=["Spam", "Ham"],
            text_auto=True,
            color_continuous_scale=[[0, "#1E293B"], [1, "#6366F1"]],
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1", margin=dict(t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ROC curve
    with col_right:
        st.markdown(
            f'<p class="section-title">ROC Curve  (AUC = {metrics["roc_auc"]:.3f})</p>',
            unsafe_allow_html=True,
        )
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=metrics["fpr"], y=metrics["tpr"],
            mode="lines", name=f"AUC = {metrics['roc_auc']:.3f}",
            line=dict(color="#6366F1", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="#475569", dash="dash"), showlegend=False,
        ))
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CBD5E1",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Metric bar chart
    st.markdown('<p class="section-title">Metrics Summary</p>', unsafe_allow_html=True)
    metric_names  = ["Accuracy", "Precision", "Recall", "F1 Score"]
    metric_values = [
        metrics["accuracy"], metrics["precision"],
        metrics["recall"],   metrics["f1"],
    ]
    fig_bar = px.bar(
        x=metric_names, y=metric_values,
        color=metric_values,
        color_continuous_scale=[[0, "#334155"], [1, "#6366F1"]],
        text=[f"{v:.1%}" for v in metric_values],
        range_y=[0, 1.05],
        labels={"x": "", "y": "Score"},
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#CBD5E1", coloraxis_showscale=False, margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Live Predictor
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Live Predictor":
    st.title("🔍 Live Spam Predictor")
    st.markdown("Type or paste an SMS message below to classify it in real time.")

    user_input = st.text_area(
        "SMS Message",
        placeholder="e.g. Congratulations! You've won a £1000 Tesco gift card. Call now!",
        height=120,
    )

    if st.button("🔍 Classify", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a message first.")
        elif model is None or vectorizer is None:
            st.error("Model not found. Run `python src/main.py` to train first.")
        else:
            # Minimal preprocessing (match training pipeline)
            from preprocess import clean_text
            cleaned = clean_text(user_input)
            X       = vectorizer.transform([cleaned])
            pred    = model.predict(X)[0]
            proba   = model.predict_proba(X)[0]
            spam_p  = proba[list(model.classes_).index("spam")]
            ham_p   = 1 - spam_p

            is_spam = pred == "spam"
            color   = SPAM_COLOR if is_spam else HAM_COLOR
            icon    = "🚨" if is_spam else "✅"
            label   = "SPAM" if is_spam else "HAM"

            st.markdown(f"""
            <div style="background:{CARD_COLOR}; border:2px solid {color};
                        border-radius:12px; padding:24px; text-align:center; margin-top:12px;">
                <div style="font-size:3rem">{icon}</div>
                <div style="font-size:1.8rem; font-weight:700; color:{color}">{label}</div>
                <div style="color:#94A3B8; margin-top:8px">
                    Spam confidence: <b style="color:{SPAM_COLOR}">{spam_p:.1%}</b>
                    &nbsp;|&nbsp;
                    Ham confidence: <b style="color:{HAM_COLOR}">{ham_p:.1%}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=spam_p * 100,
                title={"text": "Spam Probability", "font": {"color": "#CBD5E1"}},
                number={"suffix": "%", "font": {"color": "#F8FAFC"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569"},
                    "bar":  {"color": color},
                    "bgcolor": CARD_COLOR,
                    "steps": [
                        {"range": [0, 40],  "color": "#14532D"},
                        {"range": [40, 70], "color": "#713F12"},
                        {"range": [70, 100],"color": "#7F1D1D"},
                    ],
                    "threshold": {
                        "line": {"color": "#F8FAFC", "width": 2},
                        "thickness": 0.75, "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#CBD5E1",
                height=280,
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
