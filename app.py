"""
InsightForge AI — Streamlit Application
Auto-EDA system with LLM-powered insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import os

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsightForge AI",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Theme ────────────────────────────────────────────────────────────────────
_theme = {
    "bg":      "#0A0E1A",
    "surface": "#111827",
    "surface2":"#1A2235",
    "border":  "#1F2937",
    "text":    "#E2E8F0",
    "muted":   "#64748B",
    "primary": "#00D4FF",
    "secondary":"#7B61FF",
    "accent":  "#FF6B6B",
    "success": "#00E5A0",
    "warning": "#FFB800",
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --primary: """ + _theme["primary"] + """;
    --secondary: """ + _theme["secondary"] + """;
    --accent: """ + _theme["accent"] + """;
    --success: """ + _theme["success"] + """;
    --warning: """ + _theme["warning"] + """;
    --bg: """ + _theme["bg"] + """;
    --surface: """ + _theme["surface"] + """;
    --surface2: """ + _theme["surface2"] + """;
    --border: """ + _theme["border"] + """;
    --text: """ + _theme["text"] + """;
    --muted: """ + _theme["muted"] + """;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg); }

[data-testid="stSidebar"] {
    background:var(--surface) !important;
    border-right: 1px solid var(--border);
}

.insight-card {
    background:var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    position: relative;
    overflow: hidden;
}
.insight-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--primary);
}
.insight-card.anomaly::before { background: var(--accent); }
.insight-card.correlation::before { background: var(--secondary); }
.insight-card.chat::before { background: var(--success); }
.insight-card.ml::before { background: var(--warning); }

.insight-card h4 {
    margin: 0 0 12px 0;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color:var(--muted);
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.metric-box {
    background:var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-box .value {
    font-family: 'DM Mono', monospace;
    font-size: 24px;
    font-weight: 500;
    color: var(--primary);
    display: block;
}
.metric-box .label {
    font-size: 11px;
    color:var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.forge-header {
    background: linear-gradient(135deg, var(--surface) 0%, #0D1829 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.forge-header::after {
    content: '';
    position: absolute;
    right: 40px; top: 20px;
    font-size: 80px;
    opacity: 0.06;
}
.forge-header h1 {
    font-size: 32px;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}
.forge-header p {
    color:var(--muted);
    font-size: 14px;
    margin: 0;
    font-family: 'DM Mono', monospace;
}

.section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 3px;
    color:var(--muted);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

.tag { display: inline-block; background:var(--surface2); border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; font-family: 'DM Mono', monospace; font-size: 11px; color: var(--primary); margin: 2px; }
.tag.cat { color: var(--secondary); }
.tag.binary { color: var(--success); }
.tag.high { color: var(--warning); }
.tag.id { color: var(--accent); }

.llm-insight { font-size: 14px; line-height: 1.7; color: var(--text); white-space: pre-wrap; }

.stButton button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    margin-bottom: 16px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-radius: 7px !important;
    padding: 6px 14px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--border) !important;
    color: var(--text) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: #000 !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

.score-ring {
    text-align: center;
    padding: 24px;
    background:var(--surface2);
    border-radius: 12px;
    border: 1px solid var(--border);
}
.score-ring .score-num {
    font-family: 'DM Mono', monospace;
    font-size: 64px;
    font-weight: 500;
    line-height: 1;
}
.score-ring .score-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color:var(--muted);
    margin-top: 8px;
}

.chat-bubble-user {
    background:var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    text-align: right;
}
.chat-bubble-ai {
    background:var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--primary);
    border-radius: 4px 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Streamlit native element text color overrides */
.stMarkdown, .stText, label, .stSelectbox label,
.stToggle label, p, li, span {{
    color: var(--text) !important;
}}
.stCaption, .stCaption p {{
    color: var(--muted) !important;
}}
/* Metric delta */
[data-testid="stMetricDelta"] {{
    color: var(--muted) !important;
}}
/* Dataframe */
[data-testid="stDataFrame"] {{
    background: var(--surface) !important;
}}
/* Input fields */
.stTextInput input, .stChatInput textarea {{
    background: var(--surface2) !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
}}
/* Expander */
.streamlit-expanderHeader {{
    color: var(--text) !important;
    background: var(--surface) !important;
}}
/* Success/warning/info boxes */
.stSuccess, .stWarning, .stInfo {{
    background: var(--surface2) !important;
}}
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="forge-header">
    <h1>InsightForge AI</h1>
    <p>Agentic Auto-EDA  ·  LLM-Powered Insights  ·  Automated Data Profiling</p>
</div>
""", unsafe_allow_html=True)


# ─── API Key (server-side only — never shown in UI for security) ──────────────
def get_api_key():
    # 1. Check Streamlit secrets (for deployed apps — most secure)
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        pass
    # 2. Check environment variable (for local development)
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    # 3. Fallback: show input only if no key found anywhere (local dev only)
    return None

api_key = get_api_key()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Only show key input if not configured server-side
    if not api_key:
        st.markdown('<div class="section-title">API Key Required</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get a free key at console.groq.com. For production, set GROQ_API_KEY in Streamlit secrets instead.",
            placeholder="gsk_..."
        )
        st.caption("For a public site, set the key in Streamlit Secrets — never expose it in the UI.")


    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
    use_llm = st.toggle("Enable LLM Insights", value=True)

    st.markdown("---")
    st.markdown(f"""
<div style="font-family:'DM Mono',monospace;font-size:11px;color:{_theme['muted']};line-height:1.8">
<b style="color:{_theme['primary']}">Pipeline Modules:</b><br>
1. Data Profiler<br>
2. Anomaly Detector<br>
3. Correlation Analyzer<br>
4. Feature Summarizer<br>
5. LLM Insight Generator<br>
6. ML Readiness Scorer<br>
7. Data Cleaner<br>
8. Dataset Chat
</div>
""", unsafe_allow_html=True)

    # Session history
    history = st.session_state.get("dataset_history", [])
    if history:
        st.markdown(f'<div class="section-title">Recent Datasets</div>', unsafe_allow_html=True)
        for i, h in enumerate(history):
            label = f"{h['name']}  ({h['rows']:,} × {h['cols']})"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state["results"] = h["results"]
                st.rerun()

    st.markdown("---")
    st.markdown(f"""
<div style="font-family:'DM Mono',monospace;font-size:10px;color:{_theme['muted']};line-height:1.6">
Built with Streamlit · Plotly · Groq<br>
Powered by LLaMA 3.1
</div>
""", unsafe_allow_html=True)

# Sample + upload state — driven entirely from main screen, no sidebar widget
sample = st.session_state.get("active_sample", "— None —")
uploaded_file = None


# ─── Load data ────────────────────────────────────────────────────────────────
def load_sample(name):
    if name == "Titanic":
        np.random.seed(42)
        n = 891
        return pd.DataFrame({
            "PassengerId": range(1, n+1),
            "Survived": np.random.binomial(1, 0.38, n),
            "Pclass": np.random.choice([1,2,3], n, p=[0.24,0.21,0.55]),
            "Name": [f"Passenger_{i}" for i in range(n)],
            "Sex": np.random.choice(["male","female"], n, p=[0.65,0.35]),
            "Age": np.where(np.random.random(n) < 0.2, np.nan,
                            np.random.normal(29.7, 14.5, n).clip(0.4, 80)),
            "SibSp": np.random.choice([0,1,2,3,4,5,8], n),
            "Parch": np.random.choice([0,1,2,3,4,5,6], n),
            "Fare": np.concatenate([np.random.exponential(30, n-10), [512.3]*10]),
            "Embarked": np.random.choice(["S","C","Q",None], n, p=[0.72,0.19,0.086,0.004]),
        })

    elif name == "Iris":
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df["species"] = [iris.target_names[t] for t in iris.target]
        return df

    elif name == "Boston Housing":
        np.random.seed(0)
        n = 506
        return pd.DataFrame({
            "CRIM": np.random.exponential(3.6, n),
            "ZN": np.where(np.random.random(n) < 0.73, 0, np.random.uniform(0, 100, n)),
            "INDUS": np.random.uniform(0.46, 27.74, n),
            "CHAS": np.random.binomial(1, 0.07, n),
            "NOX": np.random.uniform(0.38, 0.87, n),
            "RM": np.random.normal(6.28, 0.70, n),
            "AGE": np.random.uniform(2.9, 100, n),
            "DIS": np.random.exponential(3.8, n).clip(1.1, 12),
            "RAD": np.random.choice([1,2,3,4,5,6,7,8,24], n),
            "TAX": np.random.choice([187,222,226,233,241,270,277,285,330,391,403,432,666,711], n),
            "PTRATIO": np.random.uniform(12.6, 22, n),
            "B": np.random.uniform(0.32, 396.9, n),
            "LSTAT": np.random.exponential(12, n).clip(1.7, 37.97),
            "MEDV": np.random.normal(22.5, 9.2, n).clip(5, 50),
        })

    elif name == "Synthetic E-Commerce":
        np.random.seed(7)
        n = 1200
        return pd.DataFrame({
            "order_id": range(10000, 10000+n),
            "customer_age": np.where(np.random.random(n) < 0.03, np.nan,
                                     np.random.normal(35, 12, n).clip(18, 80).astype(int)),
            "gender": np.random.choice(["M","F","Other"], n, p=[0.49,0.49,0.02]),
            "product_category": np.random.choice(
                ["Electronics","Clothing","Home","Beauty","Sports","Books"], n,
                p=[0.28,0.22,0.18,0.12,0.12,0.08]),
            "order_value": np.concatenate([np.random.lognormal(4.2, 0.8, n-15), np.random.uniform(500, 2000, 15)]),
            "discount_pct": np.random.choice([0,5,10,15,20,25,30], n, p=[0.35,0.15,0.2,0.1,0.1,0.06,0.04]),
            "items_count": np.random.poisson(2.3, n).clip(1, 15),
            "shipping_days": np.random.choice([1,2,3,4,5,6,7,8,9,10], n),
            "returned": np.random.binomial(1, 0.12, n),
            "customer_rating": np.where(np.random.random(n) < 0.08, np.nan,
                                        np.random.choice([1,2,3,4,5], n, p=[0.05,0.08,0.15,0.35,0.37])),
            "region": np.random.choice(["North","South","East","West","Central"], n),
        })

    elif name == "Credit Card Fraud":
        np.random.seed(11)
        n = 2000
        fraud = np.random.binomial(1, 0.03, n)  # 3% fraud rate
        return pd.DataFrame({
            "transaction_id": range(1, n+1),
            "amount": np.where(fraud, np.random.uniform(500, 5000, n),
                               np.random.exponential(80, n).clip(1, 2000)),
            "hour_of_day": np.random.randint(0, 24, n),
            "day_of_week": np.random.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], n),
            "merchant_category": np.random.choice(
                ["Grocery","Travel","Online","Restaurant","Gas","Entertainment"], n),
            "distance_from_home_km": np.where(fraud,
                np.random.uniform(50, 5000, n), np.random.exponential(15, n).clip(0, 200)),
            "num_transactions_24h": np.where(fraud,
                np.random.randint(5, 30, n), np.random.randint(1, 8, n)),
            "card_present": np.where(fraud,
                np.random.binomial(1, 0.2, n), np.random.binomial(1, 0.85, n)),
            "pin_used": np.random.binomial(1, 0.6, n),
            "age_of_account_days": np.random.randint(30, 3650, n),
            "credit_limit": np.random.choice([1000,2000,5000,10000,15000,20000,50000], n),
            "is_fraud": fraud,
        })

    elif name == "Diabetes":
        np.random.seed(3)
        n = 768
        outcome = np.random.binomial(1, 0.35, n)
        return pd.DataFrame({
            "Pregnancies": np.random.poisson(3.8, n).clip(0, 17),
            "Glucose": np.where(np.random.random(n) < 0.005, 0,
                                np.where(outcome, np.random.normal(141, 31, n),
                                         np.random.normal(110, 26, n)).clip(44, 199)),
            "BloodPressure": np.where(np.random.random(n) < 0.04, 0,
                                      np.random.normal(69, 19, n).clip(24, 122)),
            "SkinThickness": np.where(np.random.random(n) < 0.30, 0,
                                      np.random.normal(29, 15, n).clip(7, 99)),
            "Insulin": np.where(np.random.random(n) < 0.49, 0,
                                np.random.exponential(100, n).clip(14, 846)),
            "BMI": np.where(np.random.random(n) < 0.01, 0,
                            np.random.normal(32, 7.9, n).clip(18, 67)),
            "DiabetesPedigreeFunction": np.random.exponential(0.47, n).clip(0.08, 2.42),
            "Age": np.random.exponential(33, n).clip(21, 81).astype(int),
            "Outcome": outcome,
        })

    elif name == "Wine Quality":
        np.random.seed(5)
        n = 1599
        quality = np.random.choice([3,4,5,6,7,8], n, p=[0.006,0.033,0.425,0.399,0.124,0.013])
        return pd.DataFrame({
            "fixed_acidity": np.random.normal(8.3, 1.7, n).clip(4.6, 15.9),
            "volatile_acidity": np.random.normal(0.53, 0.18, n).clip(0.12, 1.58),
            "citric_acid": np.random.exponential(0.27, n).clip(0, 1),
            "residual_sugar": np.random.exponential(2.5, n).clip(1.2, 15.5),
            "chlorides": np.random.normal(0.087, 0.047, n).clip(0.012, 0.611),
            "free_sulfur_dioxide": np.random.exponential(15, n).clip(1, 72),
            "total_sulfur_dioxide": np.random.normal(46, 33, n).clip(6, 289),
            "density": np.random.normal(0.9967, 0.0019, n).clip(0.990, 1.004),
            "pH": np.random.normal(3.31, 0.15, n).clip(2.74, 4.01),
            "sulphates": np.random.normal(0.66, 0.17, n).clip(0.33, 2.0),
            "alcohol": np.random.normal(10.4, 1.07, n).clip(8.4, 14.9),
            "quality": quality,
        })

    elif name == "Heart Disease":
        np.random.seed(9)
        n = 1025
        target = np.random.binomial(1, 0.51, n)
        return pd.DataFrame({
            "age": np.random.normal(54, 9, n).clip(29, 77).astype(int),
            "sex": np.random.binomial(1, 0.68, n),
            "chest_pain_type": np.random.choice([0,1,2,3], n, p=[0.47,0.17,0.28,0.08]),
            "resting_bp": np.where(np.random.random(n)<0.01, np.nan,
                                   np.random.normal(131, 17, n).clip(94, 200)),
            "cholesterol": np.where(np.random.random(n)<0.02, np.nan,
                                    np.random.normal(246, 51, n).clip(126, 564)),
            "fasting_blood_sugar": np.random.binomial(1, 0.15, n),
            "resting_ecg": np.random.choice([0,1,2], n, p=[0.49,0.02,0.49]),
            "max_heart_rate": np.random.normal(149, 23, n).clip(71, 202).astype(int),
            "exercise_angina": np.random.binomial(1, 0.33, n),
            "st_depression": np.random.exponential(1.04, n).clip(0, 6.2).round(1),
            "st_slope": np.random.choice([0,1,2], n, p=[0.07,0.46,0.47]),
            "num_vessels": np.random.choice([0,1,2,3], n, p=[0.58,0.22,0.13,0.07]),
            "thalassemia": np.random.choice([0,1,2,3], n, p=[0.01,0.06,0.55,0.38]),
            "target": target,
        })

    elif name == "Car Price Prediction":
        np.random.seed(13)
        n = 1500
        brands = ["Toyota","Honda","BMW","Mercedes","Ford","Hyundai","Audi","Kia","Nissan","Volkswagen"]
        brand = np.random.choice(brands, n, p=[0.18,0.15,0.1,0.1,0.12,0.1,0.08,0.07,0.06,0.04])
        year = np.random.randint(2005, 2024, n)
        mileage = np.random.exponential(45000, n).clip(500, 250000).astype(int)
        engine_cc = np.random.choice([1000,1200,1400,1500,1600,1800,2000,2500,3000,3500,4000], n)
        premium = np.isin(brand, ["BMW","Mercedes","Audi"]).astype(float)
        price = (
            15000
            + premium * 20000
            + (year - 2005) * 800
            - mileage * 0.05
            + engine_cc * 2
            + np.random.normal(0, 3000, n)
        ).clip(1000, 120000).astype(int)
        return pd.DataFrame({
            "brand": brand,
            "year": year,
            "mileage_km": mileage,
            "engine_cc": engine_cc,
            "fuel_type": np.random.choice(["Petrol","Diesel","Hybrid","Electric"], n, p=[0.55,0.25,0.15,0.05]),
            "transmission": np.random.choice(["Manual","Automatic"], n, p=[0.45,0.55]),
            "doors": np.random.choice([2,4,5], n, p=[0.1,0.7,0.2]),
            "color": np.random.choice(["White","Black","Silver","Blue","Red","Grey","Other"], n,
                                       p=[0.25,0.2,0.18,0.12,0.1,0.1,0.05]),
            "num_owners": np.random.choice([1,2,3,4], n, p=[0.5,0.3,0.15,0.05]),
            "accident_history": np.random.binomial(1, 0.18, n),
            "price_usd": price,
        })

    elif name == "Stock Market":
        np.random.seed(17)
        n = 1000
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        close = np.cumsum(np.random.normal(0.05, 1.5, n)) + 150
        close = close.clip(50, 500)
        high = close + np.random.uniform(0, 3, n)
        low = close - np.random.uniform(0, 3, n)
        open_ = low + np.random.uniform(0, high - low)
        return pd.DataFrame({
            "date": dates,
            "open": open_.round(2),
            "high": high.round(2),
            "low": low.round(2),
            "close": close.round(2),
            "volume": np.random.lognormal(15, 0.8, n).astype(int),
            "ma_7": pd.Series(close).rolling(7).mean().values.round(2),
            "ma_30": pd.Series(close).rolling(30).mean().values.round(2),
            "daily_return_pct": pd.Series(close).pct_change().mul(100).round(3).values,
            "volatility": pd.Series(close).rolling(10).std().round(3).values,
            "ticker": np.random.choice(["AAPL","GOOGL","MSFT","AMZN","TSLA"], n),
        })

    return None


def load_df():
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    active = st.session_state.get("active_sample", "— None —")
    if active and active != "— None —":
        return load_sample(active)
    return None


df = load_df()

# ─── Main content ─────────────────────────────────────────────────────────────
if df is None:
    st.markdown("""
<div style="text-align:center;padding:40px 20px 20px 20px;color:var(--muted)">
    <div style="font-size:18px;font-weight:600;margin-bottom:8px;color:var(--text)" style="color:var(--text)">Upload a dataset to begin</div>

</div>
""", unsafe_allow_html=True)

    # Centre-column upload widget on main screen
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        main_upload = st.file_uploader(
            "Drop your CSV or Excel file here",
            type=["csv", "xlsx", "xls"],
            key="main_uploader",

        )
        if main_upload is not None:
            uploaded_file = main_upload
            if main_upload.name.endswith(".csv"):
                df = pd.read_csv(main_upload)
            else:
                df = pd.read_excel(main_upload)
            st.rerun()

        st.markdown('<div style="text-align:center;color:var(--muted);font-size:12px;font-family:DM Mono,monospace;margin:16px 0">— or try a sample dataset —</div>', unsafe_allow_html=True)

        sample_options = [
            "Titanic", "Iris",
            "Boston Housing", "Synthetic E-Commerce",
            "Credit Card Fraud", "Diabetes",
            "Wine Quality", "Heart Disease",
            "Car Price Prediction", "Stock Market",
        ]
        cols_s = st.columns(2)
        for i, s in enumerate(sample_options):
            with cols_s[i % 2]:
                if st.button(s, key=f"main_sample_{i}", use_container_width=True):
                    st.session_state["active_sample"] = s
                    st.rerun()

else:
    num_numeric = len(df.select_dtypes(include=[np.number]).columns)
    num_cat = len(df.select_dtypes(exclude=[np.number]).columns)
    missing_pct = round(df.isna().sum().sum() / df.size * 100, 1)

    st.markdown(f"""
<div class="metric-grid">
    <div class="metric-box"><span class="value">{df.shape[0]:,}</span><span class="label">Rows</span></div>
    <div class="metric-box"><span class="value">{df.shape[1]}</span><span class="label">Columns</span></div>
    <div class="metric-box"><span class="value">{num_numeric}</span><span class="label">Numeric</span></div>
    <div class="metric-box"><span class="value">{num_cat}</span><span class="label">Categorical</span></div>
    <div class="metric-box"><span class="value" style="color:{'#FF6B6B' if missing_pct > 5 else '#00E5A0'}">{missing_pct}%</span><span class="label">Missing</span></div>
    <div class="metric-box"><span class="value">{df.duplicated().sum()}</span><span class="label">Duplicates</span></div>
</div>
""", unsafe_allow_html=True)

    run_col, change_col, _ = st.columns([2, 2, 3])
    with run_col:
        run_analysis = st.button("Run Analysis", use_container_width=True)
    with change_col:
        if st.button("Change Dataset", use_container_width=True):
            st.session_state.pop("active_sample", None)
            st.session_state.pop("results", None)
            st.session_state.pop("df", None)
            st.session_state.pop("chat_history", None)
            st.session_state.pop("ml_recommendations", None)
            st.session_state.pop("feat_suggestions", None)
            st.session_state.pop("pdf_report", None)
            st.rerun()

    if run_analysis or "results" in st.session_state:
        if run_analysis:
            from eda_engine import InsightForgePipeline
            progress_box = st.empty()
            status_msgs = []

            def progress_cb(msg):
                status_msgs.append(msg)
                progress_box.markdown(
                    "".join([f'<div style="font-family:DM Mono,monospace;font-size:12px;color:var(--muted)">{m}</div>'
                              for m in status_msgs[-5:]]),
                    unsafe_allow_html=True)

            with st.spinner(""):
                pipeline = InsightForgePipeline(df, api_key=api_key or None)
                results = pipeline.run(use_llm=use_llm, progress_callback=progress_cb)
                st.session_state["results"] = results
                st.session_state["df"] = df
                st.session_state["chat_history"] = []
                st.session_state["ml_recommendations"] = None
                st.session_state["feat_suggestions"] = None

                # Session history — remember last 3 datasets
                history = st.session_state.get("dataset_history", [])
                dataset_name = st.session_state.get("active_sample", "Uploaded file")
                entry = {
                    "name": dataset_name,
                    "rows": df.shape[0],
                    "cols": df.shape[1],
                    "results": results,
                    "df_cols": df.columns.tolist(),
                }
                history = [h for h in history if h["name"] != dataset_name]
                history.insert(0, entry)
                st.session_state["dataset_history"] = history[:3]

            progress_box.empty()

        results = st.session_state.get("results", {})
        df = st.session_state.get("df", df)

        if results:
            from visualizations import (missing_heatmap, distribution_plots,
                                         correlation_heatmap, outlier_boxplots,
                                         categorical_bar_charts, feature_overview_sunburst,
                                         correlation_network, before_after_distributions,
                                         column_distribution_detail, health_scorecard_chart)

            profile = results.get("profile", {})
            anomalies = results.get("anomalies", {})
            correlations = results.get("correlations", {})
            feature_summary = results.get("feature_summary", {})
            insights = results.get("insights", {})

            # ── Summary banner ────────────────────────────────────────────────
            _total_missing = sum(v["count"] for v in profile.get("missing", {}).values())
            _outlier_cols = len([c for c, v in anomalies.items() if v["iqr_outlier_count"] > 0])
            _strong_corr = len(correlations.get("strong_pairs", []))
            _num_feat = len(feature_summary.get("numeric_features", []))
            _cat_feat = len(feature_summary.get("categorical_features", []))

            _missing_color  = "#FF6B6B" if _total_missing > 0 else "#00E5A0"
            _outlier_color  = "#FFB800" if _outlier_cols > 0 else "#00E5A0"
            _corr_snippet   = f'&nbsp;·&nbsp;<b style="color:#7B61FF">{_strong_corr} strong correlations</b>' if _strong_corr > 0 else ""

            st.markdown(f"""
<div style="background:#111827;border:1px solid #1F2937;border-left:3px solid #00D4FF;
            border-radius:10px;padding:16px 24px;margin:16px 0 8px 0;">
  <div style="font-family:DM Mono,monospace;font-size:11px;color:#64748B;
              letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">
    Analysis complete
  </div>
  <div style="font-size:13px;color:#E2E8F0;line-height:1.6">
    Found <b style="color:#00D4FF">{_num_feat} numeric</b> and
    <b style="color:#7B61FF">{_cat_feat} categorical</b> features &nbsp;·&nbsp;
    <b style="color:{_missing_color}">{_total_missing} missing values</b> &nbsp;·&nbsp;
    <b style="color:{_outlier_color}">{_outlier_cols} columns with outliers</b>{_corr_snippet}
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Tabs ──────────────────────────────────────────────────────────
            tabs = st.tabs([
                "Overview", "Distributions", "Correlations",
                "Anomalies", "LLM Insights", "Data Cleaning",
                "ML Readiness", "Health Scorecard", "Column Explorer",
                "Feature Engineering", "Chat", "Raw Data"
            ])
            tab_overview, tab_dist, tab_corr, tab_anomaly, tab_insights, \
                tab_clean, tab_ml, tab_health, tab_col, tab_feat, tab_chat, tab_raw = tabs



            # ── TAB: Overview ─────────────────────────────────────────────────
            with tab_overview:
                col1, col2 = st.columns([1, 1])
                with col1:
                    fig = feature_overview_sunburst(feature_summary)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = missing_heatmap(profile)
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown('<div class="section-title">Feature Classification</div>', unsafe_allow_html=True)
                tag_html = ""
                for col in feature_summary.get("numeric_features", []):
                    tag_html += f'<span class="tag">{col}</span>'
                for col in feature_summary.get("categorical_features", []):
                    tag_html += f'<span class="tag cat">{col}</span>'
                for col in feature_summary.get("binary_features", []):
                    tag_html += f'<span class="tag binary">{col}</span>'
                for col in feature_summary.get("high_cardinality_features", []):
                    tag_html += f'<span class="tag high">{col}</span>'
                for col in feature_summary.get("potential_id_columns", []):
                    tag_html += f'<span class="tag id">{col}</span>'
                legend = ('<span style="font-size:11px;color:var(--muted);font-family:DM Mono,monospace">'
                         '■ <span style="color:#00D4FF">Numeric</span> &nbsp;'
                         '■ <span style="color:#7B61FF">Categorical</span> &nbsp;'
                         '■ <span style="color:#00E5A0">Binary</span> &nbsp;'
                         '■ <span style="color:#FFB800">High Cardinality</span> &nbsp;'
                         '■ <span style="color:#FF6B6B">Potential ID</span></span>')
                st.markdown(tag_html + "<br>" + legend, unsafe_allow_html=True)

            # ── TAB: Distributions ────────────────────────────────────────────
            with tab_dist:
                fig = distribution_plots(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                fig = categorical_bar_charts(df, profile)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            # ── TAB: Correlations ─────────────────────────────────────────────
            with tab_corr:
                fig = correlation_heatmap(correlations)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                net_fig = correlation_network(correlations)
                if net_fig:
                    st.markdown('<div class="section-title">Correlation Network</div>', unsafe_allow_html=True)
                    st.plotly_chart(net_fig, use_container_width=True)
                strong_pairs = correlations.get("strong_pairs", [])
                if strong_pairs:
                    st.markdown('<div class="section-title">Strong Correlations (|r| > 0.5)</div>', unsafe_allow_html=True)
                    pairs_df = pd.DataFrame(strong_pairs)
                    pairs_df["correlation"] = pairs_df["correlation"].apply(lambda x: f"{x:.4f}")
                    st.dataframe(pairs_df, use_container_width=True, hide_index=True)

            # ── TAB: Anomalies ────────────────────────────────────────────────
            with tab_anomaly:
                fig = outlier_boxplots(df, anomalies)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                if anomalies:
                    st.markdown('<div class="section-title">Outlier Summary</div>', unsafe_allow_html=True)
                    anomaly_rows = []
                    for col, v in anomalies.items():
                        anomaly_rows.append({
                            "Column": col,
                            "IQR Outliers": v["iqr_outlier_count"],
                            "IQR %": f"{v['iqr_outlier_pct']:.1f}%",
                            "Z-score Outliers": v["zscore_outlier_count"],
                            "IQR Lower": v["iqr_bounds"]["lower"],
                            "IQR Upper": v["iqr_bounds"]["upper"],
                        })
                    st.dataframe(pd.DataFrame(anomaly_rows), use_container_width=True, hide_index=True)

            # ── TAB: LLM Insights ─────────────────────────────────────────────
            with tab_insights:
                if not use_llm:
                    st.info("Enable LLM Insights in the sidebar to see AI-generated analysis.")
                else:
                    st.markdown(f"""
<div class="insight-card">
    <h4>Dataset Overview</h4>
    <div class="llm-insight">{insights.get("overview", "")}</div>
</div>
<div class="insight-card anomaly">
    <h4>Anomaly Analysis</h4>
    <div class="llm-insight">{insights.get("anomalies", "")}</div>
</div>
<div class="insight-card correlation">
    <h4>Correlation Analysis</h4>
    <div class="llm-insight">{insights.get("correlations", "")}</div>
</div>
""", unsafe_allow_html=True)

            # ── TAB: Data Cleaning ────────────────────────────────────────────
            with tab_clean:
                st.markdown('<div class="section-title">Auto-Fix Options</div>', unsafe_allow_html=True)

                clean_df = df.copy()
                changes = []

                col1, col2 = st.columns(2)
                with col1:
                    fix_missing = st.checkbox("Fill missing values", value=True,
                        help="Numeric → median, Categorical → mode")
                    fix_duplicates = st.checkbox("Remove duplicate rows", value=True)
                with col2:
                    fix_outliers = st.checkbox("Cap outliers (IQR method)", value=False,
                        help="Clips values beyond 1.5×IQR to the boundary")
                    fix_dtypes = st.checkbox("Fix numeric columns stored as strings", value=True)

                if st.button("Apply Cleaning", use_container_width=False):
                    if fix_duplicates:
                        before = len(clean_df)
                        clean_df = clean_df.drop_duplicates()
                        removed = before - len(clean_df)
                        if removed > 0:
                            changes.append(f"Removed {removed} duplicate rows")

                    if fix_dtypes:
                        for col in clean_df.columns:
                            if clean_df[col].dtype == object:
                                try:
                                    clean_df[col] = pd.to_numeric(clean_df[col])
                                    changes.append(f"Converted '{col}' to numeric")
                                except:
                                    pass

                    if fix_missing:
                        for col in clean_df.columns:
                            missing = clean_df[col].isna().sum()
                            if missing > 0:
                                if pd.api.types.is_numeric_dtype(clean_df[col]):
                                    fill_val = clean_df[col].median()
                                    clean_df[col] = clean_df[col].fillna(fill_val)
                                    changes.append(f"Filled {missing} missing in '{col}' with median ({fill_val:.2f})")
                                else:
                                    fill_val = clean_df[col].mode()[0] if len(clean_df[col].mode()) > 0 else "Unknown"
                                    clean_df[col] = clean_df[col].fillna(fill_val)
                                    changes.append(f"Filled {missing} missing in '{col}' with mode ('{fill_val}')")

                    if fix_outliers:
                        for col in clean_df.select_dtypes(include=[np.number]).columns:
                            Q1, Q3 = clean_df[col].quantile(0.25), clean_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                            outlier_count = ((clean_df[col] < lower) | (clean_df[col] > upper)).sum()
                            if outlier_count > 0:
                                clean_df[col] = clean_df[col].clip(lower, upper)
                                changes.append(f"Capped {outlier_count} outliers in '{col}'")

                    st.session_state["clean_df"] = clean_df
                    st.session_state["clean_changes"] = changes

                if "clean_changes" in st.session_state:
                    for msg in st.session_state["clean_changes"]:
                        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:12px;color:#00E5A0;padding:2px 0">{msg}</div>',
                                    unsafe_allow_html=True)
                    if not st.session_state["clean_changes"]:
                        st.info("No changes needed — dataset is already clean!")

                if "clean_df" in st.session_state:
                    cdf = st.session_state["clean_df"]
                    st.markdown('<div class="section-title">Cleaned Dataset Preview</div>', unsafe_allow_html=True)
                    st.dataframe(cdf.head(20), use_container_width=True)

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Rows", f"{len(cdf):,}", delta=f"{len(cdf)-len(df):,}")
                    with col_b:
                        st.metric("Missing Values", f"{cdf.isna().sum().sum()}", delta=f"{cdf.isna().sum().sum()-df.isna().sum().sum()}")
                    with col_c:
                        st.metric("Duplicates", f"{cdf.duplicated().sum()}", delta=f"{cdf.duplicated().sum()-df.duplicated().sum()}")

                    csv_data = cdf.to_csv(index=False)
                    st.download_button(
                        "Download Cleaned CSV",
                        data=csv_data,
                        file_name="insightforge_cleaned.csv",
                        mime="text/csv",
                        use_container_width=False,
                    )

            # ── TAB: ML Readiness ─────────────────────────────────────────────
            with tab_ml:
                st.markdown('<div class="section-title">ML Readiness Score</div>', unsafe_allow_html=True)

                # Compute score programmatically
                score = 100
                penalties = []
                bonuses = []

                missing_pct_val = df.isna().sum().sum() / df.size * 100
                if missing_pct_val > 30:
                    score -= 25
                    penalties.append(f"Very high missing data ({missing_pct_val:.1f}%) — major issue")
                elif missing_pct_val > 10:
                    score -= 15
                    penalties.append(f"High missing data ({missing_pct_val:.1f}%) — needs imputation")
                elif missing_pct_val > 0:
                    score -= 5
                    penalties.append(f"Some missing values ({missing_pct_val:.1f}%) — minor, easy to fix")

                dup_pct = df.duplicated().sum() / len(df) * 100
                if dup_pct > 5:
                    score -= 10
                    penalties.append(f"{dup_pct:.1f}% duplicate rows detected")

                if len(df) < 100:
                    score -= 20
                    penalties.append(f"Very small dataset ({len(df)} rows) — ML may not generalize")
                elif len(df) < 500:
                    score -= 10
                    penalties.append(f"Small dataset ({len(df)} rows) — limited for complex models")
                else:
                    bonuses.append(f"Good dataset size ({len(df):,} rows)")

                high_outlier_cols = [c for c, v in anomalies.items() if v["iqr_outlier_pct"] > 10]
                if len(high_outlier_cols) > 3:
                    score -= 10
                    penalties.append(f"Many columns with heavy outliers: {high_outlier_cols}")
                elif high_outlier_cols:
                    score -= 5
                    penalties.append(f"Some outliers in: {high_outlier_cols}")

                high_cardinality = feature_summary.get("high_cardinality_features", [])
                if len(high_cardinality) > 3:
                    score -= 10
                    penalties.append(f"High cardinality features need encoding: {high_cardinality}")

                strong_pairs = correlations.get("strong_pairs", [])
                very_strong = [p for p in strong_pairs if abs(p["correlation"]) > 0.9]
                if very_strong:
                    score -= 5
                    penalties.append(f"Near-perfect correlations (multicollinearity risk): {[p['feature_a']+'×'+p['feature_b'] for p in very_strong]}")

                num_features = len(feature_summary.get("numeric_features", []))
                if num_features >= 3:
                    bonuses.append(f"{num_features} numeric features available for modeling")

                score = max(0, min(100, score))

                if score >= 80:
                    score_color = "#00E5A0"
                    score_label = "ML Ready"
                elif score >= 60:
                    score_color = "#FFB800"
                    score_label = "Needs Work"
                else:
                    score_color = "#FF6B6B"
                    score_label = "Not Ready"

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"""
<div class="score-ring">
    <div class="score-num" style="color:{score_color}">{score}</div>
    <div class="score-label" style="color:{score_color}">{score_label}</div>
    <div style="font-size:11px;color:var(--muted);margin-top:4px;font-family:DM Mono,monospace">out of 100</div>
</div>
""", unsafe_allow_html=True)

                with col2:
                    if penalties:
                        st.markdown("**Issues Found:**")
                        for p in penalties:
                            st.markdown(f'<div style="font-size:13px;padding:3px 0;font-family:DM Mono,monospace;color:var(--text)">{p}</div>',
                                        unsafe_allow_html=True)
                    if bonuses:
                        st.markdown("**Strengths:**")
                        for b in bonuses:
                            st.markdown(f'<div style="font-size:13px;padding:3px 0;font-family:DM Mono,monospace;color:var(--text)">{b}</div>',
                                        unsafe_allow_html=True)

                # LLM ML Readiness commentary
                if use_llm and api_key:
                    if st.button("Get ML Recommendations"):
                        from eda_engine import LLMInsightGenerator
                        with st.spinner("Generating ML recommendations..."):
                            llm = LLMInsightGenerator(api_key=api_key)
                            prompt_text = f"""You are a senior ML engineer. Given this dataset analysis, provide 5 specific, actionable recommendations to prepare this dataset for machine learning.

ML Readiness Score: {score}/100
Issues: {penalties}
Strengths: {bonuses}
Shape: {df.shape[0]} rows x {df.shape[1]} columns
Features: numeric={feature_summary['numeric_features']}, categorical={feature_summary['categorical_features']}
Missing: {missing_pct_val:.1f}%
Strong correlations: {strong_pairs[:3]}

Give concrete steps like "Use SMOTE for class imbalance", "Apply log transform to X column", etc.
Use • for bullets. Be specific."""
                            st.session_state["ml_recommendations"] = llm._call_groq(prompt_text)

                    if st.session_state.get("ml_recommendations"):
                        st.markdown(f"""
<div class="insight-card ml">
    <h4>ML Preparation Recommendations</h4>
    <div class="llm-insight">{st.session_state["ml_recommendations"]}</div>
</div>
""", unsafe_allow_html=True)


            # ── TAB: Health Scorecard ─────────────────────────────────────────
            with tab_health:
                from eda_engine import compute_health_scores
                health_scores = compute_health_scores(profile, anomalies, feature_summary, correlations)
                overall = round(sum(v["score"] for v in health_scores.values()) / len(health_scores))
                overall_color = "#00E5A0" if overall >= 80 else "#FFB800" if overall >= 60 else "#FF6B6B"
                overall_grade = "A" if overall >= 90 else "B" if overall >= 75 else "C" if overall >= 60 else "D"

                st.markdown(f"""
<div style="display:flex;gap:16px;margin-bottom:20px;align-items:stretch">
  <div style="background:#111827;border:1px solid #1F2937;border-radius:12px;
              padding:24px 32px;text-align:center;min-width:140px">
    <div style="font-size:56px;font-weight:700;color:{overall_color};
                font-family:DM Mono,monospace;line-height:1">{overall}</div>
    <div style="font-size:28px;color:{overall_color};font-family:DM Mono,monospace">{overall_grade}</div>
    <div style="font-size:11px;color:#64748B;text-transform:uppercase;
                letter-spacing:2px;margin-top:6px">Overall Health</div>
  </div>
  <div style="flex:1;background:#111827;border:1px solid #1F2937;border-radius:12px;padding:20px 24px">
    {"".join([
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0;border-bottom:1px solid #1F2937">'
        f'<span style="font-size:13px;color:#E2E8F0">{dim}</span>'
        f'<span style="font-family:DM Mono,monospace;font-size:12px;'
        f'color:{"#00E5A0" if v["score"]>=80 else "#FFB800" if v["score"]>=60 else "#FF6B6B"}">'
        f'{v["score"]}/100 &nbsp; {v["note"]}</span></div>'
        for dim, v in health_scores.items()
    ])}
  </div>
</div>
""", unsafe_allow_html=True)

                fig = health_scorecard_chart(health_scores)
                st.plotly_chart(fig, use_container_width=True)

            # ── TAB: Column Explorer ──────────────────────────────────────────
            with tab_col:
                st.markdown('<div class="section-title">Select a column to explore</div>', unsafe_allow_html=True)
                all_cols = df.columns.tolist()
                selected_col = st.selectbox("Column", all_cols, key="col_explorer_select",
                                             label_visibility="collapsed")

                if selected_col:
                    col_stats = profile["columns"].get(selected_col, {})
                    missing_n = col_stats.get("missing_count", 0)
                    missing_p = col_stats.get("missing_pct", 0)
                    unique_n  = col_stats.get("unique_count", 0)

                    s1, s2, s3, s4 = st.columns(4)
                    with s1:
                        st.metric("Type", col_stats.get("dtype", "—"))
                    with s2:
                        st.metric("Unique Values", f"{unique_n:,}")
                    with s3:
                        st.metric("Missing", f"{missing_n:,} ({missing_p}%)")
                    with s4:
                        if pd.api.types.is_numeric_dtype(df[selected_col]):
                            st.metric("Mean", f"{col_stats.get('mean', '—')}")
                        else:
                            top = col_stats.get("top_values", {})
                            top_val = list(top.keys())[0] if top else "—"
                            st.metric("Most Common", str(top_val)[:20])

                    fig = column_distribution_detail(df, selected_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # LLM column commentary
                    if use_llm and api_key:
                        if st.button("Get AI commentary on this column"):
                            from eda_engine import LLMInsightGenerator
                            llm = LLMInsightGenerator(api_key=api_key)
                            col_prompt = f"""You are a data scientist. Give a brief 3-4 bullet analysis of this column.

Column: {selected_col}
Type: {col_stats.get("dtype")}
Missing: {missing_p}%
Unique values: {unique_n}
Stats: {json.dumps({k: v for k, v in col_stats.items() if k not in ["top_values", "dtype"]})}

Focus on: data quality, distribution shape, modeling considerations.
Use • for bullets. Be specific and concise."""
                            with st.spinner("Analysing column..."):
                                col_insight = llm._call_groq(col_prompt)
                            st.session_state[f"col_insight_{selected_col}"] = col_insight

                    if st.session_state.get(f"col_insight_{selected_col}"):
                        st.markdown(f"""
<div class="insight-card">
    <h4>Column Analysis — {selected_col}</h4>
    <div class="llm-insight">{st.session_state[f"col_insight_{selected_col}"]}</div>
</div>
""", unsafe_allow_html=True)

            # ── TAB: Feature Engineering ──────────────────────────────────────
            with tab_feat:
                st.markdown('<div class="section-title">Target Column</div>', unsafe_allow_html=True)
                target_options = ["— None —"] + df.columns.tolist()
                target_col = st.selectbox("Select target column (optional)", target_options,
                                           key="feat_target", label_visibility="collapsed")

                if target_col != "— None —":
                    target_series = df[target_col]
                    n_unique = target_series.nunique()
                    is_classification = n_unique <= 20 or not pd.api.types.is_numeric_dtype(target_series)
                    task_type = "Classification" if is_classification else "Regression"

                    tc1, tc2, tc3 = st.columns(3)
                    with tc1:
                        st.metric("Task Type", task_type)
                    with tc2:
                        st.metric("Unique Classes", n_unique)
                    with tc3:
                        balance = target_series.value_counts(normalize=True).max() * 100
                        st.metric("Majority Class", f"{balance:.1f}%")

                    if is_classification and n_unique <= 10:
                        import plotly.express as px
                        vc = target_series.value_counts()
                        bar_fig = px.bar(x=vc.index.astype(str), y=vc.values,
                                         labels={"x": target_col, "y": "Count"})
                        bar_fig.update_traces(marker_color="#7B61FF")
                        bar_fig.update_layout(
                            paper_bgcolor="#111827", plot_bgcolor="#111827",
                            font=dict(color="#E2E8F0"), height=280,
                            title="Class Distribution"
                        )
                        st.plotly_chart(bar_fig, use_container_width=True)

                    if balance > 80:
                        st.warning(f"Class imbalance detected — majority class is {balance:.1f}%. Consider SMOTE or class weights.")

                st.markdown('<div class="section-title">AI Feature Engineering Suggestions</div>', unsafe_allow_html=True)
                if use_llm and api_key:
                    if st.button("Generate feature engineering suggestions"):
                        from eda_engine import LLMInsightGenerator
                        llm = LLMInsightGenerator(api_key=api_key)
                        feat_prompt = f"""You are a senior ML engineer. Suggest specific feature engineering steps for this dataset.

Dataset: {profile["shape"]["rows"]} rows x {profile["shape"]["columns"]} columns
Numeric features: {feature_summary["numeric_features"]}
Categorical features: {feature_summary["categorical_features"]}
High cardinality: {feature_summary["high_cardinality_features"]}
Target column: {target_col if target_col != "— None —" else "not specified"}
Strong correlations: {correlations.get("strong_pairs", [])[:5]}
Skewed columns: {[c for c, v in profile["columns"].items() if abs(v.get("skewness", 0) or 0) > 1]}

For each suggestion, provide:
1. The feature engineering step
2. Which column(s) it applies to
3. A ready-to-run Python code snippet using pandas

Format each as:
SUGGESTION: <name>
APPLIES TO: <columns>
REASON: <one line>
CODE:
```python
<code>
```

Give 4-6 suggestions. Be specific to this dataset, not generic."""
                        with st.spinner("Generating suggestions..."):
                            feat_response = llm._call_groq(feat_prompt)
                        st.session_state["feat_suggestions"] = feat_response
                else:
                    st.info("Enable LLM Insights and add API key to use this feature.")

                if st.session_state.get("feat_suggestions"):
                    # Parse and render each suggestion nicely
                    raw = st.session_state["feat_suggestions"]
                    blocks = raw.split("SUGGESTION:")
                    for block in blocks[1:]:
                        lines = block.strip().splitlines()
                        title = lines[0].strip() if lines else "Suggestion"
                        body  = "<br>".join(lines[1:]).strip()
                        st.markdown(f"""
<div class="insight-card" style="margin-bottom:12px">
    <h4>{title}</h4>
    <div class="llm-insight" style="font-size:13px">{body}</div>
</div>
""", unsafe_allow_html=True)

            # ── TAB: Chat ─────────────────────────────────────────────────────
            with tab_chat:
                st.markdown('<div class="section-title">Chat With Your Dataset</div>', unsafe_allow_html=True)
                st.caption("Ask anything about your data — patterns, cleaning advice, modeling suggestions, column explanations.")

                if "chat_history" not in st.session_state:
                    st.session_state["chat_history"] = []

                # Process any pending question FIRST before rendering
                question = st.session_state.pop("pending_question", None)

                if question and api_key:
                    st.session_state["chat_history"].append({"role": "user", "content": question})
                    from eda_engine import LLMInsightGenerator
                    llm = LLMInsightGenerator(api_key=api_key)
                    context = f"""You are a data science assistant. Answer questions about this dataset concisely and helpfully.

DATASET CONTEXT:
- Shape: {profile['shape']['rows']} rows x {profile['shape']['columns']} columns
- Numeric features: {feature_summary['numeric_features']}
- Categorical features: {feature_summary['categorical_features']}
- Binary features: {feature_summary['binary_features']}
- Missing data: {json.dumps({k: v for k, v in profile['missing'].items() if v['count'] > 0})}
- Outlier columns: {[c for c, v in anomalies.items() if v['iqr_outlier_count'] > 0]}
- Strong correlations: {correlations.get('strong_pairs', [])[:5]}
- Duplicates: {profile['duplicates']}

CONVERSATION HISTORY:
{chr(10).join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state['chat_history'][-6:]])}

USER QUESTION: {question}
Answer clearly and concisely. Use bullet points if listing multiple items."""
                    with st.spinner("Thinking..."):
                        response = llm._call_groq(context)
                    st.session_state["chat_history"].append({"role": "assistant", "content": response})

                # Render full chat history
                for msg in st.session_state["chat_history"]:
                    if msg["role"] == "user":
                        st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="chat-bubble-ai">{msg["content"]}</div>', unsafe_allow_html=True)

                # Suggested questions — always show, not just when history is empty
                st.markdown("---")
                st.markdown('<div style="font-size:12px;color:var(--muted);font-family:DM Mono,monospace;margin-bottom:8px">Suggested Questions</div>', unsafe_allow_html=True)
                suggestions = [
                    "Which column has the most outliers?",
                    "Is this dataset suitable for classification?",
                    "What's the best way to handle missing values here?",
                    "Which features are most likely useful for prediction?",
                ]
                s_cols = st.columns(2)
                for i, suggestion in enumerate(suggestions):
                    with s_cols[i % 2]:
                        if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                            st.session_state["pending_question"] = suggestion
                            st.rerun()

                # Chat input at the bottom
                user_input = st.chat_input("Ask a question about your dataset...")
                if user_input:
                    if not api_key:
                        st.warning("Please provide a Groq API key to use chat.")
                    else:
                        st.session_state["pending_question"] = user_input
                        st.rerun()

                if st.session_state.get("chat_history"):
                    if st.button("Clear Chat"):
                        st.session_state["chat_history"] = []
                        st.rerun()

            # ── TAB: Raw Data ─────────────────────────────────────────────────
            with tab_raw:
                st.markdown(f'<div class="section-title">Preview · {df.shape[0]:,} rows × {df.shape[1]} columns</div>',
                            unsafe_allow_html=True)
                st.dataframe(df.head(100), use_container_width=True)
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Statistical Summary**")
                    st.dataframe(df.describe(), use_container_width=True)
                with col_b:
                    st.markdown("**Data Types**")
                    dtype_df = pd.DataFrame({
                        "Column": df.dtypes.index,
                        "Type": df.dtypes.values.astype(str),
                        "Non-Null": df.count().values,
                        "Null %": (df.isna().mean() * 100).round(1).values
                    })
                    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

                results_json = json.dumps(
                    {k: v for k, v in results.items() if k != "insights"},
                    indent=2, default=str
                )
                st.download_button(
                    "Download EDA Results (JSON)",
                    data=results_json,
                    file_name="insightforge_results.json",
                    mime="application/json"
                )

                st.markdown('<div class="section-title">PDF Report</div>', unsafe_allow_html=True)
                if st.button("Generate PDF Report"):
                    with st.spinner("Building report..."):
                        try:
                            from reportlab.lib.pagesizes import letter
                            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                            from reportlab.lib.units import inch
                            from reportlab.lib import colors
                            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                                             Table, TableStyle, HRFlowable)
                            import io as _io

                            buf = _io.BytesIO()
                            doc = SimpleDocTemplate(buf, pagesize=letter,
                                                     leftMargin=0.75*inch, rightMargin=0.75*inch,
                                                     topMargin=0.75*inch, bottomMargin=0.75*inch)
                            styles = getSampleStyleSheet()
                            elems = []

                            title_style = ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold",
                                                          spaceAfter=4, textColor=colors.HexColor("#00D4FF"))
                            h2_style = ParagraphStyle("h2", fontSize=13, fontName="Helvetica-Bold",
                                                       spaceAfter=6, spaceBefore=14,
                                                       textColor=colors.HexColor("#7B61FF"))
                            body_style = ParagraphStyle("body", fontSize=10, fontName="Helvetica",
                                                         spaceAfter=4, leading=14,
                                                         textColor=colors.HexColor("#333333"))
                            mono_style = ParagraphStyle("mono", fontSize=9, fontName="Courier",
                                                         spaceAfter=3, leading=13,
                                                         textColor=colors.HexColor("#555555"))

                            # Title
                            elems.append(Paragraph("InsightForge AI — EDA Report", title_style))
                            elems.append(Paragraph(
                                f"Dataset: {st.session_state.get('active_sample','Uploaded file')}  |  "
                                f"{profile['shape']['rows']:,} rows × {profile['shape']['columns']} columns",
                                body_style))
                            elems.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC")))
                            elems.append(Spacer(1, 10))

                            # Dataset summary table
                            elems.append(Paragraph("Dataset Summary", h2_style))
                            summary_data = [
                                ["Metric", "Value"],
                                ["Rows", f"{profile['shape']['rows']:,}"],
                                ["Columns", str(profile['shape']['columns'])],
                                ["Duplicates", str(profile['duplicates'])],
                                ["Memory", f"{profile['memory_usage_mb']} MB"],
                                ["Missing Values", str(sum(v['count'] for v in profile['missing'].values()))],
                                ["Numeric Features", str(len(feature_summary.get('numeric_features',[])))],
                                ["Categorical Features", str(len(feature_summary.get('categorical_features',[])))],
                            ]
                            t = Table(summary_data, colWidths=[2.5*inch, 4*inch])
                            t.setStyle(TableStyle([
                                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8E8E8")),
                                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                                ("FONTSIZE", (0,0), (-1,-1), 9),
                                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F8F8")]),
                                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
                                ("PADDING", (0,0), (-1,-1), 5),
                            ]))
                            elems.append(t)
                            elems.append(Spacer(1, 10))

                            # Missing values
                            missing_cols = [(k, v["count"], v["pct"]) for k, v in profile["missing"].items() if v["count"] > 0]
                            if missing_cols:
                                elems.append(Paragraph("Missing Values", h2_style))
                                miss_data = [["Column", "Missing Count", "Missing %"]]
                                for c, n, p in sorted(missing_cols, key=lambda x: -x[1]):
                                    miss_data.append([c, str(n), f"{p}%"])
                                t2 = Table(miss_data, colWidths=[3*inch, 2*inch, 1.5*inch])
                                t2.setStyle(TableStyle([
                                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8E8E8")),
                                    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                                    ("FONTSIZE", (0,0), (-1,-1), 9),
                                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F8F8")]),
                                    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
                                    ("PADDING", (0,0), (-1,-1), 5),
                                ]))
                                elems.append(t2)
                                elems.append(Spacer(1, 8))

                            # Correlations
                            strong_pairs = correlations.get("strong_pairs", [])
                            if strong_pairs:
                                elems.append(Paragraph("Strong Correlations", h2_style))
                                corr_data = [["Feature A", "Feature B", "Correlation", "Strength"]]
                                for p in strong_pairs:
                                    corr_data.append([p["feature_a"], p["feature_b"],
                                                      f"{p['correlation']:.4f}", p["strength"]])
                                t3 = Table(corr_data, colWidths=[2*inch, 2*inch, 1.5*inch, 1*inch])
                                t3.setStyle(TableStyle([
                                    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E8E8E8")),
                                    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                                    ("FONTSIZE", (0,0), (-1,-1), 9),
                                    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F8F8")]),
                                    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
                                    ("PADDING", (0,0), (-1,-1), 5),
                                ]))
                                elems.append(t3)
                                elems.append(Spacer(1, 8))

                            # LLM Insights
                            if insights.get("overview") and "disabled" not in insights["overview"]:
                                elems.append(Paragraph("LLM Insights", h2_style))
                                for section, title in [("overview","Overview"), ("anomalies","Anomalies"), ("correlations","Correlations")]:
                                    elems.append(Paragraph(title, ParagraphStyle("sh", fontSize=11,
                                        fontName="Helvetica-Bold", spaceAfter=4, spaceBefore=8,
                                        textColor=colors.HexColor("#444444"))))
                                    text = insights.get(section, "").replace("•", "-")
                                    for line in text.splitlines():
                                        if line.strip():
                                            elems.append(Paragraph(line.strip(), body_style))

                            doc.build(elems)
                            pdf_bytes = buf.getvalue()
                            st.session_state["pdf_report"] = pdf_bytes
                        except ImportError:
                            st.session_state["pdf_report"] = None
                            st.error("Install reportlab to generate PDFs: pip install reportlab")

                if st.session_state.get("pdf_report"):
                    st.download_button(
                        "Download PDF Report",
                        data=st.session_state["pdf_report"],
                        file_name="insightforge_report.pdf",
                        mime="application/pdf",
                    )

    else:
        st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown(f"""
<div style="font-family:'DM Mono',monospace;font-size:12px;color:var(--muted);margin-top:8px">
Showing first 20 rows · {df.shape[0]:,} total rows · {df.shape[1]} columns ·
Click <b style="color:#00D4FF">Run InsightForge Analysis</b> to begin
</div>
""", unsafe_allow_html=True)
