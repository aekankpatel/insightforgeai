# ⚡ InsightForge AI

**Agentic Auto-EDA system powered by Claude LLM — automated data profiling, anomaly detection, correlation analysis, and feature summarization.**

---

## Architecture

```
insightforge/
├── app.py              # Streamlit frontend (dark, futuristic UI)
├── eda_engine.py       # Core pipeline orchestrator + all EDA modules
├── visualizations.py   # Plotly chart generators
├── requirements.txt    # Dependencies
└── README.md
```

### Pipeline Modules (in `eda_engine.py`)

| Module | Class | Function |
|--------|-------|----------|
| Data Profiler | `DataProfiler` | Shape, dtypes, per-column stats (mean, std, skew, kurtosis), missing values |
| Anomaly Detector | `AnomalyDetector` | IQR + Z-score outlier detection per numeric column |
| Correlation Analyzer | `CorrelationAnalyzer` | Pearson correlation matrix, strong pair identification |
| Feature Summarizer | `FeatureSummarizer` | Classifies columns into: numeric, categorical, binary, high-cardinality, ID, datetime |
| LLM Insight Generator | `LLMInsightGenerator` | Claude-powered natural language insights for overview, anomalies, and correlations |
| Pipeline Orchestrator | `InsightForgePipeline` | Runs all modules in sequence, returns unified results dict |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key (optional — or enter in UI)
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Usage

1. **Upload** a CSV/Excel file, or select a built-in sample dataset from the sidebar
2. **Click** "⚡ Run InsightForge Analysis"
3. **Explore** results across 6 tabs:
   - 📊 **Overview** — feature type sunburst + missing value chart
   - 📈 **Distributions** — histograms + categorical bar charts
   - 🔗 **Correlations** — interactive heatmap + strong pair table
   - ⚠️ **Anomalies** — box plots + outlier summary table
   - 🤖 **LLM Insights** — Claude-generated analysis cards
   - 🗂 **Raw Data** — preview, describe(), dtypes, JSON export

---

## Use as a Python Library

```python
import pandas as pd
from eda_engine import InsightForgePipeline

df = pd.read_csv("your_data.csv")
pipeline = InsightForgePipeline(df, api_key="sk-ant-...")
results = pipeline.run(use_llm=True)

# Access results
print(results["profile"]["shape"])          # {"rows": 1000, "columns": 12}
print(results["anomalies"]["price"])        # IQR + Z-score stats
print(results["correlations"]["strong_pairs"])  # Highly correlated feature pairs
print(results["insights"]["overview"])      # LLM-generated insights string
```

### Use individual modules

```python
from eda_engine import DataProfiler, AnomalyDetector, CorrelationAnalyzer

profile = DataProfiler(df).profile()
anomalies = AnomalyDetector(df).detect()
correlations = CorrelationAnalyzer(df).analyze()
```

---

## Sample Datasets (built-in)

| Dataset | Rows | Columns | Use Case |
|---------|------|---------|----------|
| Titanic | 891 | 10 | Classification (missing values, mixed types) |
| Iris | 150 | 5 | Classification (clean, numeric-heavy) |
| Boston Housing | 506 | 14 | Regression (high correlation) |
| Synthetic E-Commerce | 1,200 | 11 | Business analytics (outliers, skewed data) |

---

## LLM Insight Examples

**Overview Insight:**
> • The dataset has 891 rows and 10 columns with a mix of numeric and categorical features.
> • Age (19.8% missing) and Embarked (0.2% missing) require imputation before modeling.
> • Fare is heavily right-skewed — consider log transformation for regression tasks.

**Anomaly Insight:**
> • Fare has 116 outliers (13% of data), with extreme values up to $512 — investigate VIP tickets.
> • SibSp outliers (families with 5-8 siblings) may represent edge cases that need separate treatment.

**Correlation Insight:**
> • Pclass and Fare have strong negative correlation (r = -0.55) — class is a proxy for fare.
> • Consider dropping one to avoid multicollinearity in linear models.

---

## Extending the Pipeline

Add a new module by following this pattern:

```python
class MyCustomModule:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze(self) -> dict:
        # Your logic here
        return {"results": ...}
```

Then add it to `InsightForgePipeline.run()`.
