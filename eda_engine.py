"""
InsightForge AI - Core EDA Engine
Modular pipeline for automated exploratory data analysis with LLM-powered insights.
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# MODULE 1: DATA PROFILER
# ─────────────────────────────────────────

class DataProfiler:
    def __init__(self, df):
        self.df = df

    def profile(self):
        df = self.df
        profile = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": {},
            "missing": {},
            "duplicates": int(df.duplicated().sum()),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        }
        for col in df.columns:
            col_data = df[col]
            missing_count = int(col_data.isna().sum())
            missing_pct = round(missing_count / len(df) * 100, 2)
            col_profile = {
                "dtype": str(col_data.dtype),
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_count": int(col_data.nunique()),
            }
            if pd.api.types.is_numeric_dtype(col_data):
                desc = col_data.describe()
                col_profile.update({
                    "mean": round(float(desc["mean"]), 4) if not pd.isna(desc["mean"]) else None,
                    "std": round(float(desc["std"]), 4) if not pd.isna(desc["std"]) else None,
                    "min": round(float(desc["min"]), 4) if not pd.isna(desc["min"]) else None,
                    "max": round(float(desc["max"]), 4) if not pd.isna(desc["max"]) else None,
                    "q25": round(float(desc["25%"]), 4) if not pd.isna(desc["25%"]) else None,
                    "median": round(float(desc["50%"]), 4) if not pd.isna(desc["50%"]) else None,
                    "q75": round(float(desc["75%"]), 4) if not pd.isna(desc["75%"]) else None,
                    "skewness": round(float(col_data.skew()), 4),
                    "kurtosis": round(float(col_data.kurt()), 4),
                })
            else:
                vc = col_data.value_counts()
                col_profile["top_values"] = {str(k): int(v) for k, v in vc.head(5).items()}
            profile["columns"][col] = col_profile
            profile["missing"][col] = {"count": missing_count, "pct": missing_pct}
        return profile


# ─────────────────────────────────────────
# MODULE 2: ANOMALY DETECTOR
# ─────────────────────────────────────────

class AnomalyDetector:
    def __init__(self, df):
        self.df = df

    def detect(self):
        results = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            series = self.df[col].dropna()
            if len(series) < 4:
                continue
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            iqr_outliers = series[(series < lower) | (series > upper)]
            z_scores = (series - series.mean()) / series.std()
            zscore_outliers = series[np.abs(z_scores) > 3]
            results[col] = {
                "iqr_outlier_count": int(len(iqr_outliers)),
                "iqr_outlier_pct": round(len(iqr_outliers) / len(series) * 100, 2),
                "iqr_bounds": {"lower": round(float(lower), 4), "upper": round(float(upper), 4)},
                "zscore_outlier_count": int(len(zscore_outliers)),
                "zscore_outlier_pct": round(len(zscore_outliers) / len(series) * 100, 2),
                "outlier_values_sample": [round(float(v), 4) for v in iqr_outliers.head(5).tolist()],
            }
        return results


# ─────────────────────────────────────────
# MODULE 3: CORRELATION ANALYZER
# ─────────────────────────────────────────

class CorrelationAnalyzer:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {"matrix": {}, "strong_pairs": [], "note": "Not enough numeric columns"}
        corr_matrix = numeric_df.corr(method="pearson")
        strong_pairs = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val) and abs(val) > 0.5:
                    strong_pairs.append({
                        "feature_a": cols[i],
                        "feature_b": cols[j],
                        "correlation": round(float(val), 4),
                        "direction": "positive" if val > 0 else "negative",
                        "strength": "strong" if abs(val) > 0.7 else "moderate",
                    })
        strong_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return {
            "matrix": {col: {c: round(float(v), 4) for c, v in row.items()}
                       for col, row in corr_matrix.to_dict().items()},
            "strong_pairs": strong_pairs[:10],
            "numeric_columns": cols,
        }


# ─────────────────────────────────────────
# MODULE 4: FEATURE SUMMARIZER
# ─────────────────────────────────────────

class FeatureSummarizer:
    def __init__(self, df):
        self.df = df

    def summarize(self):
        summary = {
            "numeric_features": [], "categorical_features": [],
            "high_cardinality_features": [], "binary_features": [],
            "potential_id_columns": [], "datetime_features": [],
        }
        for col in self.df.columns:
            series = self.df[col]
            unique_ratio = series.nunique() / len(series)
            if pd.api.types.is_datetime64_any_dtype(series):
                summary["datetime_features"].append(col)
            elif pd.api.types.is_numeric_dtype(series):
                if series.nunique() == 2:
                    summary["binary_features"].append(col)
                elif unique_ratio > 0.9 and series.nunique() > 100:
                    summary["potential_id_columns"].append(col)
                else:
                    summary["numeric_features"].append(col)
            else:
                if series.nunique() == 2:
                    summary["binary_features"].append(col)
                elif series.nunique() > 50:
                    summary["high_cardinality_features"].append(col)
                else:
                    summary["categorical_features"].append(col)
        return summary


# ─────────────────────────────────────────
# MODULE 5: LLM INSIGHT GENERATOR (GROQ)
# ─────────────────────────────────────────

class LLMInsightGenerator:
    """
    Uses Groq API for LLM insights.
    - 14,400 free requests/day (resets daily)
    - Single API call for all three sections (overview, anomalies, correlations)
    - Works on hosted websites (Streamlit Cloud, Render, etc.)
    """

    def __init__(self, api_key=None):
        import os
        from groq import Groq
        key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = Groq(api_key=key)
        self.model = "llama-3.1-8b-instant"  # fast, free, capable

    def _call_groq(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[Groq Error: {str(e)}]"

    def generate_all_insights(self, profile, feature_summary, anomalies, correlation):
        """Single API call returning all three insight sections at once."""
        significant_anomalies = {k: v for k, v in anomalies.items() if v["iqr_outlier_count"] > 0}
        strong_pairs = correlation.get("strong_pairs", [])

        prompt = f"""You are a senior data scientist. Analyze this dataset and return insights using EXACTLY these three section headers:

OVERVIEW:
(3-5 bullet points about data quality, feature types, missing data, modeling challenges. Use • for bullets.)

ANOMALIES:
(3-5 bullet points about outliers and recommended treatments. Use • for bullets. If none, say so briefly.)

CORRELATIONS:
(3-5 bullet points about feature relationships and multicollinearity. Use • for bullets. If none, say so briefly.)

--- DATASET INFO ---
Shape: {profile['shape']['rows']} rows x {profile['shape']['columns']} columns
Duplicates: {profile['duplicates']}
Memory: {profile['memory_usage_mb']} MB
Numeric features: {feature_summary['numeric_features']}
Categorical features: {feature_summary['categorical_features']}
Binary features: {feature_summary['binary_features']}
High cardinality: {feature_summary['high_cardinality_features']}
Potential IDs: {feature_summary['potential_id_columns']}
Missing data: {json.dumps({k: v for k, v in profile['missing'].items() if v['count'] > 0})}
Outliers (IQR): {json.dumps(significant_anomalies) if significant_anomalies else "None detected"}
Strong correlations: {json.dumps(strong_pairs) if strong_pairs else "None found"}

Return ONLY the three labeled sections above. No preamble, no extra commentary."""

        raw = self._call_groq(prompt)

        if "[Groq Error" in raw:
            return {"overview": raw, "anomalies": raw, "correlations": raw}

        def extract_section(text, header):
            try:
                start = text.index(header) + len(header)
                other = [h for h in ["OVERVIEW:", "ANOMALIES:", "CORRELATIONS:"] if h != header]
                end = len(text)
                for h in other:
                    if h in text[start:]:
                        end = min(end, start + text[start:].index(h))
                return text[start:end].strip()
            except ValueError:
                return "Analysis not available for this section."

        return {
            "overview": extract_section(raw, "OVERVIEW:"),
            "anomalies": extract_section(raw, "ANOMALIES:"),
            "correlations": extract_section(raw, "CORRELATIONS:"),
        }


# ─────────────────────────────────────────
# MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────

class InsightForgePipeline:
    def __init__(self, df, api_key=None):
        self.df = df
        self.api_key = api_key
        self.results = {}

    def run(self, use_llm=True, progress_callback=None):
        def update(msg):
            if progress_callback:
                progress_callback(msg)

        update("Running data profiling...")
        self.results["profile"] = DataProfiler(self.df).profile()

        update("Detecting anomalies...")
        self.results["anomalies"] = AnomalyDetector(self.df).detect()

        update("Analyzing correlations...")
        self.results["correlations"] = CorrelationAnalyzer(self.df).analyze()

        update("Summarizing features...")
        self.results["feature_summary"] = FeatureSummarizer(self.df).summarize()

        if use_llm:
            update("Generating LLM insights...")
            llm = LLMInsightGenerator(api_key=self.api_key)
            self.results["insights"] = llm.generate_all_insights(
                self.results["profile"],
                self.results["feature_summary"],
                self.results["anomalies"],
                self.results["correlations"],
            )
        else:
            self.results["insights"] = {
                "overview": "LLM insights disabled.",
                "anomalies": "LLM insights disabled.",
                "correlations": "LLM insights disabled.",
            }

        update("Analysis complete!")
        return self.results

def compute_health_scores(profile, anomalies, feature_summary, correlations):
    """Compute dataset health scores across 5 dimensions. Returns dict of {dim: {score, notes}}."""

    def score_completeness():
        missing_pct = sum(v["count"] for v in profile["missing"].values()) / max(
            profile["shape"]["rows"] * profile["shape"]["columns"], 1) * 100
        if missing_pct == 0:   return 100, "No missing values"
        if missing_pct < 2:    return 90,  f"{missing_pct:.1f}% missing — negligible"
        if missing_pct < 10:   return 70,  f"{missing_pct:.1f}% missing — moderate"
        if missing_pct < 25:   return 45,  f"{missing_pct:.1f}% missing — high"
        return 20, f"{missing_pct:.1f}% missing — critical"

    def score_consistency():
        dup_pct = profile["duplicates"] / max(profile["shape"]["rows"], 1) * 100
        if dup_pct == 0:      return 100, "No duplicates"
        if dup_pct < 1:       return 85,  f"{dup_pct:.1f}% duplicates — minor"
        if dup_pct < 5:       return 65,  f"{dup_pct:.1f}% duplicates — notable"
        return 35, f"{dup_pct:.1f}% duplicates — significant"

    def score_outliers():
        if not anomalies:     return 100, "No numeric columns to check"
        high = [c for c, v in anomalies.items() if v["iqr_outlier_pct"] > 10]
        med  = [c for c, v in anomalies.items() if 2 < v["iqr_outlier_pct"] <= 10]
        if not high and not med: return 100, "No significant outliers"
        if not high:             return 75,  f"{len(med)} columns with moderate outliers"
        return 45, f"{len(high)} columns with heavy outliers"

    def score_feature_quality():
        id_cols    = len(feature_summary.get("potential_id_columns", []))
        high_card  = len(feature_summary.get("high_cardinality_features", []))
        total      = profile["shape"]["columns"]
        problem_ratio = (id_cols + high_card) / max(total, 1)
        if problem_ratio == 0:    return 100, "All features are clean types"
        if problem_ratio < 0.1:   return 85,  f"{id_cols} ID-like, {high_card} high-cardinality columns"
        if problem_ratio < 0.25:  return 65,  f"Several problematic feature types"
        return 40, "Many ID/high-cardinality columns — needs cleanup"

    def score_correlation_structure():
        pairs = correlations.get("strong_pairs", [])
        perfect = [p for p in pairs if abs(p["correlation"]) > 0.95]
        if not pairs:        return 100, "No multicollinearity issues"
        if not perfect:      return 80,  f"{len(pairs)} moderate correlations — manageable"
        return 50, f"{len(perfect)} near-perfect correlations — multicollinearity risk"

    s1, n1 = score_completeness()
    s2, n2 = score_consistency()
    s3, n3 = score_outliers()
    s4, n4 = score_feature_quality()
    s5, n5 = score_correlation_structure()

    return {
        "Completeness":          {"score": s1, "note": n1},
        "Consistency":           {"score": s2, "note": n2},
        "Outlier Health":        {"score": s3, "note": n3},
        "Feature Quality":       {"score": s4, "note": n4},
        "Correlation Structure": {"score": s5, "note": n5},
    }
