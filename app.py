import json
import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Email Reply Generator & Evaluator",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, modern typography and clean cards
st.markdown("""
<style>
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px 16px;
        border: 1px solid #e9ecef;
    }
    .score-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .score-high { background-color: #d1e7dd; color: #0f5132; }
    .score-med { background-color: #fff3cd; color: #664d03; }
    .score-low { background-color: #f8d7da; color: #842029; }
    .email-box {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .section-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6c757d;
        font-weight: 700;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

SCORES_FILE = os.path.join("data", "scores.json")

def load_data():
    if not os.path.exists(SCORES_FILE):
        return None
    with open(SCORES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()

# Header
st.title("✉️ Customer Support AI Reply Generator & Evaluator")
st.caption("Automated generation and multi-dimensional LLM-as-a-judge QA evaluation pipeline")

if not data:
    st.warning("⚠️ No evaluation data found at `data/scores.json`.")
    st.info("Please execute the pipeline first using `python run_all.py`.")
    if st.button("Run Pipeline Now"):
        with st.spinner("Executing pipeline (fetch_dataset -> generator -> evaluator)..."):
            from run_all import main as run_pipeline
            run_pipeline()
            st.rerun()
    st.stop()

summary = data.get("summary", {})
evaluations = data.get("evaluations", [])

# ==================== SIDEBAR ====================
st.sidebar.header("📊 Evaluation Overview")

# Key Metrics
avg_composite = summary.get("composite", {}).get("mean", 0.0)
min_composite = summary.get("composite", {}).get("min", 0.0)
max_composite = summary.get("composite", {}).get("max", 0.0)

st.sidebar.metric(
    label="Overall Average Score",
    value=f"{avg_composite:.1f} / 100",
    help="Weighted combination: 70% LLM Judge Dimensions + 30% Lexical Overlap"
)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    st.metric(label="Min Score", value=f"{min_composite:.1f}")
with col_sb2:
    st.metric(label="Max Score", value=f"{max_composite:.1f}")

st.sidebar.markdown("---")

# Category Bar Chart
st.sidebar.subheader("📈 Composite Score by Category")
cat_means = summary.get("category_means", {})
if cat_means:
    df_cat = pd.DataFrame([
        {"Category": cat.title(), "Composite Score": score}
        for cat, score in cat_means.items()
    ])
    st.sidebar.bar_chart(data=df_cat.set_index("Category"), color="#0d6efd")

st.sidebar.markdown("---")

# Dimension Breakdown
st.sidebar.subheader("🎯 QA Dimension Means (1-5)")
dim_means = summary.get("dimension_means", {})
if dim_means:
    st.sidebar.write(f"• **Relevance:** {dim_means.get('relevance', 0):.2f} / 5")
    st.sidebar.write(f"• **Tone:** {dim_means.get('tone', 0):.2f} / 5")
    st.sidebar.write(f"• **Completeness:** {dim_means.get('completeness', 0):.2f} / 5")
    st.sidebar.write(f"• **Conciseness:** {dim_means.get('conciseness', 0):.2f} / 5")
    st.sidebar.write(f"• **Lexical Overlap:** {dim_means.get('overlap', 0) * 100:.1f}%")

st.sidebar.markdown("---")

# Filters
st.sidebar.subheader("🔍 Filters")
all_categories = sorted(list(set(e["category"] for e in evaluations)))
selected_category = st.sidebar.selectbox("Category Filter", ["All"] + [c.title() for c in all_categories])
search_query = st.sidebar.text_input("Search Customer Inquiry", "")

sort_option = st.sidebar.selectbox(
    "Sort By",
    ["Composite (High to Low)", "Composite (Low to High)", "ID (Ascending)"]
)

# ==================== MAIN CONTENT ====================

# Top Metrics Row
top_col1, top_col2, top_col3, top_col4 = st.columns(4)
with top_col1:
    st.metric("Total Inquiries", len(evaluations))
with top_col2:
    st.metric("Judge Score Avg (70%)", f"{dim_means.get('judge_score_100', 0):.1f}%")
with top_col3:
    st.metric("Lexical Overlap Avg (30%)", f"{dim_means.get('overlap', 0) * 100:.1f}%")
with top_col4:
    st.metric("Composite Metric", f"{avg_composite:.1f} / 100")

# Filter evaluations
filtered_evals = evaluations
if selected_category != "All":
    filtered_evals = [e for e in filtered_evals if e["category"].lower() == selected_category.lower()]

if search_query:
    filtered_evals = [
        e for e in filtered_evals
        if search_query.lower() in e["customer_message"].lower() or search_query.lower() in e["id"].lower()
    ]

# Sort evaluations
if sort_option == "Composite (High to Low)":
    filtered_evals = sorted(filtered_evals, key=lambda x: x["scores"]["composite"], reverse=True)
elif sort_option == "Composite (Low to High)":
    filtered_evals = sorted(filtered_evals, key=lambda x: x["scores"]["composite"])
else:
    filtered_evals = sorted(filtered_evals, key=lambda x: x["id"])

# Tabs: Detailed Expanders vs Tabular Overview
tab_expanders, tab_table = st.tabs(["📋 Detailed Email Reviews", "📊 Full Data Table"])

with tab_expanders:
    st.markdown(f"Showing **{len(filtered_evals)}** inquiries:")
    
    for item in filtered_evals:
        email_id = item["id"]
        category = item["category"].title()
        scores = item["scores"]
        comp = scores["composite"]
        
        badge_class = "score-high" if comp >= 75 else ("score-med" if comp >= 65 else "score-low")
        expander_title = f"{email_id} | [{category}] - Composite Score: {comp:.1f}/100 | Relevance: {scores['relevance']}/5 | Tone: {scores['tone']}/5"
        
        with st.expander(expander_title, expanded=False):
            # Customer Message Callout
            st.markdown('<div class="section-label">Customer Inquiry</div>', unsafe_allow_html=True)
            st.info(f"🗣️ **Customer:** {item['customer_message']}")

            # Replies Comparison
            col_gen, col_ref = st.columns(2)
            with col_gen:
                st.markdown('<div class="section-label">AI Generated Reply</div>', unsafe_allow_html=True)
                st.success(f"🤖 {item['generated_reply']}")

            with col_ref:
                st.markdown('<div class="section-label">Reference Ground Truth</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="email-box">📝 {item["reference_reply"]}</div>', unsafe_allow_html=True)

            # Score Breakdown Badges / Columns
            st.markdown('<div class="section-label">Evaluation Rubric & Overlap</div>', unsafe_allow_html=True)
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            sc1.metric("Relevance", f"{scores['relevance']} / 5")
            sc2.metric("Tone", f"{scores['tone']} / 5")
            sc3.metric("Completeness", f"{scores['completeness']} / 5")
            sc4.metric("Conciseness", f"{scores['conciseness']} / 5")
            sc5.metric("Overlap", f"{scores['overlap'] * 100:.1f}%")
            sc6.metric("Composite", f"{comp:.1f} / 100")

            # Judge Reasoning
            if item.get("reasoning"):
                st.caption(f"**Judge Auditor Feedback:** {item['reasoning']}")

with tab_table:
    table_rows = []
    for item in filtered_evals:
        sc = item["scores"]
        table_rows.append({
            "ID": item["id"],
            "Category": item["category"].title(),
            "Relevance (1-5)": sc["relevance"],
            "Tone (1-5)": sc["tone"],
            "Completeness (1-5)": sc["completeness"],
            "Conciseness (1-5)": sc["conciseness"],
            "Judge Avg (100)": sc["judge_score_100"],
            "Overlap %": round(sc["overlap"] * 100, 1),
            "Composite Score": sc["composite"]
        })
    df_table = pd.DataFrame(table_rows)
    try:
        st.dataframe(df_table, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df_table, use_container_width=True, hide_index=True)
