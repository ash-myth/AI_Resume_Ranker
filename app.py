import streamlit as st
import pandas as pd
from core.utils import extract_texts, load_skills, to_table_download
from core.embedding import Embedder
from core.extract import extract_profile
from core.ranking import score_candidates, explain_candidate
from core.visuals import plot_leaderboard, plot_skill_coverage, plot_radar

st.set_page_config(page_title="ResumeIQ", layout="wide", page_icon="🎯")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #080a10;
    color: #d4d8e8;
}

header[data-testid="stHeader"] {
    display: none !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1240px !important;
}

/* ── HEADER BAND ── */
.riq-header {
    background: #0d1120;
    border: 1px solid #1c2038;
    border-radius: 14px;
    padding: 1.4rem 1.8rem;
    margin-top: 1rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.riq-header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.riq-logo {
    width: 44px;
    height: 44px;
    background: #3b5bdb;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    line-height: 1;
    flex-shrink: 0;
}

.riq-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0 0 4px 0;
}

.riq-subtitle {
    font-size: 0.78rem;
    color: #6b7aaa;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.04em;
    margin: 0;
}

.riq-badge {
    background: rgba(59, 91, 219, 0.12);
    border: 1px solid rgba(59, 91, 219, 0.3);
    color: #7899f6;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    padding: 5px 12px;
    border-radius: 6px;
    letter-spacing: 0.06em;
    white-space: nowrap;
}

/* ── SECTION LABELS ── */
.riq-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #4a5480;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 0.5rem 0;
}

/* ── DOMAIN PILL ── */
.riq-domain {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: #0d1120;
    border: 1px solid #1c2440;
    border-left: 3px solid #3b5bdb;
    border-radius: 8px;
    padding: 0.65rem 1.1rem;
    font-size: 0.83rem;
    color: #9aa8d0;
    margin-bottom: 1.6rem;
}

.riq-domain strong {
    color: #7899f6;
    font-weight: 600;
}

/* ── STAT CARDS ── */
.riq-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.8rem;
}

.riq-stat {
    background: #0d1020;
    border: 1px solid #1a1e30;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    position: relative;
    overflow: hidden;
}

.riq-stat::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: #3b5bdb;
    opacity: 0.5;
}

.riq-stat-val {
    font-size: 1.85rem;
    font-weight: 800;
    color: #ffffff;
    font-family: 'DM Mono', monospace;
    line-height: 1;
    letter-spacing: -0.03em;
}

.riq-stat-lbl {
    font-size: 0.68rem;
    color: #4a5480;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 5px;
    font-weight: 600;
}

/* ── INPUTS ── */
.stTextArea textarea {
    background: #0d1020 !important;
    border: 1px solid #1c2038 !important;
    border-radius: 10px !important;
    color: #d4d8e8 !important;
    font-size: 0.875rem !important;
    font-family: 'Syne', sans-serif !important;
    transition: border-color 0.15s ease !important;
}

.stTextArea textarea:focus {
    border-color: #3b5bdb !important;
    box-shadow: 0 0 0 3px rgba(59, 91, 219, 0.12) !important;
    outline: none !important;
}

.stFileUploader {
    border: 1px dashed #1c2440 !important;
    border-radius: 10px !important;
    background: #0d1020 !important;
    padding: 0.5rem !important;
    transition: border-color 0.15s ease !important;
}

.stFileUploader:hover {
    border-color: #3b5bdb !important;
}

/* ── ANALYZE BUTTON ── */
.stButton > button {
    background: #3b5bdb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
}

.stButton > button:hover {
    background: #2f4cc4 !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
    background: #2643ad !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1020 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid #1a1e30 !important;
    margin-bottom: 1.5rem !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4a5480 !important;
    border-radius: 7px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: color 0.15s ease !important;
}

.stTabs [aria-selected="true"] {
    background: #161b35 !important;
    color: #7899f6 !important;
}

/* ── SELECT ── */
.stSelectbox > div > div {
    background: #0d1020 !important;
    border: 1px solid #1c2038 !important;
    border-radius: 8px !important;
    color: #d4d8e8 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    background: #0d1020 !important;
    color: #7899f6 !important;
    border: 1px solid #263060 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    transition: all 0.15s ease !important;
}

.stDownloadButton > button:hover {
    background: #3b5bdb !important;
    color: #ffffff !important;
    border-color: #3b5bdb !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid #131626 !important;
    margin: 1.5rem 0 !important;
}

/* ── HEADINGS ── */
h2, h3, h4 {
    color: #e8ecf8 !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em !important;
}

/* ── DATAFRAME ── */
.stDataFrame {
    border: 1px solid #1a1e30 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── WARNING ── */
.stAlert {
    background: #130e00 !important;
    border: 1px solid #3d2a00 !important;
    border-radius: 8px !important;
    color: #d4a84b !important;
}

/* ── SPINNER ── */
.stSpinner > div {
    border-top-color: #3b5bdb !important;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="riq-header">
    <div class="riq-header-left">
        <div class="riq-logo">🎯</div>
        <div>
            <div class="riq-title">ResumeIQ</div>
            <div class="riq-subtitle">AI-powered candidate ranking engine</div>
        </div>
    </div>
    <div class="riq-badge">v1.0 · Semantic Matching</div>
</div>
""", unsafe_allow_html=True)

# ── INPUT ROW ────────────────────────────────────────────────────────────
col_jd, col_up = st.columns([1.1, 0.9])

with col_jd:
    st.markdown('<p class="riq-label">Job Description</p>', unsafe_allow_html=True)
    jd = st.text_area(
        label="jd",
        height=220,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed"
    )

with col_up:
    st.markdown('<p class="riq-label">Candidate Resumes</p>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        label="upload",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⚡  Analyze Candidates")

st.markdown("---")

# ── RUN PIPELINE ─────────────────────────────────────────────────────────
if run and jd and uploads:
    with st.spinner("Analyzing candidates..."):
        texts    = extract_texts(uploads)
        df       = pd.DataFrame([{"candidate_id": k, "raw_text": v} for k, v in texts.items()])
        skills   = load_skills("models/skills_taxonomy.txt")
        embedder = Embedder()
        df_prof  = df.apply(lambda r: extract_profile(r["raw_text"], skills), axis=1, result_type="expand")
        df       = pd.concat([df, df_prof], axis=1)
        scores   = score_candidates(df, jd, skills, embedder)
        st.session_state.scores = scores
        st.session_state.jd     = jd

elif run and (not jd or not uploads):
    st.warning("Please provide both a job description and at least one resume.")

# ── RESULTS ──────────────────────────────────────────────────────────────
if "scores" in st.session_state:
    scores = st.session_state.scores
    domain = scores["jd_domain"].iloc[0] if "jd_domain" in scores.columns else "general"
    domain_label = domain.replace("_", " ").title()

    st.markdown(f"""
    <div class="riq-domain">
        🎯 &nbsp;Detected domain: <strong>{domain_label}</strong>
        &nbsp;·&nbsp; Scoring weights calibrated for this role type
    </div>
    """, unsafe_allow_html=True)

    top_score    = scores["final_score"].max()
    avg_score    = scores["final_score"].mean()
    n_candidates = len(scores)
    top_skills   = len(scores["skills_found"].iloc[0]) if "skills_found" in scores.columns else 0

    st.markdown(f"""
    <div class="riq-stats">
        <div class="riq-stat">
            <div class="riq-stat-val">{n_candidates}</div>
            <div class="riq-stat-lbl">Candidates</div>
        </div>
        <div class="riq-stat">
            <div class="riq-stat-val">{top_score:.0%}</div>
            <div class="riq-stat-lbl">Top Score</div>
        </div>
        <div class="riq-stat">
            <div class="riq-stat-val">{avg_score:.0%}</div>
            <div class="riq-stat-lbl">Avg Score</div>
        </div>
        <div class="riq-stat">
            <div class="riq-stat-val">{top_skills}</div>
            <div class="riq-stat-lbl">Top Skills Found</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_chart, tab_table, tab_insights = st.tabs(
        ["📊  Leaderboard", "📋  Data Table", "🔍  Deep Dive"]
    )

    with tab_chart:
        st.plotly_chart(plot_leaderboard(scores), use_container_width=True)

    with tab_table:
        hide_cols = [
            "embedding", "jd_embedding", "raw_text", "clean_text",
            "skills_missing", "jd_found_skills", "years_experience",
            "edu_score", "skill_value_score", "jd_domain"
        ]
        df_show = scores.drop(columns=[c for c in hide_cols if c in scores.columns])

        def color_cgpa(v):
            try:
                return "color: #4ade80; font-weight: 600;" if float(v) >= 8 else ""
            except:
                return ""

        st.dataframe(
            df_show.style.map(color_cgpa, subset=["cgpa"] if "cgpa" in df_show.columns else []),
            use_container_width=True,
            height=400
        )
        cols_to_drop = [c for c in ["embedding", "jd_embedding", "raw_text"] if c in scores.columns]
        st.download_button(
            "⬇  Download CSV",
            data=to_table_download(scores.drop(columns=cols_to_drop)),
            file_name="resume_matches.csv",
            mime="text/csv"
        )

    with tab_insights:
        pick = st.selectbox(
            "Select candidate",
            scores["candidate_id"].tolist(),
            format_func=lambda x: x.replace(".pdf", "").replace(".docx", "").replace(".txt", "")
        )
        row = scores[scores["candidate_id"] == pick].iloc[0]

        left_col, right_col = st.columns(2)
        with left_col:
            st.plotly_chart(plot_skill_coverage(row["jd_found_skills"], row["jd_missing_skills"]), use_container_width=True)
        with right_col:
            st.plotly_chart(plot_radar(row), use_container_width=True)

        st.markdown("#### Analysis")
        st.markdown(explain_candidate(row).replace("\n", "  \n"), unsafe_allow_html=True)
