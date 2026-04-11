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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0f1117; color: #e8eaf0; }
.block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 1200px; }
.resume-header { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid #1e2130; }
.resume-header h1 { font-size: 1.6rem; font-weight: 600; color: #ffffff; margin: 0; letter-spacing: -0.02em; }
.resume-header span { font-size: 0.8rem; color: #6b7280; font-family: 'DM Mono', monospace; background: #1a1d27; padding: 3px 10px; border-radius: 20px; border: 1px solid #2a2d3a; }
.input-label { font-size: 0.72rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
.stTextArea textarea { background: #1a1d27 !important; border: 1px solid #2a2d3a !important; border-radius: 8px !important; color: #e8eaf0 !important; font-size: 0.875rem !important; }
.stTextArea textarea:focus { border-color: #4f6ef7 !important; box-shadow: 0 0 0 2px rgba(79,110,247,0.15) !important; }
.stButton > button { background: #4f6ef7 !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 0.875rem !important; padding: 0.6rem 2rem !important; width: 100% !important; transition: all 0.2s ease !important; }
.stButton > button:hover { background: #3a56d4 !important; transform: translateY(-1px) !important; box-shadow: 0 4px 20px rgba(79,110,247,0.3) !important; }
.domain-badge { display: inline-flex; align-items: center; gap: 8px; background: #1a1d27; border: 1px solid #2a2d3a; border-left: 3px solid #4f6ef7; border-radius: 8px; padding: 0.6rem 1rem; font-size: 0.85rem; color: #c4c9e0; margin-bottom: 1.5rem; }
.domain-badge strong { color: #4f6ef7; }
.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 1.5rem; }
.stat-card { background: #13151f; border: 1px solid #1e2130; border-radius: 10px; padding: 1rem 1.2rem; text-align: center; }
.stat-card .val { font-size: 1.6rem; font-weight: 600; color: #ffffff; font-family: 'DM Mono', monospace; line-height: 1; }
.stat-card .lbl { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
.stTabs [data-baseweb="tab-list"] { background: #13151f !important; border-radius: 10px !important; padding: 4px !important; gap: 2px !important; border: 1px solid #1e2130 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #6b7280 !important; border-radius: 8px !important; font-weight: 500 !important; font-size: 0.85rem !important; padding: 0.4rem 1rem !important; }
.stTabs [aria-selected="true"] { background: #1e2336 !important; color: #4f6ef7 !important; }
.stSelectbox > div > div { background: #1a1d27 !important; border: 1px solid #2a2d3a !important; border-radius: 8px !important; color: #e8eaf0 !important; }
.stDownloadButton > button { background: #1a1d27 !important; color: #4f6ef7 !important; border: 1px solid #4f6ef7 !important; border-radius: 8px !important; font-weight: 500 !important; font-size: 0.8rem !important; }
.stDownloadButton > button:hover { background: #4f6ef7 !important; color: white !important; }
h2, h3, h4 { color: #ffffff !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="resume-header">
    <h1>🎯 ResumeIQ</h1>
    <span>AI-Powered Resume Ranker</span>
</div>
""", unsafe_allow_html=True)

col_jd, col_up = st.columns([1.1, 0.9])
with col_jd:
    st.markdown('<div class="input-label">Job Description</div>', unsafe_allow_html=True)
    jd = st.text_area(label="jd", height=220, placeholder="Paste the full job description here...", label_visibility="collapsed")
with col_up:
    st.markdown('<div class="input-label">Candidate Resumes</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(label="upload", type=["pdf","docx","txt"], accept_multiple_files=True, label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("⚡ Analyze Candidates")

st.markdown("---")

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

if "scores" in st.session_state:
    scores = st.session_state.scores
    domain = scores["jd_domain"].iloc[0] if "jd_domain" in scores.columns else "general"
    domain_label = domain.replace("_", " ").title()

    st.markdown(f"""
    <div class="domain-badge">
        🎯 Detected domain: <strong>{domain_label}</strong> &nbsp;·&nbsp; Scoring weights calibrated for this role type
    </div>""", unsafe_allow_html=True)

    top_score    = scores["final_score"].max()
    avg_score    = scores["final_score"].mean()
    n_candidates = len(scores)
    top_skills   = len(scores["skills_found"].iloc[0]) if "skills_found" in scores.columns else 0

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card"><div class="val">{n_candidates}</div><div class="lbl">Candidates</div></div>
        <div class="stat-card"><div class="val">{top_score:.0%}</div><div class="lbl">Top Score</div></div>
        <div class="stat-card"><div class="val">{avg_score:.0%}</div><div class="lbl">Avg Score</div></div>
        <div class="stat-card"><div class="val">{top_skills}</div><div class="lbl">Top Skills Found</div></div>
    </div>""", unsafe_allow_html=True)

    tab_chart, tab_table, tab_insights = st.tabs(["📊  Leaderboard", "📋  Data Table", "🔍  Deep Dive"])

    with tab_chart:
        st.plotly_chart(plot_leaderboard(scores), use_container_width=True)

    with tab_table:
        hide_cols = ["embedding","jd_embedding","raw_text","clean_text","skills_missing",
                     "jd_found_skills","years_experience","edu_score","skill_value_score","jd_domain"]
        df_show = scores.drop(columns=[c for c in hide_cols if c in scores.columns])
        def color_cgpa(v):
            try: return "color: #4ade80; font-weight: 600;" if float(v) >= 8 else ""
            except: return ""
        st.dataframe(
            df_show.style.map(color_cgpa, subset=["cgpa"] if "cgpa" in df_show.columns else []),
            use_container_width=True, height=400
        )
        cols_to_drop = [c for c in ["embedding","jd_embedding","raw_text"] if c in scores.columns]
        st.download_button("⬇ Download CSV", data=to_table_download(scores.drop(columns=cols_to_drop)), file_name="resume_matches.csv", mime="text/csv")

    with tab_insights:
        pick = st.selectbox("Select candidate", scores["candidate_id"].tolist(),
                            format_func=lambda x: x.replace(".pdf","").replace(".docx","").replace(".txt",""))
        row  = scores[scores["candidate_id"] == pick].iloc[0]
        left_col, right_col = st.columns(2)
        with left_col:
            st.plotly_chart(plot_skill_coverage(row["jd_found_skills"], row["jd_missing_skills"]), use_container_width=True)
        with right_col:
            st.plotly_chart(plot_radar(row), use_container_width=True)
        st.markdown("#### Analysis")
        st.markdown(explain_candidate(row).replace("\n", "  \n"), unsafe_allow_html=True)
