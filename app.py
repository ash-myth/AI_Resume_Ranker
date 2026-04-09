import streamlit as st
import pandas as pd
from core.utils import extract_texts, load_skills, to_table_download
from core.embedding import Embedder
from core.extract import extract_profile
from core.ranking import score_candidates, explain_candidate
from core.visuals import plot_leaderboard, plot_skill_coverage, plot_radar

st.set_page_config(page_title="Smart Resume Ranker", layout="wide", page_icon="🎯")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

.app-header {
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 2rem; padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.app-header h1 { font-size: 1.6rem; font-weight: 600; margin: 0; letter-spacing: -0.02em; }
.app-header .badge {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; font-weight: 500;
    background: #2563eb; color: #fff; padding: 3px 8px; border-radius: 4px;
    letter-spacing: 0.06em; text-transform: uppercase; margin-top: 2px;
}

.section-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6b7280; margin-bottom: 0.5rem;
}

.metric-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.5rem 0; }
.metric-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 1rem 1.25rem;
}
.metric-card .metric-val { font-size: 1.8rem; font-weight: 600; line-height: 1; letter-spacing: -0.03em; color: #f9fafb; }
.metric-card .metric-label { font-size: 0.72rem; color: #9ca3af; margin-top: 4px; font-weight: 500; }
.metric-card.blue  { border-left: 3px solid #3b82f6; }
.metric-card.green { border-left: 3px solid #10b981; }
.metric-card.amber { border-left: 3px solid #f59e0b; }
.metric-card.rose  { border-left: 3px solid #f43f5e; }

.candidate-pill {
    display: inline-block; background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.3); border-radius: 20px;
    padding: 4px 14px; font-size: 0.8rem; color: #93c5fd; font-weight: 500; margin-bottom: 1rem;
}

.soft-divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.5rem 0; }

.analysis-block {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 1.25rem 1.5rem;
    font-size: 0.88rem; line-height: 1.7; color: #d1d5db;
}

.stButton > button {
    background: #2563eb; color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.75rem; font-family: 'DM Sans', sans-serif;
    font-weight: 500; font-size: 0.9rem; transition: background 0.15s;
}
.stButton > button:hover { background: #1d4ed8; color: white; }

.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: rgba(255,255,255,0.03); border-radius: 8px;
    padding: 4px; border: 1px solid rgba(255,255,255,0.07);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px; font-size: 0.82rem; font-weight: 500;
    padding: 6px 16px; color: #9ca3af;
}
.stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }

.stSelectbox label {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span style="font-size:1.6rem">🎯</span>
    <h1>Smart Resume Ranker</h1>
    <span class="badge">AI Powered</span>
</div>
""", unsafe_allow_html=True)


# ── Embedder (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_embedder():
    return Embedder()


# ── Input Section ─────────────────────────────────────────────────────────────
col_jd, col_up = st.columns([1.1, 0.9], gap="large")

with col_jd:
    st.markdown('<div class="section-label">Job Description</div>', unsafe_allow_html=True)
    jd = st.text_area(
        label="jd_input",
        height=220,
        placeholder="Paste responsibilities, requirements, preferred skills...",
        label_visibility="collapsed"
    )

with col_up:
    st.markdown('<div class="section-label">Resume Files</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        label="upload_input",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploads:
        st.markdown(
            f'<div style="font-size:0.8rem;color:#6ee7b7;margin-top:6px;">✓ {len(uploads)} file{"s" if len(uploads)>1 else ""} ready</div>',
            unsafe_allow_html=True
        )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
run = st.button("Analyze Candidates →")


# ── Analysis ──────────────────────────────────────────────────────────────────
if run and jd and uploads:
    with st.spinner("Analyzing resumes..."):
        texts = extract_texts(uploads)
        df = pd.DataFrame([{"candidate_id": k, "raw_text": v} for k, v in texts.items()])
        skills = load_skills("models/skills_taxonomy.txt")
        embedder = load_embedder()
        df_prof = df.apply(lambda r: extract_profile(r["raw_text"], skills), axis=1, result_type="expand")
        df = pd.concat([df, df_prof], axis=1)
        scores = score_candidates(df, jd, skills, embedder)
        st.session_state.scores = scores
        st.session_state.df = df
        st.session_state.jd = jd

elif run and (not jd or not uploads):
    st.warning("Please provide both a job description and at least one resume.")


# ── Results ───────────────────────────────────────────────────────────────────
if "scores" in st.session_state:
    scores = st.session_state.scores

    st.markdown('<hr class="soft-divider">', unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────────
    top = scores.iloc[0]
    avg_sim = scores["jd_similarity"].mean()
    avg_cov = scores["skill_coverage"].mean()
    n = len(scores)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card blue">
            <div class="metric-val">{n}</div>
            <div class="metric-label">Candidates Ranked</div>
        </div>
        <div class="metric-card green">
            <div class="metric-val">{top['final_score']:.2f}</div>
            <div class="metric-label">Top Score</div>
        </div>
        <div class="metric-card amber">
            <div class="metric-val">{avg_sim:.0%}</div>
            <div class="metric-label">Avg JD Similarity</div>
        </div>
        <div class="metric-card rose">
            <div class="metric-val">{avg_cov:.0%}</div>
            <div class="metric-label">Avg Skill Coverage</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_rank, tab_insights = st.tabs(["📊  Rankings", "🔍  Candidate Insights"])

    # ── Tab 1: Rankings ───────────────────────────────────────────────────────
    with tab_rank:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        fig = plot_leaderboard(scores)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#d1d5db",
            margin=dict(t=20, b=20, l=10, r=10),
            height=280,
        )
        fig.update_traces(marker_color="#3b82f6")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        hide_cols = ["embedding", "jd_embedding", "raw_text", "clean_text",
                     "skills_missing", "jd_found_skills", "years_experience",
                     "edu_score", "skill_value_score"]
        df_show = scores.drop(columns=[c for c in hide_cols if c in scores.columns]).copy()

        for col in ["final_score", "jd_similarity", "skill_coverage", "exp_score", "recency_score"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].apply(lambda x: f"{x:.3f}")

        def color_cgpa(v):
            try:
                return "color: #6ee7b7; font-weight: 600;" if float(v) >= 8 else ""
            except:
                return ""

        st.dataframe(
            df_show.style.map(color_cgpa, subset=["cgpa"] if "cgpa" in df_show.columns else []),
            use_container_width=True,
            height=min(40 + len(df_show) * 35, 380),
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        cols_to_drop = [c for c in ["embedding", "jd_embedding", "raw_text"] if c in scores.columns]
        st.download_button(
            "⬇ Download CSV",
            data=to_table_download(scores.drop(columns=cols_to_drop)),
            file_name="resume_rankings.csv",
            mime="text/csv"
        )

    # ── Tab 2: Candidate Insights ─────────────────────────────────────────────
    with tab_insights:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        pick = st.selectbox(
            "CANDIDATE",
            scores["candidate_id"].tolist(),
            format_func=lambda x: f"  {x}"
        )

        row = scores[scores["candidate_id"] == pick].iloc[0]
        rank = scores["candidate_id"].tolist().index(pick) + 1
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")

        st.markdown(
            f'<div class="candidate-pill">{rank_emoji} Rank {rank} · Score {row["final_score"]:.3f}</div>',
            unsafe_allow_html=True
        )

        ch1, ch2 = st.columns(2, gap="medium")

        with ch1:
            st.markdown('<div class="section-label">Skill Coverage</div>', unsafe_allow_html=True)
            fig_pie = plot_skill_coverage(row["jd_found_skills"], row["jd_missing_skills"])
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#d1d5db",
                margin=dict(t=10, b=10, l=10, r=10),
                height=240,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                showlegend=True
            )
            fig_pie.update_traces(marker_colors=["#10b981", "#f43f5e"], textfont_color="#fff")
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            st.markdown('<div class="section-label">Profile Radar</div>', unsafe_allow_html=True)
            fig_rad = plot_radar(row)
            fig_rad.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#d1d5db",
                margin=dict(t=20, b=20, l=20, r=20),
                height=240,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#6b7280"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#9ca3af"),
                )
            )
            fig_rad.update_traces(fillcolor="rgba(59,130,246,0.2)", line_color="#3b82f6")
            st.plotly_chart(fig_rad, use_container_width=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">Analysis</div>', unsafe_allow_html=True)
        analysis_html = explain_candidate(row).replace("\n", "<br>")
        st.markdown(f'<div class="analysis-block">{analysis_html}</div>', unsafe_allow_html=True)
