import streamlit as st
import pandas as pd

from core.utils import extract_texts, load_skills, to_table_download
from core.embedding import Embedder
from core.extract import extract_profile
from core.ranking import score_candidates, explain_candidate
from core.visuals import plot_leaderboard, plot_skill_coverage, plot_radar

st.set_page_config(page_title="ResumeMatcher", layout="wide")
st.title("Smart Resume Ranker")

left, right = st.columns([1,1])

with left:
    jd = st.text_area("Paste the Job Description", height=240, placeholder="Responsibilities... Requirements... Preferred...")

with right:
    uploads = st.file_uploader("Upload resumes (PDF, DOCX, or TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)

run = st.button("Analyze")

if run and jd and uploads:
    with st.spinner("Processing..."):
        texts = extract_texts(uploads)
        df = pd.DataFrame([{"candidate_id":k,"raw_text":v} for k,v in texts.items()])
        skills = load_skills("models/skills_taxonomy.txt")
        embedder = Embedder()

        df_prof = df.apply(lambda r: extract_profile(r["raw_text"], skills), axis=1, result_type="expand")
        df = pd.concat([df, df_prof], axis=1)

        scores = score_candidates(df, jd, skills, embedder)

        st.session_state.scores = scores
        st.session_state.df = df
        st.session_state.jd = jd

if "scores" in st.session_state:
    scores = st.session_state.scores

    st.subheader("Ranked Candidates")
    st.plotly_chart(plot_leaderboard(scores), use_container_width=True)

    hide_cols = ["embedding","jd_embedding","raw_text","clean_text","skills_missing","jd_found_skills","years_experience","edu_score"]
    df_show = scores.drop(columns=[c for c in hide_cols if c in scores.columns])

    def color_cgpa(v):
        try:
            v = float(v)
            return "color: green; font-weight: 600;" if v >= 8 else ""
        except:
            return ""

    st.dataframe(df_show.style.applymap(color_cgpa, subset=["cgpa"]))

    cols_to_drop = [c for c in ["embedding","jd_embedding","raw_text"] if c in scores.columns]
    st.download_button(
        "Download results CSV",
        data=to_table_download(scores.drop(columns=cols_to_drop)),
        file_name="resume_matches.csv"
    )

    st.subheader("Insights")

    pick = st.selectbox(
        "Select a candidate",
        scores["candidate_id"].tolist()
    )

    row = scores[scores["candidate_id"] == pick].iloc[0]

    st.plotly_chart(plot_skill_coverage(row["jd_found_skills"], row["jd_missing_skills"]))
    st.plotly_chart(plot_radar(row), use_container_width=True)

    st.markdown("**Explanation**")
    st.markdown("### Candidate Analysis")
    st.markdown(explain_candidate(row).replace("\n", "  \n"), unsafe_allow_html=True)

elif run and (not jd or not uploads):
    st.warning("Provide both a job description and at least one resume.")
