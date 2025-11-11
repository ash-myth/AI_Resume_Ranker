# ResumeMatcher Pro

End-to-end resume screening tool. Upload multiple resumes, paste a job description, and get ranked matches with explainable insights and interactive visuals.

## Quick start
1) python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
2) pip install -r requirements.txt
3) streamlit run app.py

## Features
- Transformer embeddings with automatic TF-IDF fallback
- Skill taxonomy matching and gap analysis
- Multi-factor scoring: JD similarity, skill coverage, experience, education, recency
- Interactive visuals and per-candidate drilldown
- Exportable CSV of ranked results

## Notes
- If downloads for transformer models fail, the app uses TF-IDF automatically.
- Supported files: PDF, DOCX, TXT.