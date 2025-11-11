# Smart Resume Ranker

Smart Resume Ranker is an end to end resume screening and ranking system. You provide a job description and upload multiple resumes. The system extracts relevant information, identifies skills, computes semantic similarity, evaluates candidate profile strength, and produces a sorted list of the best matches. It also provides visual explanations and skill gap insights for each candidate.

---

## Demo Flow

1. Upload resumes (PDF, DOCX, or TXT)
2. Paste the job description
3. System analyzes candidates
4. Ranked results and visual insights are displayed
5. Export the rankings to CSV if needed

---

## Features

- Transformer based semantic matching with automatic fallback to TF IDF
- Skill extraction using curated taxonomy and synonym mapping
- Weighted scoring considering:
  - JD similarity
  - Skill coverage percentage
  - Skill rarity score among applied candidates
  - Experience duration
  - Education qualification level
  - Work or project recency
  - CGPA if available
- Interactive visualizations:
  - Candidate leaderboard chart
  - Skill match vs gap donut chart
  - Candidate profile radar chart
- Export final results to CSV

---

## Project Structure

```
AI_Resume_Ranker
│ app.py Main Streamlit application
│ README.md Documentation
│ requirements.txt List of dependencies
│ skill_db_relax_20.json Extended skill dataset for mapping
│ token_dist.json Token distribution reference
│
├── core
│ embedding.py Embedding model and similarity scoring
│ extract.py Resume parsing and structured profile extraction
│ ranking.py Weighted scoring and candidate ranking logic
│ skill_extractor.py Skill detection, normalization, and synonym support
│ utils.py File readers and helper utility functions
│ visuals.py Visualization and chart generation utilities
│
└── models
skills_taxonomy.txt Core domain specific skill whitelist
```

---

## Installation

```
git clone https://github.com/<your-username>/ResumeMatcherPro.git
cd ResumeMatcherPro
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## How It Works

1. Resume text is extracted from PDF, DOCX, or TXT files.
2. Text is cleaned and standardized for consistent processing.
3. Skills are identified using a curated skill taxonomy along with synonym mapping.
4. The job description and each resume are converted into vector embeddings.
   - If the transformer model is not available, TF IDF is used automatically.
5. Each candidate receives separate sub scores for:
   - JD semantic similarity
   - Skill coverage percentage (skills required vs skills present)
   - Skill rarity (unique or less common skills across the applicant set)
   - Total experience duration
   - Education qualification level
   - Recency of latest project or work
   - CGPA if detected
6. All sub scores are normalized and combined using a weighted formula.
7. Final scores determine the ranking order shown in the results.
8. Interactive charts and candidate specific breakdowns are displayed for clarity.

---

## Customization

To modify or expand detected skills: ```models/skills_taxonomy.txt ```

To change scoring weights: ```core/ranking.py```

To adjust visualization styling: ```core/visuals.py```

---

## Output Provided

- Ranked candidate table with final score
- Skill match and missing skill breakdown
- Experience and education summaries
- Skill coverage donut chart
- Candidate profile radar chart
- Leaderboard bar chart
- Downloadable CSV of complete ranking results

## Use Cases

- HR recruitment pre-screening
- Placement cell automation in universities
- Internship applicant filtering
- Technical project team selection
- Large batch resume evaluation

## License

This project is intended for academic, research, and demonstration purposes.  
You may modify and adapt it for institutional or personal workflows.
