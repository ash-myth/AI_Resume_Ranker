import numpy as np, pandas as pd, re
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from core.skill_extractor import _norm, order_skills_jd_first
from core.skill_extractor import extract_skills_whitelist, build_skill_index

def _onehot_edu(x):
    m = {"PhD":3,"Masters":2,"Bachelors":1}
    return m.get(x,0)

# Determine which skills JD is actually asking for
def extract_required_skills_from_jd(jd_text, skills):
    jd_text = jd_text.lower()
    required = []

    for s in skills:
        s = s.lower().strip()
        # Match only exact term presence, not fuzzy
        if re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", jd_text):
            required.append(s)

    # If JD explicitly lists very few skills (<5), keep a fallback to top matches
    if len(required) < 5:
        # Extract nouns-like words as backup
        words = re.findall(r"[a-zA-Z]{3,}", jd_text)
        for w in words:
            close = get_close_matches(w.lower(), skills, cutoff=0.85)
            required.extend(close)

    required = list(set(required))  # unique

    return required if required else skills[:15]  # ensure length ~8–12, not 40+


def score_candidates(df, jd, skills, embedder):
    # --- Identify JD-required skills ---
    skill_idx = build_skill_index(skills)
    jd_required = extract_skills_whitelist(jd, skill_idx, n_max=4, fuzzy=False)
    jd_required_norm = set(_norm(s) for s in jd_required)

    out = df.copy()

    # --- Mark JD skill matches / missing ---
    out["jd_found_skills"] = out["skills_found"].apply(
        lambda r: [s for s in r if _norm(s) in jd_required_norm]
    )
    out["jd_missing_skills"] = out["skills_found"].apply(
        lambda r: sorted(list(jd_required_norm - set(_norm(s) for s in r)))
    )

    # Move JD-matching skills to the front for display
    out["skills_found"] = out["skills_found"].apply(
        lambda r: order_skills_jd_first(r, jd_required_norm)
    )

    # --- Embedding similarity ---
    texts = df["clean_text"].tolist()
    emb = embedder.encode(texts + [jd])
    cand_emb, jd_emb = emb[:-1], emb[-1:]
    sim = cosine_similarity(cand_emb, jd_emb).ravel()

    # --- JD Skill Coverage ---
    coverage = out["jd_found_skills"].apply(
        lambda r: len(r) / max(1, len(jd_required))
    )

    # --- NEW: Skill Rarity Score ---
    from core.skill_extractor import compute_rarity_scores
    rarity = compute_rarity_scores(df)   # gives rarity weight per skill

    out["skill_value_score"] = out["skills_found"].apply(
        lambda skills: sum(rarity.get(_norm(s), 0) for s in skills) / max(1, len(skills))
    )

    # --- Other Normalized Factors ---
    exp_norm = (df["years_experience"].fillna(0) / 10).clip(0, 1)
    cgpa_norm = (df["cgpa"].fillna(0) / 10).clip(0, 1)
    edu_norm = df["education"].fillna("Other").apply(_onehot_edu) / 3.0
    rec_norm = df["recency"].fillna(0)

    # --- Weights ---
    w = {
        "similarity": 0.38,
        "skills": 0.28,
        "rarity": 0.14,
        "experience": 0.08,
        "education": 0.06,
        "recency": 0.03,
        "cgpa": 0.03
    }

    # --- Final Score ---
    final = (
        w["similarity"] * sim +
        w["skills"] * coverage +
        w["rarity"] * out["skill_value_score"] +
        w["experience"] * exp_norm +
        w["education"] * edu_norm +
        w["recency"] * rec_norm +
        w["cgpa"] * cgpa_norm
    )

    # Save Scores
    out["jd_similarity"] = sim
    out["skill_coverage"] = coverage
    out["skill_rarity_score"] = out["skill_value_score"]
    out["edu_score"] = edu_norm
    out["exp_score"] = exp_norm
    out["recency_score"] = rec_norm
    out["cgpa_score"] = cgpa_norm
    out["final_score"] = final

    # Rank Top to Bottom
    out = out.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    return out


def explain_candidate(row):
    jd_set = set(_norm(s) for s in row.get("jd_found_skills", []))
    parts = []
    parts.append(f"Similarity to Job Description: {row['jd_similarity']:.2f}")
    parts.append(f"Skill Coverage: {row['skill_coverage']:.2f}")
    parts.append(f"Internship Experience: {row['months_experience']} months ({row['years_experience']:.2f} years)")
    parts.append(f"Education Level Score: {row['edu_score']:.2f}")
    parts.append(f"Recency Score: {row['recency_score']:.2f}")
    if row.get("cgpa") and not pd.isna(row["cgpa"]):
        parts.append(f"CGPA: {row['cgpa']:.2f}")
    else:
        parts.append("CGPA not mentioned.")
    all_sk = row.get("skills_found", [])
    view = []
    for s in all_sk:
        if _norm(s) in jd_set:
            view.append(f"<span style='color:#1a7f37;font-weight:600'>{s}</span>")
        else:
            view.append(s)
    parts.append("")
    parts.append("Strengths (Skills Present):")
    parts.append(", ".join(view) if view else "None")
    gaps = row.get("jd_missing_skills", [])
    parts.append("")
    parts.append("Skill Gaps (Missing Skills for Role):")
    parts.append(", ".join(gaps) if gaps else "None")
    return "\n".join(parts)

