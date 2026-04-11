import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from core.skill_extractor import _norm, order_skills_jd_first
from core.skill_extractor import extract_skills_whitelist, build_skill_index
from core.extract import normalize_experience, detect_domain

def _onehot_edu(x):
    return {"PhD": 4, "Masters": 3, "Bachelors": 2, "Diploma": 1}.get(x, 0)

def extract_required_skills_from_jd(jd_text, skills):
    jd_text = jd_text.lower()
    required = []
    for s in skills:
        s_lower = s.lower().strip()
        if re.search(rf"(?<![a-z0-9]){re.escape(s_lower)}(?![a-z0-9])", jd_text):
            required.append(s_lower)
    if len(required) < 5:
        words = re.findall(r"[a-zA-Z]{3,}", jd_text)
        for w in words:
            close = get_close_matches(w.lower(), skills, cutoff=0.85)
            required.extend(close)
    return list(set(required)) if required else skills[:15]


# ── Domain weight profiles ─────────────────────────────────────────────────────
# similarity  = semantic embedding cosine sim to JD
# skills      = JD skill coverage score
# rarity      = how rare/valuable the candidate's skills are in this pool
# experience  = normalised years (domain-aware cap)
# education   = degree level score
# recency     = how recent the experience is
# cgpa        = academic score (low weight for experienced roles)

WEIGHT_PROFILES = {
    "web_development": {
        "similarity": 0.35, "skills": 0.35, "rarity": 0.10,
        "experience": 0.08, "education": 0.06, "recency": 0.04, "cgpa": 0.02,
    },
    "data_science": {
        "similarity": 0.38, "skills": 0.30, "rarity": 0.12,
        "experience": 0.08, "education": 0.07, "recency": 0.03, "cgpa": 0.02,
    },
    "data_analytics": {
        "similarity": 0.35, "skills": 0.30, "rarity": 0.10,
        "experience": 0.12, "education": 0.06, "recency": 0.05, "cgpa": 0.02,
    },
    "data_engineering": {
        "similarity": 0.35, "skills": 0.32, "rarity": 0.10,
        "experience": 0.12, "education": 0.05, "recency": 0.04, "cgpa": 0.02,
    },
    "devops": {
        "similarity": 0.33, "skills": 0.33, "rarity": 0.12,
        "experience": 0.12, "education": 0.05, "recency": 0.04, "cgpa": 0.01,
    },
    "engineering": {
        "similarity": 0.36, "skills": 0.32, "rarity": 0.12,
        "experience": 0.10, "education": 0.05, "recency": 0.03, "cgpa": 0.02,
    },
    "mobile_development": {
        "similarity": 0.35, "skills": 0.35, "rarity": 0.10,
        "experience": 0.08, "education": 0.06, "recency": 0.04, "cgpa": 0.02,
    },
    # Finance — experience & education matter a lot; tools matter less
    "finance": {
        "similarity": 0.28, "skills": 0.20, "rarity": 0.07,
        "experience": 0.25, "education": 0.12, "recency": 0.06, "cgpa": 0.02,
    },
    # Accounting — qualifications (CA/CPA) dominate; experience important
    "accounting": {
        "similarity": 0.25, "skills": 0.20, "rarity": 0.07,
        "experience": 0.22, "education": 0.18, "recency": 0.06, "cgpa": 0.02,
    },
    # Legal — qualification + experience critical; semantic match matters
    "legal": {
        "similarity": 0.25, "skills": 0.20, "rarity": 0.08,
        "experience": 0.30, "education": 0.12, "recency": 0.05, "cgpa": 0.02,
    },
    "business": {
        "similarity": 0.35, "skills": 0.20, "rarity": 0.08,
        "experience": 0.25, "education": 0.07, "recency": 0.04, "cgpa": 0.01,
    },
    "consulting": {
        "similarity": 0.33, "skills": 0.18, "rarity": 0.08,
        "experience": 0.22, "education": 0.12, "recency": 0.05, "cgpa": 0.02,
    },
    "supply_chain": {
        "similarity": 0.30, "skills": 0.28, "rarity": 0.08,
        "experience": 0.22, "education": 0.07, "recency": 0.04, "cgpa": 0.01,
    },
    "marketing": {
        "similarity": 0.35, "skills": 0.28, "rarity": 0.10,
        "experience": 0.17, "education": 0.05, "recency": 0.04, "cgpa": 0.01,
    },
    "hr": {
        "similarity": 0.33, "skills": 0.25, "rarity": 0.08,
        "experience": 0.20, "education": 0.08, "recency": 0.05, "cgpa": 0.01,
    },
    "sales": {
        "similarity": 0.28, "skills": 0.20, "rarity": 0.05,
        "experience": 0.32, "education": 0.05, "recency": 0.08, "cgpa": 0.02,
    },
    "product": {
        "similarity": 0.35, "skills": 0.25, "rarity": 0.10,
        "experience": 0.20, "education": 0.05, "recency": 0.04, "cgpa": 0.01,
    },
    "healthcare": {
        "similarity": 0.28, "skills": 0.22, "rarity": 0.08,
        "experience": 0.22, "education": 0.15, "recency": 0.04, "cgpa": 0.01,
    },
    "education": {
        "similarity": 0.30, "skills": 0.22, "rarity": 0.08,
        "experience": 0.25, "education": 0.10, "recency": 0.04, "cgpa": 0.01,
    },
    "general": {
        "similarity": 0.35, "skills": 0.25, "rarity": 0.10,
        "experience": 0.18, "education": 0.07, "recency": 0.03, "cgpa": 0.02,
    },
}


def score_candidates(df, jd, skills, embedder):
    skill_idx        = build_skill_index(skills)
    jd_required      = extract_skills_whitelist(jd, skill_idx, n_max=4, fuzzy=False)
    jd_required_norm = set(_norm(s) for s in jd_required)

    jd_domain = detect_domain(jd)
    w = WEIGHT_PROFILES.get(jd_domain, WEIGHT_PROFILES["general"])

    out = df.copy()

    out["jd_found_skills"] = out["skills_found"].apply(
        lambda r: [s for s in r if _norm(s) in jd_required_norm]
    )
    out["jd_missing_skills"] = out["jd_found_skills"].apply(
        lambda found: sorted(list(jd_required_norm - set(_norm(s) for s in found)))
    )
    out["skills_found"] = out["skills_found"].apply(
        lambda r: order_skills_jd_first(r, jd_required_norm)
    )

    texts    = df["clean_text"].tolist()
    emb      = embedder.encode(texts + [jd])
    cand_emb, jd_emb = emb[:-1], emb[-1:]
    sim      = cosine_similarity(cand_emb, jd_emb).ravel()

    coverage = out["jd_found_skills"].apply(
        lambda r: len(r) / max(1, len(jd_required))
    )

    from core.skill_extractor import compute_rarity_scores
    rarity = compute_rarity_scores(df)
    out["skill_value_score"] = out["skills_found"].apply(
        lambda sk: sum(rarity.get(_norm(s), 0) for s in sk) / max(1, len(sk))
    )

    exp_norm = df.apply(
        lambda r: normalize_experience(
            r.get("years_experience", 0) or 0,
            jd_domain
        ), axis=1
    )

    cgpa_norm = (df["cgpa"].fillna(0) / 10).clip(0, 1)
    edu_norm  = df["education"].fillna("Other").apply(_onehot_edu) / 4.0
    rec_norm  = df["recency"].fillna(0)

    final = (
        w["similarity"]  * sim      +
        w["skills"]      * coverage +
        w["rarity"]      * out["skill_value_score"] +
        w["experience"]  * exp_norm +
        w["education"]   * edu_norm +
        w["recency"]     * rec_norm +
        w["cgpa"]        * cgpa_norm
    )

    out["jd_similarity"]      = sim
    out["skill_coverage"]     = coverage
    out["skill_rarity_score"] = out["skill_value_score"]
    out["edu_score"]          = edu_norm
    out["exp_score"]          = exp_norm
    out["recency_score"]      = rec_norm
    out["cgpa_score"]         = cgpa_norm
    out["jd_domain"]          = jd_domain
    out["final_score"]        = final

    return out.sort_values("final_score", ascending=False).reset_index(drop=True)


def explain_candidate(row):
    jd_set = set(_norm(s) for s in row.get("jd_found_skills", []))
    domain_label = row.get("jd_domain", "general").replace("_", " ").title()
    parts = [
        f"**Detected Domain:** {domain_label}",
        f"Similarity to Job Description: {row['jd_similarity']:.2f}",
        f"Skill Coverage: {row['skill_coverage']:.2f}",
        f"Experience: {row['months_experience']} months ({row['years_experience']:.2f} years)",
        f"Education: {row.get('education', 'N/A')} (score: {row['edu_score']:.2f})",
        f"Recency Score: {row['recency_score']:.2f}",
    ]
    cgpa = row.get("cgpa")
    parts.append(f"CGPA: {cgpa:.2f}" if cgpa and not pd.isna(cgpa) else "CGPA: not mentioned")

    all_sk = row.get("skills_found", [])
    view = [
        f"<span style='color:#4ade80;font-weight:600'>{s}</span>"
        if _norm(s) in jd_set else s
        for s in all_sk
    ]
    parts += [
        "",
        "**Matched Skills (green = JD match):**",
        ", ".join(view) if view else "None detected",
        "",
        "**Missing Skills for this Role:**",
        ", ".join(row.get("jd_missing_skills", [])) or "None — full coverage!",
    ]
    return "\n".join(parts)
