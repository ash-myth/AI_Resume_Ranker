import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from core.skill_extractor import _norm, order_skills_jd_first
from core.skill_extractor import extract_skills_whitelist, build_skill_index
from core.extract import normalize_experience, detect_domain

# ── Education one-hot ──────────────────────────────────────────────────────────
def _onehot_edu(x):
    return {"PhD": 3, "Masters": 2, "Bachelors": 1}.get(x, 0)


# ── JD Required Skills ─────────────────────────────────────────────────────────
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
    required = list(set(required))
    return required if required else skills[:15]


# ── Domain-aware weight profiles ──────────────────────────────────────────────
WEIGHT_PROFILES = {
    "data_science": {
        "similarity": 0.38, "skills": 0.30, "rarity": 0.12,
        "experience": 0.08, "education": 0.07, "recency": 0.03, "cgpa": 0.02
    },
    "finance": {
        "similarity": 0.30, "skills": 0.22, "rarity": 0.08,
        "experience": 0.22, "education": 0.10, "recency": 0.05, "cgpa": 0.03
    },
    "business": {
        "similarity": 0.35, "skills": 0.20, "rarity": 0.08,
        "experience": 0.25, "education": 0.07, "recency": 0.04, "cgpa": 0.01
    },
    "marketing": {
        "similarity": 0.35, "skills": 0.28, "rarity": 0.10,
        "experience": 0.17, "education": 0.05, "recency": 0.04, "cgpa": 0.01
    },
    "hr": {
        "similarity": 0.33, "skills": 0.25, "rarity": 0.08,
        "experience": 0.20, "education": 0.08, "recency": 0.05, "cgpa": 0.01
    },
    "product": {
        "similarity": 0.35, "skills": 0.25, "rarity": 0.10,
        "experience": 0.20, "education": 0.05, "recency": 0.04, "cgpa": 0.01
    },
    "engineering": {
        "similarity": 0.36, "skills": 0.32, "rarity": 0.12,
        "experience": 0.10, "education": 0.05, "recency": 0.03, "cgpa": 0.02
    },
    "sales": {
        "similarity": 0.30, "skills": 0.20, "rarity": 0.05,
        "experience": 0.30, "education": 0.05, "recency": 0.08, "cgpa": 0.02
    },
    "general": {
        "similarity": 0.35, "skills": 0.25, "rarity": 0.10,
        "experience": 0.18, "education": 0.07, "recency": 0.03, "cgpa": 0.02
    },
}


# ── Main Scorer ────────────────────────────────────────────────────────────────
def score_candidates(df, jd, skills, embedder):
    skill_idx   = build_skill_index(skills)
    jd_required = extract_skills_whitelist(jd, skill_idx, n_max=4, fuzzy=False)
    jd_required_norm = set(_norm(s) for s in jd_required)

    # Detect JD domain to pick weight profile
    jd_domain = detect_domain(jd)
    w = WEIGHT_PROFILES.get(jd_domain, WEIGHT_PROFILES["general"])

    out = df.copy()

    # ── JD skill match / missing (computed BEFORE reordering) ─────────────────
    out["jd_found_skills"] = out["skills_found"].apply(
        lambda r: [s for s in r if _norm(s) in jd_required_norm]
    )
    out["jd_missing_skills"] = out["jd_found_skills"].apply(
        lambda found: sorted(list(jd_required_norm - set(_norm(s) for s in found)))
    )

    # Reorder: JD-matching skills first for display
    out["skills_found"] = out["skills_found"].apply(
        lambda r: order_skills_jd_first(r, jd_required_norm)
    )

    # ── Embedding similarity ───────────────────────────────────────────────────
    texts   = df["clean_text"].tolist()
    emb     = embedder.encode(texts + [jd])
    cand_emb, jd_emb = emb[:-1], emb[-1:]
    sim     = cosine_similarity(cand_emb, jd_emb).ravel()

    # ── Skill coverage ────────────────────────────────────────────────────────
    coverage = out["jd_found_skills"].apply(
        lambda r: len(r) / max(1, len(jd_required))
    )

    # ── Skill rarity ─────────────────────────────────────────────────────────
    from core.skill_extractor import compute_rarity_scores
    rarity = compute_rarity_scores(df)
    out["skill_value_score"] = out["skills_found"].apply(
        lambda sk: sum(rarity.get(_norm(s), 0) for s in sk) / max(1, len(sk))
    )

    # ── Experience (domain-aware cap) ─────────────────────────────────────────
    exp_norm = df.apply(
        lambda r: normalize_experience(
            r.get("years_experience", 0) or 0,
            r.get("domain", jd_domain)
        ), axis=1
    )

    # ── Other factors ─────────────────────────────────────────────────────────
    cgpa_norm = (df["cgpa"].fillna(0) / 10).clip(0, 1)
    edu_norm  = df["education"].fillna("Other").apply(_onehot_edu) / 3.0
    rec_norm  = df["recency"].fillna(0)

    # ── Final weighted score ───────────────────────────────────────────────────
    final = (
        w["similarity"]  * sim +
        w["skills"]      * coverage +
        w["rarity"]      * out["skill_value_score"] +
        w["experience"]  * exp_norm +
        w["education"]   * edu_norm +
        w["recency"]     * rec_norm +
        w["cgpa"]        * cgpa_norm
    )

    out["jd_similarity"]       = sim
    out["skill_coverage"]      = coverage
    out["skill_rarity_score"]  = out["skill_value_score"]
    out["edu_score"]           = edu_norm
    out["exp_score"]           = exp_norm
    out["recency_score"]       = rec_norm
    out["cgpa_score"]          = cgpa_norm
    out["jd_domain"]           = jd_domain
    out["final_score"]         = final

    return out.sort_values(by="final_score", ascending=False).reset_index(drop=True)


# ── Explanation ────────────────────────────────────────────────────────────────
def explain_candidate(row):
    jd_set = set(_norm(s) for s in row.get("jd_found_skills", []))
    parts  = []
    parts.append(f"**Detected Domain:** {row.get('jd_domain', 'general').replace('_', ' ').title()}")
    parts.append(f"Similarity to Job Description: {row['jd_similarity']:.2f}")
    parts.append(f"Skill Coverage: {row['skill_coverage']:.2f}")
    parts.append(f"Experience: {row['months_experience']} months ({row['years_experience']:.2f} years)")
    parts.append(f"Education: {row.get('education', 'N/A')} (score: {row['edu_score']:.2f})")
    parts.append(f"Recency Score: {row['recency_score']:.2f}")
    cgpa = row.get("cgpa")
    parts.append(f"CGPA: {cgpa:.2f}" if cgpa and not pd.isna(cgpa) else "CGPA: not mentioned")

    all_sk = row.get("skills_found", [])
    view = []
    for s in all_sk:
        if _norm(s) in jd_set:
            view.append(f"<span style='color:#1a7f37;font-weight:600'>{s}</span>")
        else:
            view.append(s)
    parts.append("")
    parts.append("**Matched Skills (green = JD match):**")
    parts.append(", ".join(view) if view else "None detected")

    gaps = row.get("jd_missing_skills", [])
    parts.append("")
    parts.append("**Missing Skills for this Role:**")
    parts.append(", ".join(gaps) if gaps else "None — full coverage!")
    return "\n".join(parts)
