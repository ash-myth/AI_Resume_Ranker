import re
import numpy as np
import pandas as pd
from datetime import datetime

# ── Text Cleaning ──────────────────────────────────────────────────────────────
def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# ── Date Parsing ───────────────────────────────────────────────────────────────
MONTHS = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,
    "apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
    "aug":8,"august":8,"sep":9,"sept":9,"september":9,
    "oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
}

def _parse_to_month_year(token):
    token = token.lower().strip()
    if token in ["present", "current", "now"]:
        t = datetime.today()
        return t.year, t.month
    m = re.match(r"([a-z]{3,9})\s+(\d{4})", token)
    if m:
        return int(m.group(2)), MONTHS.get(m.group(1), 1)
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", token)
    if m:
        return int(m.group(3)), int(m.group(2))
    m = re.match(r"(\d{1,2})/(\d{4})", token)
    if m:
        return int(m.group(2)), int(m.group(1))
    return None, None


# ── Experience Extraction ──────────────────────────────────────────────────────
def extract_years_of_experience(text):
    text = text.lower()
    ranges = re.findall(
        r"([A-Za-z]{3,9}\s+\d{4}|\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{4}|present|current|now)"
        r"\s*(?:-|to|–|—|\s)\s*"
        r"([A-Za-z]{3,9}\s+\d{4}|\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{4}|present|current|now)",
        text, flags=re.I
    )
    total_months = 0
    seen = set()
    for start, end in ranges:
        sy, sm = _parse_to_month_year(start)
        ey, em = _parse_to_month_year(end)
        if sy and ey:
            months = (ey - sy) * 12 + (em - sm) + 1
            if 1 <= months <= 600:   # up to 50 years for senior professionals
                key = (sy, sm, ey, em)
                if key not in seen:
                    seen.add(key)
                    total_months += months
    if total_months == 0:
        m2 = re.findall(r"(\d+)\s+months?", text)
        for m in m2:
            total_months += int(m)
    return round(total_months / 12, 2), total_months


# ── Education Extraction ───────────────────────────────────────────────────────
def extract_education_level(t):
    t = t.lower()
    phd_patterns    = [r"ph\.?\s*d", r"doctor", r"doctoral"]
    master_patterns = [r"m\.?\s*tech", r"m\s*tech", r"m\.?\s*sc", r"mba",
                       r"m\.?\s*b\.?\s*a", r"master", r"post\s*graduate", r"pg program",
                       r"pgdm", r"m\.?\s*com", r"m\.?\s*a\.?\b"]
    bachelor_patterns = [r"b\.?\s*tech", r"b\s*tech", r"b\.?\s*e\.?\b", r"bachelor",
                         r"undergraduate", r"ug program", r"graduation", r"b\.?\s*sc",
                         r"b\.?\s*com", r"b\.?\s*b\.?\s*a", r"b\.?\s*a\.?\b"]
    for p in phd_patterns:
        if re.search(p, t): return "PhD"
    for p in master_patterns:
        if re.search(p, t): return "Masters"
    for p in bachelor_patterns:
        if re.search(p, t): return "Bachelors"
    return "Other"


# ── CGPA Extraction ────────────────────────────────────────────────────────────
def extract_cgpa(t):
    t = t.lower()
    patterns = [
        r"(\d\.\d{1,2})\s*cgpa",
        r"cgpa\s*[:=\-\s]\s*(\d\.\d{1,2})",
        r"gpa\s*[:=\-\s]\s*(\d\.\d{1,2})",
        r"(\d\.\d{1,2})\s*/\s*10",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            try:
                cg = float(m.group(1))
                if 0.0 < cg <= 10.0:
                    return round(cg, 2)
            except Exception:
                pass
    return None


# ── Contact Extraction ─────────────────────────────────────────────────────────
def extract_contacts(text):
    import unicodedata
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u00A0", " ")
    t = re.sub(r"[^\x00-\x7F]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
    if m:
        email = m.group(0)
    else:
        compressed = re.sub(r"[^A-Za-z0-9@._+-]", "", re.sub(r"\s+", "", text))
        m2 = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", compressed)
        email = m2.group(0) if m2 else ""

    digits = re.sub(r"\D", "", text)
    candidates = [digits[i:i+10] for i in range(len(digits) - 9) if digits[i] in "6789"]
    phone = candidates[0] if candidates else ""
    return email, phone


# ── Recency Score ──────────────────────────────────────────────────────────────
def recency_score(text):
    """Dynamic recency — always compares against current year."""
    current_year = datetime.today().year
    text = text.lower()

    matches = re.findall(
        r"(intern|internship|experience|project|work|employed|role|position|"
        r"data|ml|ai|analyst|manager|consultant|engineer|developer|executive)"
        r"[\s\S]{0,40}?(20\d{2})",
        text, flags=re.I
    )

    if matches:
        years = [int(y[1]) for y in matches]
        latest = max(years)
    else:
        years = [int(y) for y in re.findall(r"20\d{2}", text) if int(y) > 2015]
        if not years:
            return 0.6
        latest = max(years)

    gap = current_year - latest
    if gap <= 0:  return 1.0
    elif gap == 1: return 0.9
    elif gap == 2: return 0.75
    elif gap <= 4: return 0.6
    return 0.45


# ── Domain Detection ───────────────────────────────────────────────────────────
DOMAIN_SIGNALS = {
    "data_science":  ["machine learning", "deep learning", "scikit", "pandas", "numpy",
                      "random forest", "xgboost", "neural network", "nlp", "computer vision",
                      "data science", "model", "training", "inference", "eda"],
    "business":      ["business development", "strategy", "consulting", "operations",
                      "stakeholder", "client", "revenue", "p&l", "kpi", "okr",
                      "go to market", "market research"],
    "finance":       ["financial modeling", "valuation", "dcf", "equity", "investment",
                      "portfolio", "accounting", "audit", "balance sheet", "p&l",
                      "cfa", "bloomberg", "capital markets", "mergers"],
    "marketing":     ["seo", "sem", "digital marketing", "campaign", "google ads",
                      "social media", "content", "brand", "conversion", "crm",
                      "email marketing", "growth"],
    "hr":            ["recruitment", "talent", "onboarding", "hris", "employee engagement",
                      "performance management", "compensation", "workforce"],
    "product":       ["product management", "roadmap", "user research", "wireframe",
                      "figma", "ux", "user story", "sprint", "backlog"],
    "engineering":   ["software development", "backend", "frontend", "api", "system design",
                      "microservices", "devops", "docker", "kubernetes", "ci/cd"],
    "sales":         ["sales", "b2b", "b2c", "lead generation", "account management",
                      "quota", "pipeline", "cold calling", "revenue"],
}

def detect_domain(text):
    """Return top domain detected from text (JD or resume)."""
    t = text.lower()
    scores = {}
    for domain, signals in DOMAIN_SIGNALS.items():
        scores[domain] = sum(1 for s in signals if s in t)
    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


# ── Experience Normalization (domain-aware) ────────────────────────────────────
EXP_CAPS = {
    # Freshers/students: cap at 2 yrs so internships still score well
    "data_science": 2,
    "engineering":  2,
    # Mid-level roles
    "product":      8,
    "marketing":    8,
    "hr":           8,
    "business":     10,
    "sales":        10,
    "finance":      12,
    "general":      10,
}

def normalize_experience(years, domain="general"):
    cap = EXP_CAPS.get(domain, 10)
    return min(years / cap, 1.0)


# ── Main Profile Extractor ─────────────────────────────────────────────────────
from core.skill_extractor import build_skill_index, extract_skills_whitelist

def extract_profile(t, skills):
    t = clean_text(t)
    yrs, months = extract_years_of_experience(t)
    edu         = extract_education_level(t)
    email, phone = extract_contacts(t)
    cgpa        = extract_cgpa(t)
    rec         = recency_score(t)
    domain      = detect_domain(t)

    skill_idx = build_skill_index(skills)
    skills_found = extract_skills_whitelist(t, skill_idx, n_max=4, fuzzy=False)

    return pd.Series({
        "clean_text":        t,
        "years_experience":  yrs,
        "months_experience": months,
        "education":         edu,
        "email":             email,
        "phone":             phone,
        "skills_found":      skills_found,
        "recency":           rec,
        "cgpa":              cgpa,
        "total_skills_found": len(skills_found),
        "domain":            domain,
    })
