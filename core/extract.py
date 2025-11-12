import re, numpy as np, pandas as pd
def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()
from datetime import datetime

MONTHS = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,"jun":6,"june":6,
    "jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12
}

def _parse_to_month_year(token):
    token = token.lower().strip()

    # present
    if token in ["present","current","now"]:
        today = datetime.today()
        return today.year, today.month

    # Jun 2024 or August 2023
    m = re.match(r"([a-z]{3,9})\s+(\d{4})", token)
    if m:
        month = MONTHS.get(m.group(1), 1)
        year = int(m.group(2))
        return year, month

    # 01/06/2025 or 06/2024
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", token)
    if m:
        return int(m.group(3)), int(m.group(2))

    # 06/2024
    m = re.match(r"(\d{1,2})/(\d{4})", token)
    if m:
        return int(m.group(2)), int(m.group(1))

    return None, None


def extract_years_of_experience(text):
    text = text.lower()

    ranges = re.findall(
        r"([A-Za-z]{3,9}\s+\d{4}|\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{4}|present|current|now)"
        r"\s*(?:-|to|–|—|\s)\s*"
        r"([A-Za-z]{3,9}\s+\d{4}|\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{4}|present|current|now)",
        text,
        flags=re.I
    )

    total_months = 0
    seen = set()

    for start, end in ranges:
        sy, sm = _parse_to_month_year(start)
        ey, em = _parse_to_month_year(end)

        if sy and ey:
            months = (ey - sy) * 12 + (em - sm) + 1
            if 1 <= months <= 18:
                key = (sy, sm, ey, em)
                if key not in seen:
                    seen.add(key)
                    total_months += months

    # fallback: explicit "x months"
    if total_months == 0:
        m2 = re.findall(r"(\d+)\s+months?", text)
        for m in m2:
            total_months += int(m)

    years = round(total_months / 12, 2)
    months = total_months

    return years, months




def extract_education_level(t):
    t = t.lower()
    bachelor_patterns = [
        r"b\.?\s*tech",
        r"b\s*tech",
        r"b\.?\s*e",
        r"bachelor",
        r"undergraduate",
        r"ug program",
        r"graduation"
    ]

    master_patterns = [
        r"m\.?\s*tech",
        r"m\s*tech",
        r"m\.?\s*sc",
        r"master",
        r"post\s*graduate",
        r"pg program"
    ]

    for p in bachelor_patterns:
        if re.search(p, t):
            return "Bachelors"

    for p in master_patterns:
        if re.search(p, t):
            return "Masters"

    return "Other"



def extract_cgpa(t):
    t = t.lower()

    # Case: "7.43 CGPA"
    m = re.search(r"(\d\.\d{1,2})\s*cgpa", t)
    if m:
        try:
            cg = float(m.group(1))
            if 0.0 < cg <= 10.0:
                return round(cg, 2)
        except:
            pass

    # Case: "CGPA: 7.43" or "CGPA - 7.43" or "CGPA 7.43"
    m = re.search(r"cgpa\s*[:=\- ]\s*(\d\.\d{1,2})", t)
    if m:
        try:
            cg = float(m.group(1))
            if 0.0 < cg <= 10.0:
                return round(cg, 2)
        except:
            pass

    # Case: GPA 8.15
    m = re.search(r"gpa\s*[:=\- ]\s*(\d\.\d{1,2})", t)
    if m:
        try:
            cg = float(m.group(1))
            if 0.0 < cg <= 10.0:
                return round(cg, 2)
        except:
            pass

    # Case: "7.43/10"
    m = re.search(r"(\d\.\d{1,2})\s*/\s*10", t)
    if m:
        try:
            cg = float(m.group(1))
            if 0.0 < cg <= 10.0:
                return round(cg, 2)
        except:
            pass

    return None

def extract_contacts(text):
    import unicodedata

    # Normalize text
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u00A0", " ")
    t = re.sub(r"[^\x00-\x7F]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # ---- EMAIL SUPER-ROBUST MODE ----
    # Step 1: Try normal match (works for clean resumes)
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", t)
    if m:
        email = m.group(0)
    else:
        # Step 2: Reconstruct email from scattered characters
        compressed = re.sub(r"\s+", "", text)            # remove spaces
        compressed = re.sub(r"[^A-Za-z0-9@._+-]", "", compressed)  # keep only email characters
        m2 = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", compressed)
        email = m2.group(0) if m2 else ""

    # ---- PHONE SUPER-ROBUST MODE ----
    digits = re.sub(r"\D", "", text)
    candidates = []
    for i in range(len(digits) - 9):
        chunk = digits[i:i+10]
        if chunk[0] in "6789":
            candidates.append(chunk)
    phone = candidates[0] if candidates else ""

    return email, phone



def recency_score(text):
    text = text.lower()

    matches = re.findall(
        r"(intern|internship|experience|project|work|employed|role|position|data|ml|ai|analyst)[\s\S]{0,40}?(20\d{2})",
        text,
        flags=re.I
    )

    # Case 1: Found relevant work/project years
    if matches:
        years = [int(y[1]) for y in matches]  # y[1] extracts only the year
        latest = max(years)

    else:
        # Case 2: look for any years but ignore schooling (2018 and before)
        years = [int(y) for y in re.findall(r"20\d{2}", text) if int(y) > 2018]

        if not years:
            return 0.6  # soft neutral default

        latest = max(years)

    gap = 2025 - latest

    # Convert recency gap into score
    if gap <= 0:
        return 1.0      
    elif gap == 1:
        return 0.9
    elif gap == 2:
        return 0.75
    elif gap <= 4:
        return 0.6
    return 0.45
from core.skill_extractor import build_skill_index, extract_skills_whitelist
def extract_profile(t, skills):
    t = clean_text(t)
    yrs,months = extract_years_of_experience(t)
    edu = extract_education_level(t)
    email, phone = extract_contacts(t)
    skill_idx = build_skill_index(skills, synonyms={
        "power bi":["powerbi","ms power bi"],
        "scikit-learn":["sklearn","scikit learn"],
        "pytorch":["py torch"],
        "mysql":["my sql"],
        "postgresql":["postgre sql","postgres"],
        "computer vision":["cv"],
        "nlp":["natural language processing"],
        "llm":["large language model","large language models"],
        "tensorflow":["tf"],
        "aws":["amazon web services"],
        "gcp":["google cloud","google cloud platform"],
        "docker":["containers","containerization"],
        "rest api":["restful api","rest apis"],
        "fastapi":["fast api"],
        "opencv":["open cv"]
    })
    skills_found = extract_skills_whitelist(t, skill_idx, n_max=4, fuzzy=False)
    rec = recency_score(t)
    return pd.Series({
        "clean_text": t,
        "years_experience": yrs,
        "months_experience": months,
        "education": edu,
        "email": email,
        "phone": phone,
        "skills_found": skills_found,
        "recency": rec,
        "cgpa": extract_cgpa(t),
        "total_skills_found": len(skills_found)
    })
