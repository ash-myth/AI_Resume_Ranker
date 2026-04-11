import re
import numpy as np
import pandas as pd
from datetime import datetime

def clean_text(t):
    t = re.sub(r"\s+", " ", t)
    return t.strip()

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

def extract_years_of_experience(text):
    text = text.lower()
    ranges = re.findall(
        r"((?:[A-Za-z]{3,9}\s+\d{4})|(?:\d{4})|\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{4}|present|current|now)"
        r"\s*(?:-|to|–|—|\s)\s*"
        r"((?:[A-Za-z]{3,9}\s+\d{4})|(?:\d{4})|\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{4}|present|current|now)",
        text, flags=re.I
    )
    total_months = 0
    seen = set()
    for start, end in ranges:
        sy, sm = _parse_to_month_year(start)
        ey, em = _parse_to_month_year(end)
        if sy and ey:
            months = (ey - sy) * 12 + (em - sm) + 1
            if 1 <= months <= 600:
                key = (sy, sm, ey, em)
                if key not in seen:
                    seen.add(key)
                    total_months += months
    if total_months == 0:
        m2 = re.findall(r"(\d+)\s+months?", text)
        for m in m2:
            total_months += int(m)
    return round(total_months / 12, 2), total_months

def extract_education_level(t):
    """
    Recognises: PhD, Masters (incl. MBA, MCA, LLM, M.Tech, M.Sc, PGDM),
    Professional qualifications treated as Masters-equivalent
    (CA, CFA, CPA, ACCA, CMA, FRM, MBBS, LLB, B.Pharm etc.),
    Bachelors, and Diploma/Other.
    """
    t_l = t.lower()

    phd_patterns = [
        r"ph\.?\s*d", r"doctor(?:ate)?", r"doctoral",
        r"d\.?\s*sc\b", r"d\.?\s*litt\b",
    ]

    professional_patterns = [
        r"\bca\s+(?:final|qualified|inter|rank|foundation)\b",
        r"\bchartered\s+accountant\b",
        r"\bicai\b",
        r"\bcfa\b",                         
        r"\bcpa\b",                         
        r"\bacca\b",                      
        r"\bcma\b",                         
        r"\bfrm\b",                        
        r"\bcs\s+(?:final|qualified|inter)\b",  
        r"\bcompany\s+secretary\b",
        r"\bfca\b",                         
        r"\bllm\b", r"master\s+of\s+laws",
        r"\bmba\b", r"m\.?\s*b\.?\s*a",
        r"\bm\.?\s*tech\b", r"\bm\s*tech\b",
        r"\bm\.?\s*sc\b", r"\bmsc\b",
        r"\bm\.?\s*com\b",
        r"\bm\.?\s*a\.?\b",
        r"\bmca\b",
        r"\bpgdm\b", r"\bpgdba\b", r"\bpgpm\b",
        r"post\s*graduate",  r"pg\s+program",
        r"\bmaster\b",
        r"\bm\.?\s*phil\b",
    ]

    bachelor_patterns = [
        r"\bb\.?\s*tech\b", r"\bb\s*tech\b",
        r"\bb\.?\s*e\.?\b",
        r"\bllb\b", r"bachelor\s+of\s+laws",
        r"\bmbbs\b",                      
        r"\bb\.?\s*pharm\b", r"\bbpharm\b",
        r"\bb\.?\s*sc\b", r"\bbsc\b",
        r"\bb\.?\s*com\b", r"\bbcom\b",
        r"\bb\.?\s*b\.?\s*a\b", r"\bbba\b",
        r"\bb\.?\s*a\.?\b",
        r"\bbca\b",
        r"\bbds\b",                    
        r"\bbnys\b",
        r"\bbam\b",
        r"bachelor", r"undergraduate",
        r"ug\s+program", r"\bgraduat(?:ion|ed)\b",
    ]

    diploma_patterns = [
        r"\bdiploma\b", r"\bpoly(?:technic)?\b",
        r"\bitc\b", r"\biti\b",
    ]

    for p in phd_patterns:
        if re.search(p, t_l): return "PhD"
    for p in professional_patterns:
        if re.search(p, t_l): return "Masters"
    for p in bachelor_patterns:
        if re.search(p, t_l): return "Bachelors"
    for p in diploma_patterns:
        if re.search(p, t_l): return "Diploma"
    return "Other"


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


def recency_score(text):
    current_year = datetime.today().year
    text = text.lower()
    matches = re.findall(
        r"(intern|internship|experience|project|work|employed|role|position|"
        r"data|ml|ai|analyst|manager|consultant|engineer|developer|executive|"
        r"associate|officer|director|partner|advocate|auditor|accountant|doctor)"
        r"[\s\S]{0,40}?(20\d{2})",
        text, flags=re.I
    )
    if matches:
        years = [int(y[1]) for y in matches]
        latest = max(years)
    else:
        years = [int(y) for y in re.findall(r"20\d{2}", text) if int(y) > 2010]
        if not years:
            return 0.6
        latest = max(years)
    gap = current_year - latest
    if gap <= 0:   return 1.0
    elif gap == 1: return 0.9
    elif gap == 2: return 0.75
    elif gap <= 4: return 0.6
    return 0.45
# ── Domain Detection ───────────────────────────────────────────────────────────
# Signal weights:
#   3 = role-title / qualification level — near-impossible to misfire
#   2 = strong domain indicator — may appear in 1-2 adjacent domains
#   1 = supporting signal — generic, used only to break ties
#
# Whole-word regex matching used in detect_domain() to prevent partial hits.
DOMAIN_SIGNALS = {

    "web_development": [
        ("web developer", 3), ("web development", 3), ("frontend developer", 3),
        ("front end developer", 3), ("full stack developer", 3), ("fullstack developer", 3),
        ("full-stack developer", 3), ("backend developer", 3), ("back end developer", 3),
        ("ui developer", 3), ("javascript developer", 3), ("react developer", 3),
        ("html", 2), ("css", 2), ("javascript", 2), ("typescript", 2),
        ("react", 2), ("angular", 2), ("vue", 2), ("next.js", 2), ("node.js", 2),
        ("jquery", 2), ("bootstrap", 2), ("tailwind", 2),
        ("responsive design", 2), ("web application", 2), ("website", 2),
        ("dom", 1), ("webpack", 1), ("sass", 1), ("php", 1),
        ("laravel", 1), ("wordpress", 1), ("shopify", 1),
    ],

    "data_science": [
        ("data scientist", 3), ("machine learning engineer", 3), ("ml engineer", 3),
        ("deep learning", 3), ("neural network", 3), ("natural language processing", 3),
        ("computer vision", 3), ("mlops", 3), ("model deployment", 3),
        ("machine learning", 2), ("scikit-learn", 2), ("tensorflow", 2),
        ("pytorch", 2), ("keras", 2), ("xgboost", 2), ("lightgbm", 2),
        ("random forest", 2), ("nlp", 2), ("llm", 2),
        ("feature engineering", 2), ("huggingface", 2), ("pandas", 2),
        ("numpy", 2), ("eda", 2),
        ("data science", 1), ("prediction", 1), ("classification", 1),
        ("regression", 1), ("clustering", 1),
    ],

    "data_analytics": [
        ("data analyst", 3), ("analytics engineer", 3),
        ("bi analyst", 3), ("business intelligence", 3), ("reporting analyst", 3),
        ("power bi", 2), ("tableau", 2), ("looker", 2), ("google analytics", 2),
        ("sql", 2), ("excel", 2), ("dashboard", 2), ("reporting", 2),
        ("data visualization", 2), ("data analysis", 2),
        ("kpi", 2), ("metrics", 2), ("pivot table", 2),
        ("insights", 1), ("mixpanel", 1), ("snowflake", 1), ("bigquery", 1),
    ],

    "data_engineering": [
        ("data engineer", 3), ("etl developer", 3), ("data pipeline", 3),
        ("data warehousing", 3), ("data platform", 3),
        ("apache spark", 2), ("hadoop", 2), ("kafka", 2), ("airflow", 2),
        ("dbt", 2), ("snowflake", 2), ("databricks", 2), ("bigquery", 2),
        ("etl", 2), ("data warehouse", 2), ("data lake", 2),
        ("pyspark", 1), ("hive", 1), ("batch processing", 1), ("streaming", 1),
    ],

    "devops": [
        ("devops engineer", 3), ("site reliability", 3), ("sre", 3),
        ("platform engineer", 3), ("infrastructure engineer", 3), ("cloud engineer", 3),
        ("kubernetes", 2), ("terraform", 2), ("ansible", 2), ("ci/cd", 2),
        ("docker", 2), ("jenkins", 2), ("github actions", 2),
        ("aws", 2), ("azure", 2), ("gcp", 2), ("cloud infrastructure", 2),
        ("prometheus", 2), ("grafana", 2),
        ("linux", 1), ("bash", 1), ("container", 1), ("serverless", 1),
    ],

    "engineering": [
        ("software engineer", 3), ("software developer", 3), ("backend engineer", 3),
        ("systems engineer", 3), ("system design", 3), ("software development", 3),
        ("rest api", 2), ("microservices", 2), ("distributed systems", 2),
        ("design patterns", 2), ("java", 2), ("golang", 2), ("scala", 2),
        ("spring boot", 2), ("django", 2), ("flask", 2), ("fastapi", 2),
        ("postgresql", 2), ("redis", 2),
        ("api", 1), ("backend", 1), ("oops", 1), ("dsa", 1),
    ],

    "mobile_development": [
        ("mobile developer", 3), ("ios developer", 3), ("android developer", 3),
        ("react native developer", 3), ("flutter developer", 3),
        ("swift", 2), ("kotlin", 2), ("react native", 2), ("flutter", 2),
        ("xcode", 2), ("android studio", 2), ("app store", 2), ("play store", 2),
        ("mobile app", 1), ("ios", 1), ("android", 1), ("cross platform", 1),
    ],

    "finance": [
        ("financial analyst", 3), ("investment banker", 3), ("equity analyst", 3),
        ("financial modeling", 3), ("portfolio manager", 3),
        ("cfa", 3), ("investment analysis", 3), ("capital markets", 3),
        ("equity research", 2), ("dcf", 2), ("mergers and acquisitions", 2),
        ("bloomberg", 2), ("corporate finance", 2), ("financial model", 2),
        ("derivatives", 2), ("fixed income", 2), ("fund management", 2),
        ("valuation", 2), ("private equity", 2), ("venture capital", 2),
        ("treasury management", 2), ("forex management", 2),
        ("portfolio", 1), ("investment", 1), ("risk analysis", 1),
    ],

    "accounting": [
        ("chartered accountant", 3), ("ca final", 3), ("ca qualified", 3),
        ("icai", 3), ("cpa", 3), ("acca", 3), ("cma", 3),
        ("statutory audit", 3), ("internal audit", 3), ("tax audit", 3),
        ("gst", 2), ("tds", 2), ("income tax", 2), ("transfer pricing", 2),
        ("ifrs", 2), ("ind as", 2), ("us gaap", 2),
        ("financial statements", 2), ("balance sheet", 2),
        ("accounts payable", 2), ("accounts receivable", 2),
        ("general ledger", 2), ("bank reconciliation", 2),
        ("tally", 2), ("sap fico", 2), ("cost accounting", 2),
        ("management accounting", 2), ("variance analysis", 2),
        ("bookkeeping", 1), ("accounting", 1), ("tax compliance", 1),
        ("audit", 1), ("journal entries", 1),
    ],

    "legal": [
        ("advocate", 3), ("solicitor", 3), ("llb", 3), ("llm", 3),
        ("legal counsel", 3), ("company secretary", 3),
        ("litigation", 3), ("arbitration", 3),
        ("contract drafting", 2), ("contract review", 2),
        ("legal research", 2), ("intellectual property", 2),
        ("patent filing", 2), ("trademark registration", 2),
        ("regulatory compliance", 2), ("employment law", 2),
        ("company law", 2), ("gdpr", 2), ("due diligence legal", 2),
        ("mediation", 2), ("court appearances", 2),
        ("legal writing", 1), ("legal notices", 1),
        ("affidavits", 1), ("pleadings", 1), ("mou drafting", 1),
    ],

    "business": [
        ("business development", 3), ("strategy consultant", 3),
        ("management consultant", 3), ("operations manager", 3),
        ("business operations", 3), ("chief of staff", 3),
        ("consulting", 2), ("strategic planning", 2), ("process improvement", 2),
        ("change management", 2), ("p&l management", 2),
        ("stakeholder management", 2), ("revenue growth", 2),
        ("go to market", 2), ("business transformation", 2),
        ("competitive analysis", 2), ("market entry", 2),
        ("ebitda", 2), ("business case", 2),
        ("strategy", 1), ("operations", 1), ("market research", 1),
    ],

    "consulting": [
        ("management consulting", 3), ("strategy consulting", 3),
        ("mckinsey", 3), ("bcg", 3), ("bain", 3), ("deloitte", 3),
        ("pwc", 3), ("kpmg", 3), ("ey", 3), ("accenture", 3),
        ("engagement manager", 3), ("associate consultant", 3),
        ("senior consultant", 3), ("principal consultant", 3),
        ("digital transformation", 2), ("business process reengineering", 2),
        ("programme management", 2), ("mece framework", 2),
        ("hypothesis driven", 2), ("slide writing", 2),
        ("client management consulting", 2), ("thought leadership", 2),
        ("due diligence", 1), ("feasibility study", 1), ("benchmarking", 1),
    ],

    "supply_chain": [
        ("supply chain manager", 3), ("logistics manager", 3),
        ("procurement manager", 3), ("operations manager", 3),
        ("supply chain management", 3),
        ("inventory management", 2), ("warehouse management", 2),
        ("demand planning", 2), ("vendor management", 2),
        ("procurement", 2), ("import export", 2), ("3pl management", 2),
        ("lean manufacturing", 2), ("six sigma operations", 2),
        ("production planning", 2), ("quality control", 2),
        ("sap scm", 2), ("erp", 2),
        ("logistics", 1), ("sourcing", 1), ("freight", 1),
        ("distribution", 1), ("fleet management", 1),
    ],

    "marketing": [
        ("digital marketing", 3), ("growth marketer", 3), ("seo specialist", 3),
        ("marketing manager", 3), ("performance marketing", 3), ("content marketer", 3),
        ("brand manager", 3), ("media planner", 3),
        ("seo", 2), ("sem", 2), ("google ads", 2), ("facebook ads", 2),
        ("social media marketing", 2), ("email marketing", 2), ("content marketing", 2),
        ("brand management", 2), ("campaign management", 2), ("crm", 2),
        ("conversion optimization", 2), ("growth hacking", 2),
        ("pr management", 2), ("media buying", 2), ("programmatic advertising", 2),
        ("marketing", 1), ("brand", 1), ("customer acquisition", 1),
    ],

    "hr": [
        ("hr manager", 3), ("talent acquisition", 3), ("human resources", 3),
        ("people operations", 3), ("hr business partner", 3), ("hrbp", 3),
        ("hr director", 3), ("chief people officer", 3),
        ("recruitment", 2), ("sourcing", 2), ("onboarding", 2), ("hris", 2),
        ("employee engagement", 2), ("performance management", 2),
        ("workforce planning", 2), ("compensation", 2),
        ("learning and development", 2), ("payroll processing", 2),
        ("employer branding", 2), ("diversity and inclusion", 2),
        ("talent", 1), ("hiring", 1), ("hr compliance", 1),
    ],

    "sales": [
        ("sales manager", 3), ("account executive", 3),
        ("business development representative", 3), ("sales representative", 3),
        ("enterprise sales", 3), ("saas sales", 3), ("regional sales manager", 3),
        ("b2b sales", 2), ("b2c sales", 2), ("lead generation", 2),
        ("cold calling", 2), ("account management", 2), ("quota", 2),
        ("pipeline management", 2), ("revenue target", 2),
        ("channel sales", 2), ("key account management", 2),
        ("sales", 1), ("client relationship", 1), ("negotiation", 1),
    ],

    "product": [
        ("product manager", 3), ("product management", 3), ("product owner", 3),
        ("product strategy", 3), ("product roadmap", 3), ("vp of product", 3),
        ("roadmap", 2), ("user research", 2), ("wireframe", 2), ("figma", 2),
        ("user story", 2), ("sprint", 2), ("backlog", 2), ("ux", 2),
        ("go to market", 2), ("prioritization", 2), ("product analytics", 2),
        ("mvp", 2), ("feature prioritization", 2), ("prd", 2),
        ("stakeholder", 1), ("okr", 1), ("agile", 1),
    ],

    "healthcare": [
        ("doctor", 3), ("physician", 3), ("mbbs", 3), ("md", 3),
        ("nurse", 3), ("clinical research", 3), ("clinical trials", 3),
        ("pharmacovigilance", 3), ("medical writing", 3),
        ("regulatory affairs", 3), ("hospital administrator", 3),
        ("gcp", 2), ("fda regulations", 2), ("drug safety", 2),
        ("medical devices", 2), ("ehr systems", 2), ("medical coding", 2),
        ("icd 10", 2), ("health informatics", 2), ("public health", 2),
        ("epidemiology", 2), ("biostatistics", 2), ("pharmacy", 2),
        ("patient care", 1), ("clinical", 1), ("healthcare", 1),
    ],

    "education": [
        ("teacher", 3), ("professor", 3), ("lecturer", 3),
        ("curriculum developer", 3), ("instructional designer", 3),
        ("academic", 3), ("principal", 3), ("dean", 3),
        ("curriculum development", 2), ("instructional design", 2),
        ("lesson planning", 2), ("e-learning", 2), ("lms", 2),
        ("teaching", 2), ("training delivery", 2), ("pedagogy", 2),
        ("assessment design", 2), ("edtech", 2),
        ("coaching", 1), ("mentoring", 1), ("facilitation", 1),
    ],
}


def detect_domain(text):
    """
    Weighted domain detection with whole-word regex matching.
    Returns domain with highest cumulative weighted score.
    Falls back to 'general' if no signals fire.
    """
    t = text.lower()
    scores = {}

    for domain, signals in DOMAIN_SIGNALS.items():
        total = 0
        for keyword, weight in signals:
            pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
            if re.search(pattern, t):
                total += weight
        scores[domain] = total

    if max(scores.values()) == 0:
        return "general"

    return max(scores, key=scores.get)


EXP_CAPS = {
    "web_development":    2,
    "data_science":       2,
    "data_analytics":     3,
    "data_engineering":   3,
    "devops":             4,
    "engineering":        2,
    "mobile_development": 2,
    "finance":            8,
    "accounting":         6,
    "legal":              7,
    "business":           10,
    "consulting":         8,
    "supply_chain":       8,
    "marketing":          6,
    "hr":                 8,
    "sales":              6,
    "product":            7,
    "healthcare":         8,
    "education":          10,
    "general":            10,
}

def normalize_experience(years, domain="general"):
    cap = EXP_CAPS.get(domain, 10)
    return min(years / cap, 1.0)


from core.skill_extractor import build_skill_index, extract_skills_whitelist

def extract_profile(t, skills):
    t = clean_text(t)
    yrs, months  = extract_years_of_experience(t)
    edu          = extract_education_level(t)
    email, phone = extract_contacts(t)
    cgpa         = extract_cgpa(t)
    rec          = recency_score(t)
    domain       = detect_domain(t)

    skill_idx    = build_skill_index(skills)
    skills_found = extract_skills_whitelist(t, skill_idx, n_max=4, fuzzy=False)

    return pd.Series({
        "clean_text":         t,
        "years_experience":   yrs,
        "months_experience":  months,
        "education":          edu,
        "email":              email,
        "phone":              phone,
        "skills_found":       skills_found,
        "recency":            rec,
        "cgpa":               cgpa,
        "total_skills_found": len(skills_found),
        "domain":             domain,
    })
