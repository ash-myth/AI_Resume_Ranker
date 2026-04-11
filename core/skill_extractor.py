import re
from difflib import get_close_matches

def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\+\.\- ]", " ", s.lower())).strip()

# Universal synonyms covering DS, business, finance, marketing, HR, product domains
SYNONYMS = {
    # Tech
    "power bi":             ["powerbi", "ms power bi", "microsoft power bi"],
    "scikit-learn":         ["sklearn", "scikit learn"],
    "pytorch":              ["py torch"],
    "tensorflow":           ["tf"],
    "mysql":                ["my sql"],
    "postgresql":           ["postgres", "postgre sql"],
    "huggingface":          ["hugging face"],
    "computer vision":      ["cv"],
    "rest api":             ["restful api", "rest apis", "restful"],
    "fastapi":              ["fast api"],
    "docker":               ["container", "containers", "containerization"],
    "kubernetes":           ["k8s"],
    "mlops":                ["ml ops"],
    "etl":                  ["extract transform load"],
    "llm":                  ["large language model", "large language models"],
    "rag":                  ["retrieval augmented generation"],
    "eda":                  ["exploratory data analysis", "exploratory analysis"],
    "nlp":                  ["natural language processing"],
    "aws":                  ["amazon web services"],
    "gcp":                  ["google cloud", "google cloud platform"],
    "node.js":              ["nodejs", "node js"],
    "next.js":              ["nextjs", "next js"],
    "ci/cd":                ["ci cd", "continuous integration", "continuous deployment"],
    "beautifulsoup":        ["beautiful soup", "bs4"],
    "git":                  ["version control", "github"],
    "dsa":                  ["data structures", "data structures and algorithms"],
    "named entity recognition": ["ner"],
    "bert":                 ["bert model"],
    "streamlit":            ["streamlit app", "streamlit dashboard"],

    # Business / Finance
    "financial modeling":   ["financial model", "financial models", "fin modeling"],
    "mergers and acquisitions": ["m&a", "m & a"],
    "dcf":                  ["discounted cash flow"],
    "p&l management":       ["profit and loss", "p and l", "pnl"],
    "go to market":         ["gtm", "go-to-market"],
    "salesforce":           ["sfdc"],
    "google analytics":     ["ga4", "google analytics 4"],
    "google ads":           ["google adwords", "adwords"],
    "facebook ads":         ["meta ads", "fb ads"],
    "sem":                  ["search engine marketing", "paid search"],
    "seo":                  ["search engine optimization"],
    "crm":                  ["customer relationship management"],
    "hris":                 ["human resource information system"],
    "learning and development": ["l&d", "l and d"],
    "a/b testing":          ["ab testing", "split testing"],
    "okrs":                 ["objectives and key results"],
    "kpis":                 ["key performance indicators", "key performance indicator"],
    "six sigma":            ["6 sigma", "lean six sigma"],
    "user experience":      ["ux"],
    "ui design":            ["user interface design"],
    "ux design":            ["user experience design"],
    "supply chain":         ["supply chain management", "scm"],
    "project management":   ["pm", "project mgmt"],
    "agile":                ["agile methodology", "agile framework"],
    "variance analysis":    ["budget vs actual"],
    "equity research":      ["equity analyst"],
    "b2b sales":            ["business to business", "b-to-b"],
    "b2c sales":            ["business to consumer", "b-to-c"],
}

def build_skill_index(skills, synonyms=None):
    merged = {**SYNONYMS, **(synonyms or {})}
    idx = {}
    base = set()
    for s in skills:
        k = _norm(s)
        if k:
            idx[k] = s
            base.add(k)
    for canon, alts in merged.items():
        c = _norm(canon)
        if c in idx:
            for a in alts:
                ak = _norm(a)
                if ak:
                    idx[ak] = idx[c]
    return idx

def compute_rarity_scores(df):
    from collections import Counter
    all_sk = []
    for row in df["skills_found"]:
        all_sk.extend([_norm(s) for s in row])
    freq = Counter(all_sk)
    maxf = max(freq.values()) if freq else 1
    return {skill: 1 - (count / maxf) for skill, count in freq.items()}

def _ngrams(tokens, n_max=4):
    L = len(tokens)
    for n in range(n_max, 0, -1):
        for i in range(L - n + 1):
            yield " ".join(tokens[i:i+n])

def extract_skills_whitelist(text, skill_index, n_max=4, fuzzy=False):
    t = _norm(text)
    toks = [x for x in t.split() if x]
    found = []
    seen = set()
    for g in _ngrams(toks, n_max=n_max):
        if g in skill_index and g not in seen:
            seen.add(g)
            found.append(skill_index[g])
    if fuzzy:
        keys = list(skill_index.keys())
        for token in toks:
            if len(token) < 4:
                continue
            matches = get_close_matches(token, keys, n=1, cutoff=0.92)
            if matches:
                k = matches[0]
                if k not in seen:
                    seen.add(k)
                    found.append(skill_index[k])
    return sorted(set(found), key=lambda s: s.lower())

def order_skills_jd_first(found_skills, jd_required_set):
    jd_first = [s for s in found_skills if _norm(s) in jd_required_set]
    rest     = [s for s in found_skills if _norm(s) not in jd_required_set]
    return jd_first + rest
