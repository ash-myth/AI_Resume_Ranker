import re
from difflib import get_close_matches

def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\+\.\- ]", " ", s.lower())).strip()

def build_skill_index(skills, synonyms=None):
    synonyms = {
    "power bi": ["powerbi", "ms power bi"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "pytorch": ["py torch"],
    "tensorflow": ["tf"],
    "mysql": ["my sql"],
    "postgresql": ["postgres", "postgre sql"],
    "huggingface": ["hugging face"],
    "computer vision": ["cv"],
    "rest api": ["restful api", "rest apis"],
    "fastapi": ["fast api"],
    "docker": ["container", "containers"],
    "kubernetes": ["k8s"],
    "mlops": ["ml ops"],
    "etl": ["extract transform load"],
    "llm": ["large language model", "large language models"],
    "rag": ["retrieval augmented generation"],
    "eda": ["exploratory data analysis","data analysis","data cleaning"]
    }

    idx = {}
    base = set()
    for s in skills:
        k = _norm(s)
        if k:
            idx[k] = s
            base.add(k)
    if synonyms:
        for canon, alts in synonyms.items():
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
    rarity = {skill: 1 - (count / maxf) for skill, count in freq.items()}
    return rarity

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
    found = sorted(set(found), key=lambda s: s.lower())
    return found

def order_skills_jd_first(found_skills, jd_required_set):
    jd_first = [s for s in found_skills if _norm(s) in jd_required_set]
    rest = [s for s in found_skills if _norm(s) not in jd_required_set]
    return jd_first + rest
