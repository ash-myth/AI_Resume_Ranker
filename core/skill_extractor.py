import re
from difflib import get_close_matches

def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\+\.\- ]", " ", s.lower())).strip()

SYNONYMS = {
    "power bi":                  ["powerbi", "ms power bi", "microsoft power bi"],
    "scikit-learn":              ["sklearn", "scikit learn"],
    "pytorch":                   ["py torch"],
    "tensorflow":                ["tf"],
    "mysql":                     ["my sql"],
    "postgresql":                ["postgres", "postgre sql"],
    "huggingface":               ["hugging face"],
    "computer vision":           ["cv"],
    "rest api":                  ["restful api", "rest apis", "restful"],
    "fastapi":                   ["fast api"],
    "docker":                    ["containerization"],
    "kubernetes":                ["k8s"],
    "mlops":                     ["ml ops"],
    "etl":                       ["extract transform load"],
    "llm":                       ["large language model", "large language models"],
    "rag":                       ["retrieval augmented generation"],
    "eda":                       ["exploratory data analysis", "exploratory analysis"],
    "nlp":                       ["natural language processing"],
    "aws":                       ["amazon web services"],
    "gcp":                       ["google cloud", "google cloud platform"],
    "node.js":                   ["nodejs", "node js"],
    "next.js":                   ["nextjs", "next js"],
    "ci/cd":                     ["ci cd", "continuous integration", "continuous deployment"],
    "beautifulsoup":             ["beautiful soup", "bs4"],
    "git":                       ["version control", "github"],
    "dsa":                       ["data structures and algorithms"],
    "named entity recognition":  ["ner"],
    "bert":                      ["bert model"],
    "streamlit":                 ["streamlit app", "streamlit dashboard"],
    "a/b testing":               ["ab testing", "split testing", "a b testing"],
    "apache spark":              ["pyspark", "spark"],

    "financial modeling":        ["financial model", "financial models", "fin modeling", "financial modelling"],
    "mergers and acquisitions":  ["m&a", "m & a", "merger and acquisition"],
    "dcf":                       ["discounted cash flow", "discounted cashflow"],
    "p&l management":            ["profit and loss", "p and l", "pnl", "p&l"],
    "variance analysis":         ["budget vs actual", "budget variance"],
    "equity research":           ["equity analyst", "equity analysis"],
    "financial statements":      ["financial statement", "financial reporting"],
    "balance sheet":             ["bs analysis"],
    "working capital management":["working capital"],
    "accounts payable":          ["ap management", "creditors"],
    "accounts receivable":       ["ar management", "debtors"],
    "general ledger":            ["gl accounting"],
    "bank reconciliation":       ["brs", "bank recon"],
    "tally":                     ["tally erp", "tally prime", "tally erp9"],
    "sap fico":                  ["sap fi", "sap co", "sap finance"],
    "gst":                       ["goods and services tax", "gst filing", "gst returns"],
    "tds":                       ["tax deducted at source", "tds returns"],
    "transfer pricing":          ["tp study", "tp documentation"],
    "income tax":                ["income tax returns", "itr filing", "itr"],
    "statutory audit":           ["stat audit", "statutory auditing"],
    "internal audit":            ["internal auditing", "IA"],
    "ifrs":                      ["international financial reporting standards"],
    "ind as":                    ["indian accounting standards", "ind-as"],
    "us gaap":                   ["gaap", "generally accepted accounting principles"],
    "cost accounting":           ["cost accountant", "costing"],
    "management accounting":     ["management accountant", "managerial accounting"],
    "chartered accountant":      ["ca inter", "ca final", "icai member", "ca qualified"],
    "cfa":                       ["chartered financial analyst", "cfa level"],
    "acca":                      ["association of chartered certified accountants"],
    "cpa":                       ["certified public accountant"],
    "cma":                       ["cost and management accountant", "icma"],
    "treasury management":       ["treasury operations", "cash management"],
    "forex management":          ["foreign exchange", "fx management", "currency risk"],
    "fixed assets":              ["fixed asset management", "asset register", "capex tracking"],
    "standard costing":          ["standard cost", "cost standards"],

    "contract drafting":         ["drafting contracts", "contract preparation", "agreement drafting"],
    "contract review":           ["reviewing contracts", "contract analysis"],
    "legal research":            ["case law research", "legal analysis"],
    "litigation":                ["court proceedings", "trial work", "legal proceedings"],
    "intellectual property":     ["ip law", "ip rights"],
    "patent filing":             ["patent prosecution", "patent drafting"],
    "trademark registration":    ["trademark filing", "tm registration"],
    "regulatory compliance":     ["compliance management", "regulatory affairs"],
    "gdpr":                      ["general data protection regulation", "data protection"],
    "employment law":            ["labour law", "labor law", "employment legislation"],
    "company law":               ["companies act", "corporate law"],
    "due diligence legal":       ["legal due diligence", "legal dd"],
    "mergers and acquisitions law": ["m&a legal", "transaction advisory legal"],
    "llb":                       ["bachelor of laws", "law degree"],
    "llm":                       ["master of laws", "masters in law"],
    "manupatra":                 ["manupatra legal database"],
    "westlaw":                   ["westlaw legal research"],

    "go to market":              ["gtm", "go-to-market", "gtm strategy"],
    "salesforce":                ["sfdc", "salesforce crm"],
    "google analytics":          ["ga4", "google analytics 4"],
    "google ads":                ["google adwords", "adwords", "google ppc"],
    "facebook ads":              ["meta ads", "fb ads", "instagram ads"],
    "sem":                       ["search engine marketing", "paid search"],
    "seo":                       ["search engine optimization", "search engine optimisation"],
    "crm":                       ["customer relationship management"],
    "marketing automation":      ["marketing ops", "marketing operations"],
    "adobe creative suite":      ["photoshop", "illustrator", "adobe ps", "indesign"],
    "programmatic advertising":  ["programmatic", "dsp", "demand side platform"],
    "pr management":             ["public relations", "media relations"],
    "atl marketing":             ["above the line", "mass media marketing"],
    "btl marketing":             ["below the line", "activation marketing"],

    "hris":                      ["human resource information system", "hrms", "hr system"],
    "learning and development":  ["l&d", "l and d", "training and development"],
    "talent acquisition":        ["ta", "recruiting", "talent sourcing"],
    "payroll processing":        ["payroll management", "salary processing"],
    "hr business partner":       ["hrbp", "hr bp"],
    "diversity and inclusion":   ["d&i", "dei", "diversity equity inclusion"],
    "employer branding":         ["recruitment marketing", "evp"],
    "succession planning":       ["leadership pipeline"],
    "campus recruitment":        ["campus hiring", "college hiring"],

    "supply chain management":   ["scm", "supply chain", "end to end supply chain"],
    "inventory management":      ["stock management", "inventory control"],
    "warehouse management":      ["warehouse operations", "wms"],
    "demand planning":           ["demand forecasting", "s&op"],
    "vendor management":         ["supplier management", "vendor relations"],
    "procurement":               ["purchasing", "strategic sourcing", "buying"],
    "import export":             ["import export management", "exim", "trade documentation"],
    "3pl management":            ["third party logistics", "3pl"],
    "lean manufacturing":        ["lean production", "lean ops"],
    "six sigma operations":      ["six sigma", "lean six sigma", "6 sigma"],
    "total productive maintenance": ["tpm"],
    "production planning":       ["production scheduling", "production control"],
    "quality control":           ["qc", "quality check", "quality inspection"],
    "quality assurance":         ["qa", "quality management"],
    "sap scm":                   ["sap supply chain", "sap mm", "sap pp", "sap sd"],

    "management consulting":     ["strategy consulting", "mgmt consulting"],
    "business process reengineering": ["bpr", "process reengineering"],
    "digital transformation":    ["digital strategy", "digitisation", "digitization"],
    "programme management":      ["program management", "programme delivery"],
    "pmo":                       ["project management office"],
    "mece framework":            ["mece", "mutually exclusive collectively exhaustive"],

    "clinical trials":           ["clinical study", "clinical investigation"],
    "gcp":                       ["good clinical practice"],
    "pharmacovigilance":         ["drug safety", "pv", "adverse event reporting"],
    "medical writing":           ["clinical writing", "regulatory writing"],
    "medical coding":            ["clinical coding", "icd coding"],
    "icd 10":                    ["icd-10", "icd 10 coding"],
    "ehr systems":               ["electronic health records", "emr", "epic", "cerner"],
    "fda regulations":           ["fda compliance", "21 cfr"],
    "regulatory affairs":        ["regulatory submissions", "ra"],
    "mbbs":                      ["bachelor of medicine", "medical degree"],

    "okrs":                      ["objectives and key results", "okr"],
    "kpis":                      ["key performance indicators", "kpi"],
    "six sigma":                 ["6 sigma", "lean six sigma"],
    "user experience":           ["ux"],
    "ui design":                 ["user interface design"],
    "ux design":                 ["user experience design"],
    "project management":        ["project mgmt", "project delivery"],
    "agile":                     ["agile methodology", "agile framework"],
    "b2b sales":                 ["business to business", "b-to-b"],
    "b2c sales":                 ["business to consumer", "b-to-c"],
    "ebitda":                    ["earnings before interest tax depreciation", "ebitda analysis"],
    "balanced scorecard":        ["bsc", "strategy scorecard"],
    "microsoft office":          ["ms office", "office 365", "microsoft 365"],

    "instructional design":      ["learning design", "course design", "curriculum design"],
    "e-learning":                ["elearning", "online learning", "digital learning"],
    "lms":                       ["learning management system", "moodle", "canvas lms"],
}

def build_skill_index(skills, synonyms=None):
    merged = {**SYNONYMS, **(synonyms or {})}
    idx = {}
    for s in skills:
        k = _norm(s)
        if k:
            idx[k] = s
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
        if len(g) < 2:
            continue
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
