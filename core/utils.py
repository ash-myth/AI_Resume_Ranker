import io, re, pandas as pd
from PyPDF2 import PdfReader
from docx import Document
def read_pdf(file):
    text = ""
    try:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception:
        pass
    return text
def read_docx(file):
    try:
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""
def read_txt(file):
    try:
        return file.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
def extract_texts(files):
    out = {}
    for i,f in enumerate(files):
        name = getattr(f, "name", f"resume_{i}")
        ext = name.split(".")[-1].lower()
        if ext=="pdf":
            text = read_pdf(f)
        elif ext in ["docx","doc"]:
            text = read_docx(f)
        else:
            text = read_txt(f)
        out[name] = text
    return out
def load_skills(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [s.strip() for s in fh.readlines() if s.strip()]
def to_table_download(df):
    return df.to_csv(index=False).encode("utf-8")