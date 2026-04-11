import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model     = None
        self.tfidf     = None
        self._fitted   = False
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None

    def encode(self, texts):
        if self.model:
            return self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        # TF-IDF fallback — fit once on the full corpus (resumes + JD together)
        if not self._fitted:
            self.tfidf   = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            self.tfidf.fit(texts)
            self._fitted = True
        return self.tfidf.transform(texts).toarray()

    def similarity(self, a, b):
        return cosine_similarity(a, b)
