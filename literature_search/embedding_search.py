import re
from typing import Any, Dict, List
import faiss
import numpy as np
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class EmbeddingSearch:
    def __init__(self, logger):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.index = None
        self.embeddings = None
        self.texts = []
        self.metadata_list = []
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.logger = logger

    def create_embeddings(self, texts: List[str], titles: List[str] = None) -> np.ndarray:
        combined_texts = []
        for idx, text in enumerate(texts):
            title = titles[idx] if titles and idx < len(titles) else ''
            combined = f"{title}\n\n{text}" if title else text
            cleaned_text = self.clean_text(combined)
            combined_texts.append(cleaned_text)
        embeddings = self.model.encode(
            combined_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=64
        )
        return embeddings

    def clean_text(self, text):
        # Remove references like [1], (2020)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d{4}\)', '', text)
        # Remove non-textual content
        text = re.sub(r'\bFig\b.*', '', text)
        text = re.sub(r'\bTable\b.*', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Lowercase
        text = text.lower()
        # Tokenize
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        # Reconstruct text
        cleaned_text = ' '.join(tokens)
        return cleaned_text

    def build_faiss_index(self, embeddings: np.ndarray, texts: List[str], metadata_list: List[Dict[str, Any]]):
        self.embeddings = embeddings.astype('float32')
        self.texts = texts
        self.metadata_list = metadata_list
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        self.logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def find_top_similar_papers(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        cleaned_query = self.clean_text(query_text)
        query_embedding = self.model.encode([cleaned_query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)
        similar_papers = []
        for idx in indices[0]:
            if idx == -1:
                continue
            similar_papers.append(self.metadata_list[idx])
        return similar_papers
