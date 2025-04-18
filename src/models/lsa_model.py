from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseRecommender
import joblib
import numpy as np
import os


class LSARecommender(BaseRecommender):
    def __init__(self, rec_type):
        super().__init__(rec_type)
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.lsa = TruncatedSVD(n_components=500, random_state=42)
        self.lsa_matrix = None
        self.model_path = "saved_models/lsa.pkl"

    def train(self, data):
        if self.rec_type == "paragraph":
            tfidf_matrix = self.vectorizer.fit_transform(data["text"])
        else:
            tfidf_matrix = self.vectorizer.fit_transform(data["description"])

        self.lsa_matrix = self.lsa.fit_transform(tfidf_matrix)

        os.makedirs("saved_models", exist_ok=True)
        joblib.dump((self.vectorizer, self.lsa, self.lsa_matrix), self.model_path)

    def load_model(self):
        if self.lsa_matrix is None:
            self.vectorizer, self.lsa, self.lsa_matrix = joblib.load(self.model_path)

    def prepare_input_and_filtered(self, data, book_idx, para_idx, exclude=True):
        self.load_model()

        if para_idx is None:
            book = data[data["book_index"] == book_idx]
            if book.empty:
                raise ValueError("Book not found.")
            self.input_text = book.iloc[0]["description"]
            self.input_data = {
                "book_index": book.iloc[0]["book_index"],
                "book_title": book.iloc[0]["title"],
                "description": self.input_text,
            }
            self.filtered_data = data if not exclude else data[data["book_index"] != book_idx]
            self.filtered_data = self.filtered_data.drop_duplicates(subset=["description"]).reset_index(drop=True)
        else:
            para = data[(data["book_index"] == book_idx) & (data["paragraph_index"] == para_idx)]
            if para.empty:
                raise ValueError("Paragraph not found.")
            self.input_text = para.iloc[0]["text"]
            self.input_data = {
                "book_index": para.iloc[0]["book_index"],
                "paragraph_index": para.iloc[0]["paragraph_index"],
                "book_title": para.iloc[0]["book_title"],
                "text": self.input_text,
            }
            self.filtered_data = data if not exclude else data[data["book_index"] != book_idx]
            self.filtered_data = self.filtered_data.drop_duplicates(subset=["text"]).reset_index(drop=True)

    def get_input_vector(self):
        tfidf = self.vectorizer.transform([self.input_text])
        return self.lsa.transform(tfidf)

    def get_doc_vectors(self):
        if self.rec_type == "paragraph":
            tfidf_matrix = self.vectorizer.transform(self.filtered_data["text"])
        else:
            tfidf_matrix = self.vectorizer.transform(self.filtered_data["description"])
        return self.lsa.transform(tfidf_matrix)

    def compute_similarity(self, input_vector, doc_vector):
        return cosine_similarity(input_vector, doc_vector.reshape(1, -1))[0][0]

    def format_recommendation(self, idx, score):
        row = self.filtered_data.iloc[idx]
        if self.rec_type == "paragraph":
            return {
                "book_index": row["book_index"],
                "paragraph_index": row["paragraph_index"],
                "similarity_score": round(score, 3),
                "recommended_book": row["book_title"],
                "recommended_text": row["text"],
            }
        else:
            return {
                "book_index": row["book_index"],
                "book_title": row["title"],
                "similarity_score": round(score, 3),
                "recommended_description": row["description"],
            }
