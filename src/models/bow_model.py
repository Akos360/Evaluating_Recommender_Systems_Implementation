from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseRecommender
import joblib
import numpy as np
import os


class BoWRecommender(BaseRecommender):
    def __init__(self, rec_type):
        super().__init__(rec_type)
        self.vectorizer = CountVectorizer(stop_words="english")
        self.bow_matrix = None
        # self.model_path = "saved_models/bow.pkl"

    def use_batch_similarity(self):
        return True
        
    def train(self, data):
        if self.rec_type == "paragraph":
            self.bow_matrix = self.vectorizer.fit_transform(data["text"])
        else:
            self.bow_matrix = self.vectorizer.fit_transform(data["description"])

        # os.makedirs("saved_models", exist_ok=True)
        # joblib.dump((self.vectorizer, self.bow_matrix), self.model_path)

    # def load_model(self):
    #     if self.bow_matrix is None:
    #         self.vectorizer, self.bow_matrix = joblib.load(self.model_path)

    def prepare_input_and_filtered(self, data, book_idx, para_idx, exclude=True):
        # self.load_model()

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

            if exclude:
                self.filtered_data = data[data["book_index"] != book_idx]
            else:
                self.filtered_data = data

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

            if exclude:
                self.filtered_data = data[
                    ~((data["book_index"] == book_idx) & (data["paragraph_index"] == para_idx))
                ]
            else:
                self.filtered_data = data

    def get_input_vector(self):
        return self.vectorizer.transform([self.input_text])

    def get_doc_vectors(self):
        # We already have the matrix for the full data
        return self.bow_matrix

    def compute_similarity(self, input_vec, doc_vec):
        return cosine_similarity(input_vec, doc_vec)[0, 0]

    def compute_all_similarities(self, input_vec, doc_matrix):
        return cosine_similarity(input_vec, doc_matrix).flatten()

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
