from gensim.models import FastText
import numpy as np
from .base_model import BaseRecommender
from sklearn.metrics.pairwise import cosine_similarity
import os


class FastTextRecommender(BaseRecommender):
    def __init__(self, rec_type):
        super().__init__(rec_type)
        self.model = None
        self.vector_size = 100
        # self.model_path = "saved_models/fasttext.model"

    def use_batch_similarity(self):
        return False

    def train(self, data):
        if self.rec_type == "paragraph":
            texts = [text.split() for text in data["text"]]
        else:
            texts = [desc.split() for desc in data["description"]]
        
        self.model = FastText(
            sentences=texts,
            vector_size=self.vector_size,
            window=5,
            min_count=2,
            workers=4,
            epochs=20
        )

        # os.makedirs("saved_models", exist_ok=True)
        # self.model.save(self.model_path)

    # def load_model(self):
    #     if self.model is None:
    #         self.model = FastText.load(self.model_path)

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

    def get_document_vector(self, text):
        tokens = text.split()
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

    def get_input_vector(self):
        return self.get_document_vector(self.input_text)

    def get_doc_vectors(self):
        if self.rec_type == "paragraph":
            return [self.get_document_vector(row["text"]) for _, row in self.filtered_data.iterrows()]
        else:
            return [self.get_document_vector(row["description"]) for _, row in self.filtered_data.iterrows()]

    # def compute_similarity(self, input_vector, doc_vector):
    #     if np.linalg.norm(input_vector) == 0 or np.linalg.norm(doc_vector) == 0:
    #         return 0.0
    #     return float(np.dot(input_vector, doc_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(doc_vector)))

    def compute_similarity(self, input_vec, doc_vec):
        return cosine_similarity(input_vec.reshape(1, -1), doc_vec.reshape(1, -1))[0][0]

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
