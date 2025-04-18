from .base_model import BaseRecommender
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class BERTRecommender(BaseRecommender):
    def __init__(self, rec_type, model_name="all-MiniLM-L6-v2"):
        super().__init__(rec_type)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.to("cuda")

    def train(self, data):
        # No training needed for BERT (pretrained model)
        pass

    def load_model(self):
        # Already initialized in __init__
        pass

    def prepare_input_and_filtered(self, data, book_idx, para_idx, exclude=True):
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
        return self.model.encode(self.input_text, convert_to_tensor=True).cpu().numpy()

    def get_doc_vectors(self):
        col = "text" if self.rec_type == "paragraph" else "description"
        return self.model.encode(self.filtered_data[col].tolist(), convert_to_tensor=True).cpu().numpy()

    def compute_similarity(self, input_vector, doc_vector):
        if np.linalg.norm(input_vector) == 0 or np.linalg.norm(doc_vector) == 0:
            return 0.0
        return float(np.dot(input_vector, doc_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(doc_vector)))

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
