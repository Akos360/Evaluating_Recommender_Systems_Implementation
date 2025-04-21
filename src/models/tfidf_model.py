from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_model import BaseRecommender

class TfidfRecommender(BaseRecommender):
    def __init__(self, rec_type):
        super().__init__(rec_type)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    def use_batch_similarity(self):
        return True

    def train(self, data):
        if self.rec_type == "paragraph":
            self.vectorizer.fit(data['text'])
        else:
            self.vectorizer.fit(data['description'])

    def prepare_input_and_filtered(self, data, book_idx, para_idx, exclude=True):
        if para_idx is None:
            book = data[data['book_index'] == book_idx]
            if book.empty:
                raise ValueError("Book not found.")
            self.input_text = book.iloc[0]["description"]
            self.input_data = {
                "book_index": book.iloc[0]["book_index"],
                "book_title": book.iloc[0]["title"],
                "description": self.input_text,
            }
            self.filtered_data = data if not exclude else data[data['book_index'] != book_idx]
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
        return self.vectorizer.transform([self.input_text])

    def get_doc_vectors(self):
        if self.rec_type == "paragraph":
            return self.vectorizer.transform(self.filtered_data['text'])
        else:
            return self.vectorizer.transform(self.filtered_data['description'])

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
