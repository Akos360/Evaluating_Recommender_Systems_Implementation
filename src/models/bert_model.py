from .base_model import BaseRecommender
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from torch.nn.functional import cosine_similarity as torch_cosine
from tqdm import tqdm

class BERTRecommender(BaseRecommender):
    def __init__(self, rec_type, model_name="all-MiniLM-L6-v2"):
        super().__init__(rec_type)
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to(self.device)

    def use_batch_similarity(self):
        return False

    def train(self, data):
        # No training needed for BERT (pretrained model)
        pass

    # def load_model(self):
    #     pass

    def prepare_input_and_filtered(self, data, book_idx, para_idx, exclude=True):
        # filter descriptions
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
        
        # filter paragraphs
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
        return self.model.encode(self.input_text, convert_to_tensor=True).to(self.device)

    # def get_doc_vectors(self):
    #     col = "text" if self.rec_type == "paragraph" else "description"
    #     return [self.model.encode(text, convert_to_tensor=True).to(self.device) for text in self.filtered_data[col].tolist()]

    def get_doc_vectors(self):
        col = "text" if self.rec_type == "paragraph" else "description"
        texts = self.filtered_data[col].tolist()
        
        print("Generating BERT embeddings for all documents...")
        with tqdm(total=len(texts), desc="Encoding Texts (BERT)", unit="doc") as pbar:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=32
            ).to(self.device)
            pbar.update(len(texts))
        
        return embeddings

    # def compute_similarity(self, input_vec, doc_vec):
    #     return cosine_similarity(input_vec, doc_vec)[0, 0]

    def compute_similarity(self, input_vec, doc_vec):
        sim = torch_cosine(input_vec, doc_vec, dim=0)
        return sim.item()
        
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
