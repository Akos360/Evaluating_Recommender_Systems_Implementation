class BaseRecommender:
    def __init__(self, rec_type):
        """
        rec_type: either "paragraph" or "description"
        """
        self.rec_type = rec_type
        self.input_text = None
        self.input_data = None
        self.filtered_data = None

    def use_batch_similarity(self):
        return True

    def train(self, data):
        raise NotImplementedError()

    # def load_model(self):
    #     raise NotImplementedError()

    def prepare_input_and_filtered(self, data, book_idx, para_idx, exclude=True):
        raise NotImplementedError()

    def get_input_vector(self):
        raise NotImplementedError()

    def get_doc_vectors(self):
        raise NotImplementedError()

    def compute_similarity(self, input_vector, doc_vector):
        raise NotImplementedError()

    def format_recommendation(self, idx, score):
        raise NotImplementedError()
