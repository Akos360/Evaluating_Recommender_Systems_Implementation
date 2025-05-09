class BaseRecommender:
    def __init__(self, rec_type):
        self.rec_type = rec_type
        self.input_text = None
        self.input_data = None
        self.filtered_data = None

    def use_batch_similarity(self):
        return True

    def train(self):
        raise NotImplementedError()

    # def load_model(self):
    #     raise NotImplementedError()

    def prepare_input_and_filtered(self):
        raise NotImplementedError()

    def get_input_vector(self):
        raise NotImplementedError()

    def get_doc_vectors(self):
        raise NotImplementedError()

    def compute_similarity(self):
        raise NotImplementedError()

    def format_recommendation(self):
        raise NotImplementedError()
