from sentence_transformers import SentenceTransformer

class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):

        return self.model.encode([query])[0].tolist()

    def embed_documents(self, documents):

        return self.model.encode(documents).tolist()

def embedding_function():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding_function = EmbeddingFunctionWrapper(model)
    return embedding_function
