
from langchain_community.embeddings import OllamaEmbeddings
from config import EMBED_MODEL


def get_embedding_model():
    return OllamaEmbeddings(model=EMBED_MODEL)


def embed_query(query):
    return get_embedding_model().embed_query(query)
