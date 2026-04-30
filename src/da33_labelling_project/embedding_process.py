#%% LIB
import faiss
import numpy as np
import os

from .utils import get_openai_client

#%% CONFIG
client = get_openai_client()

#%% MAIN FUNCTIONS
#%%% Embedding
def standardize_embedding(embeddings):
    standard_embeddings = np.array(embeddings, dtype='float32')
    standard_embeddings = np.ascontiguousarray(standard_embeddings)
    return standard_embeddings


def embedding_text(texts: list, batch_size: int):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        response = client.embeddings.create(
            input=batch, 
            model="text-embedding-3-small"
            )
        batch_embedding = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embedding)
    return standardize_embedding(all_embeddings)

def recontruct_embeddings(faiss_index):
    return faiss_index.reconstruct_n(0, faiss_index.ntotal)

#%%% Embedding store
def build_vector_store(embeddings):
    embeddings = standardize_embedding(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index
    
    
def search_similar_embeddings(query_text, index, k):
    query_embedding = embedding_text([query_text], 1)
    query_embedding = standardize_embedding(query_embedding)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k)
    return D, I


def search_similar_knowledge_text(I, knowledge_texts):
    return [
        knowledge_texts[i] for i in I[0]
        ]

def save_embedding_store(faiss_index, path):
    faiss_path = os.fspath(path)
    faiss.write_index(faiss_index, faiss_path)
    return {
        'status': 'success',
        'message': 'save knowledge texts to embedding store'
        }

def load_embedding_store(path):
    index = faiss.read_index(
        os.fspath(path)
        )
    return index

