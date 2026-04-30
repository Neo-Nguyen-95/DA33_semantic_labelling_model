#%% LIB
from da33_labelling_project.config import (
    PROCESSED_DATA_DIR
    )
from da33_labelling_project.utils import (
    get_openai_client,
    load_records,
    save_records
    )
from da33_labelling_project.embedding_process import (
    embedding_text,
    build_vector_store,
    search_similar_embeddings,
    save_embedding_store
    )

#%% CONFIG
client = get_openai_client()

knowledge_data_dir = PROCESSED_DATA_DIR / 'high_school_knowledge'

#%% MAIN
knowledge_data = load_records(knowledge_data_dir)
knowledge_texts = [
    data.get('text') for data in knowledge_data
    ]

embeddings = embedding_text(knowledge_texts, 100)
faiss_index = build_vector_store(embeddings)

faiss_path = (
        PROCESSED_DATA_DIR / 
        'high_school_knowledge_embeddings' / 
        'high_school_knowledge.faiss'
        )
save_embedding_store(faiss_index, faiss_path)
json_path = (
    PROCESSED_DATA_DIR / 
    'high_school_knowledge_embeddings' / 
    'high_school_knowledge.json'
    )
save_records(knowledge_texts, json_path)


#%%% Testing
_, I = search_similar_embeddings(
    query_text='thuỷ phân ester trong môi trường base là phản ứng một chiều',
    index=faiss_index,
    k=2
    )