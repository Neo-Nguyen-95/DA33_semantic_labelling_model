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

#%% HELPERS
def chunk_and_filter_data(knowledge_data, chunk_size=500):
    holder = []
    for data in knowledge_data:
        tokens = data.get('text').split()
        part = 1
        if len(tokens) < 100:
            continue
        if len(tokens) < chunk_size:
            holder.append(data)
            continue
        for i in range(0, len(tokens), chunk_size):
            chunked_token = tokens[i: i+chunk_size]
            chunked_text = " ".join(chunked_token)
            chunked_data = {
                'url': data.get('url'),
                'title': data.get('title') + f' part {part}',
                'text': chunked_text,
                'grade': data.get('grade'),
                'subject': data.get('subject')
                }
            holder.append(chunked_data)
            part += 1
    return holder
    

#%% MAIN
knowledge_data = load_records(knowledge_data_dir)
chunked_knowledge_data = chunk_and_filter_data(knowledge_data)
knowledge_texts = [
    data.get('text') for data in chunked_knowledge_data
    ]

embeddings = embedding_text(knowledge_texts, 200)
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
save_records(chunked_knowledge_data, json_path)


#%%% Testing
_, I = search_similar_embeddings(
    query_text="Khái niệm Khi thay thế nhóm -OH ở nhóm carboxyl (-COOH) của carboxylic acid bằng nhóm −OR' thì được ester. Trong đó R' là gốc hydrocarbon. Ester đơn chức có công thức chung là RCOOR', trong đó R là gốc hydrocarbon hoặc H, R' là gốc hydrocarbo",
    index=faiss_index,
    k=2
    )