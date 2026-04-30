#%% LIB
from da33_labelling_project.config import (
    PROCESSED_DATA_DIR
    )
from da33_labelling_project.embedding_process import (
    embedding_text,
    load_embedding_store,
    search_similar_embeddings
    )
from da33_labelling_project.utils import (
    load_records
    )

#%% CONFIG
embedding_dir = (
    PROCESSED_DATA_DIR /
    'high_school_knowledge_embeddings'
    )

faiss_path = embedding_dir / 'high_school_knowledge.faiss'

#%% MAIN
knowledge_data = load_records(embedding_dir)
faiss_index = load_embedding_store(faiss_path)

#%%% Testing
D, I = search_similar_embeddings(
    query_text="Các pin năng lượng Mặt trời có nhiều ứng dụng trong thực tế. Chúng đặc biệt thích hợp cho các vùng mà điện lưới khó vươn tới như núi cao, ngoài đảo xa, hoặc phục vụ các hoạt động trên không gian; cụ thể như các vệ tinh quay xung quanh quỹ đạo trái đất, máy tính cầm tay, các máy điện thoại cầm tay từ xa, thiết bị bơm nước… Pin năng lượng mặt trời là gì? Hoạt động như thế nào? Pin năng lượng mặt trời (pin mặt trời/pin quang điện) là thiết bị giúp chuyển hóa trực tiếp năng lượng ánh sáng mặt trời (quang năng) thành năng lượng điện (điện năng) dựa trên hiệu ứng quang điện. Hiệu ứng quang điện là khả năng phát ra điện tử (electron) khi được ánh sáng chiếu vào của vật chất. Tấm pin mặt trời, những tấm có bề mặt lớn thu thập ánh nắng mặt trời và biến nó thành điện năng, được làm bằng nhiều tế bào quang điện có nhiệm vụ thực hiện quá trình tạo ra điện từ ánh sáng mặt trời.",
    index=faiss_index,
    k=3
    )

relevant_knowledge_data = [
    knowledge_data[i] for i in I[0]
    ]