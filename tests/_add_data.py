from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.langchain_module.rag import VectorStore
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings('ignore', category=DeprecationWarning)


class LocalQdrant(VectorStore):
    
    def _init_client(self):
        return QdrantClient(host="localhost", port=6333)

collection_name = "pre_screening"
lq = LocalQdrant(collection_name=collection_name)
lq._ensure_collection()

print(lq.get_collection_info())

treat_name_ko = "치아 발거"
treat_name_en = "Tooth Extraction"
treat_time = 40
symptoms = "극심한 치통"
wrong_keyword = "치아 발거거거"

data = [
"신경 치료를 받았는데 계속 붓고 아파요. 염증이 안 없어진대요.",
"이가 거의 썩어서 남아 있는 부분이 거의 없어요. 치료할 방법이 없다고 들었어요.",
"치아가 너무 흔들려서 씹을 때마다 불안해요. 잇몸뼈가 거의 없다고 들었어요.",
"어금니가 씹을 때마다 찌릿하게 아픈데, 치아 뿌리가 깨졌다고 들었어요.",
"사랑니가 누워 있어서 자꾸 잇몸이 붓고 아파요. 옆 치아도 밀린대요.",
"이가 반쯤 부러졌는데, 잇몸까지 내려가서 붙일 수 없다고 했어요.",
"치아가 너무 겹쳐서 교정을 하려면 몇 개 빼야 한다고 했어요.",
"넘어지면서 치아를 부딪쳤는데, 뿌리까지 문제가 있다고 해요.",
"치아가 너무 약해져서 틀니를 하려면 몇 개는 빼야 한대요.",
]

for line in tqdm(data):
    searched = lq.search(
        query=line,
        collection_name=collection_name,
        limit=3)
    in_db = [s for s in searched if line in s['text']]
    if any(in_db):
        print(line, "already exists in vdb")
        if in_db[0]['metadata']["치료_ko"] == wrong_keyword:
            lq.delete_item(
                item_id=in_db[0]["_id"])
        else:
            pass
    else:
        lq.add_text(
            text=line,
            collection_name=collection_name,
            metadata={
                "치료_ko": treat_name_ko,
                "치료_en": treat_name_en,
                "증상": symptoms,
                "답변 예시": f"예상되는 진료는 {treat_name_ko} 이며, 진료 시간은 {treat_time}분 으로 예상됩니다.",
                "병원": "덴컴병원"
                }
            )

search_query = data[-1]
search_query = lq.embeddings.embed_query(search_query)
print(lq.client.query_points(
    query=search_query, 
    collection_name="pre_screening",
    limit=1))
print("Done")
