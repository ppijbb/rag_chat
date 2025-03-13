from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.langchain_module.rag import VectorStore

class LocalQdrant(VectorStore):
    
    def _init_client(self):
        return QdrantClient(host="localhost", port=6333)

   
lq = LocalQdrant(collection_name="state_control")

print(lq.get_collection_info())

print(lq.search(
    query="얼마나 아프신가요?", 
    collection_name="state_control",
    limit=1))

# lq.add_text(
#     text="어떤 증상이 있으신가요?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "증상",
#         "category_en": "Symptoms"
#         }
#     )
# lq.add_text(
#     text="증상이 얼마나 지속되었나요?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "증상",
#         "category_en": "Durations"
#         }
#     )
# lq.add_text(
#     text="증상 부위가 어디인지 말씀해 주실 수 있나요?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 1,
#         "category_ko": "증상",
#         "category_en": "Symptoms Area"
#         }
#     )
# lq.add_text(
#     text="증상의 강도가 얼마나 심하신가요?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "증상",
#         "category_en": "Severity"
#         }
#     )
# lq.add_text(
#     text="어떤 상황에서 증상이 더 심해지시나요?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "증상 유발요인",
#         "category_en": "Specific Situations"
#         }
#     )
# lq.add_text(
#     text="하고 싶은 말이 있으신가요?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "하고 싶은 말",
#         "category_en": "Special Considerations"
#         }
#     )


