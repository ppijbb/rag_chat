from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.langchain_module.rag import VectorStore

class LocalQdrant(VectorStore):
    
    def _init_client(self):
        return QdrantClient(host="localhost", port=6333)

   
lq = LocalQdrant(collection_name="state_control")
# lq.delete_collection()
# lq._ensure_collection()

# print(lq.get_collection_info())


# lq.add_text(
#     text="예상되는 진료는  이며, 진료 시간은 분으로 예상됩니다.", 
#     collection_name="state_control", 
#     metadata={
#         "state": 3,
#         "category_ko": "요약",
#         "category_en": "Summary"
#         }
#     )
# lq.add_text(
#     text="Expected treatment is  Treatment, and the estimated treatment time is  minutes.", 
#     collection_name="state_control", 
#     metadata={
#         "state": 3,
#         "category_ko": "요약",
#         "category_en": "Summary"
#         }
#     )

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

# lq.add_text(
#     text="What symptoms do you have?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "증상",
#         "category_en": "Symptoms"
#         }
#     )
# lq.add_text(
#     text="How long have the symptoms lasted?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "증상",
#         "category_en": "Durations"
#         }
#     )
# lq.add_text(
#     text="Can you tell me where the symptoms are located?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 1,
#         "category_ko": "증상",
#         "category_en": "Symptoms Area"
#         }
#     )
# lq.add_text(
#     text="How severe is the intensity of the symptoms?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "증상",
#         "category_en": "Severity"
#         }
#     )
# lq.add_text(
#     text="In what situations do the symptoms worsen?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "증상 유발요인",
#         "category_en": "Specific Situations"
#         }
#     )
# lq.add_text(
#     text="Is there anything you would like to say?", 
#     collection_name="state_control", 
#     metadata={
#         "state": 2,
#         "category_ko": "하고 싶은 말",
#         "category_en": "Special Considerations"
#         }
#     )
search_query = "예상 진료는 신경치료 입니다."

print(lq.search(
    query=search_query, 
    collection_name="state_control",
    limit=1))

search_query = lq.embeddings.embed_query(search_query)
print(lq.client.query_points(
    query=search_query, 
    collection_name="state_control",
    limit=1))
