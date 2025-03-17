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

treat_name_ko = "충치치료"
treat_name_en = "Root Canal Treatment"
treat_time = 70

data = [
"이가 찬물에 좀 시려요.",
"차가운 음료 마실 때 이가 시려요.",
"아이스크림 먹을 때 이가 불편해요.",
"단 음식 먹으면 이가 시큰해요.",
"이에 검은 점이 생긴 것 같아요.",
"이가 가끔씩 시린 느낌이 들어요.",
"이에 작은 구멍이 있는 것 같아요.",
"치아에 음식물이 자꾸 껴요.",
"이가 갈색으로 변한 부분이 있어요.",
"이가 가끔 아팠다가 안 아팠다가 해요.",
"단 음식 먹을 때 이가 불편해요.",
"이가 가끔 찌릿하고 시려요.",
"이 사이에 음식이 계속 끼어요.",
"어금니에 까만 점이 생겼어요.",
"국가구강검진할 때 충치치료가 필요하대요",
"이를 칫솔로 닦을 때 시려요.",
"충치인지 확인받고 싶어요.",
"이에 작은 충치가 생긴 것 같아요.",
"이가 검은색으로 변했어요.",
"이에 구멍이 생겼어요.",
"이가 약간 거칠게 느껴져요.",
"혀로 만지면 이가 울퉁불퉁해요.",
"이가 시린 증상이 있어요.",
"단것 먹을 때 이가 아파요, 그러다 금방 괜찮아져요.",
"음식물이 이 사이에 자꾸 끼어요.",
"이가 약간 아프지만 견딜 만해요.",
"이가 자꾸 시린 느낌이 들어요.",
"치실할 때 이 사이가 아파요.",
"치석 제거하는데 이 부분이 좀 아팠어요.",
"이가 조금 시려서 왔어요.",
"이가 외부 자극에만 반응해요.",
"이 표면이 거칠어진 것 같아요.",
"충치 검진 받고 싶어요.",
"이에 작은 금이 간 것 같아요.",
"찬 음식 먹으면 이가 아파요, 그래도 금방 괜찮아져요.",
"이가 시리지만 약 먹을 정도는 아니에요.",
"이가 가끔씩 아파요.",
"이에 검은 줄이 생겼어요.",
"이 사이가 불편해요.",
"이가 약간 아픈데 신경치료까진 아닌 것 같아요.",
"이가 뜨겁거나 차가울 때만 반응해요.",
"이가 색이 변한 것 같아요.",
"이에 음식이 들어가서 불편해요.",
"이에 하얀 반점이 생겼어요.",
"충치 치료 받으러 왔어요.",
"이가 예전보다 시린 것 같아요.",
"이가 가끔 욱씬거려요.",
"이가 간간히 통증이 있어요.",
"찬물 마시면 이가 약간 아파요.",
"이가 조금 불편해요.",
"이가 음식 씹을 때만 아파요.",
"이가 살짝 시려요.",
"이 색깔이 다른 치아랑 달라요.",
"찬 공기에 이가 시려요.",
"이 표면이 매끄럽지 않아요.",
"이에 작은 홈이 생긴 것 같아요.",
"이가 시리지만 참을 만해요.",
"이가 약간 쑤시는 느낌이 들어요.",
"이가 간헐적으로 시려요.",
"이 표면이 거칠어요.",
"이 사이로 음식이 들어가요.",
"이가 부분적으로 색이 변했어요.",
"이가 살짝 욱씬거려요.",
"찬 음식 먹을 때만 이가 시려요.",
"이가 아프다가 안 아프다가 해요.",
"이에 작은 구멍이 생겼어요.",
"이가 가끔 통증이 있어요.",
"단 음료 마시면 이가 시려요.",
"이에 작은 충치가 있어요.",
"이가 시린 것 같아 검진받으러 왔어요.",
"이가 음식물이 닿으면 아파요.",
"찬바람 쐬면 이가 시려요.",
"이 색이 살짝 어두워진 것 같아요.",
"이에 음식이 끼어요.",
"이가 시원한 것에 민감해요.",
"이가 가끔 시리고 불편해요.",
"이가 조금 아프고 시려요.",
"이가 약간 시큰해요.",
"이에 갈색 반점이 있어요.",
"이가 간혹 시린 느낌이 있어요.",
"치실을 사용하면 이 사이가 불편해요.",
"이가 찬물에 반응해요.",
"이가 충치인 것 같아 왔어요.",
"음식이 이 사이에 끼는 느낌이에요.",
"이가 날카로운 느낌이 있어요.",
"이가 약간 아프지만 견딜 만해요.",
"이에 조그만 구멍이 있어요.",
"이가 음식 씹을 때 살짝 아파요.",
"이가 시려서 치과 왔어요.",
"이에 검은 점이 있어요.",
"이가 시리지만 진통제는 필요 없어요.",
"이가 가끔 시큰시큰해요.",
"이가 차가운 것에 예민해졌어요.",
"이에 작은 틈이 생긴 것 같아요.",
"이가 종종 시려요.",
"이가 외부 자극에 반응해요.",
"이가 자주는 아니고 가끔 시려요.",
"이에 작은 충치가 있는 것 같아요.",
"이가 무언가 닿으면 시려요.",
"이가 시려서 충치인지 확인하고 싶어요.",
]

for line in tqdm(data):
    searched = lq.search(
        query=line,
        collection_name=collection_name,
        limit=3)
    if any([s for s in searched if line in s['text']]):
        print(line, "already exists in vdb")
        pass
    else:
        lq.add_text(
            text=line,
            collection_name=collection_name,
            metadata={
                "치료_ko": treat_name_ko,
                "치료_en": treat_name_en,
                "증상": "치통",
                "답변 예시": f"예상되는 진료는 {treat_name_ko} 이며, 진료 시간은 {treat_time}분 으로 예상됩니다.",
                "병원": "덴컴병원"
                }
            )

search_query = "이가 가만히 있어도 아파요."
search_query = lq.embeddings.embed_query(search_query)
print(lq.client.query_points(
    query=search_query, 
    collection_name="pre_screening",
    limit=1))
