from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.langchain_module.rag import VectorStore
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


class LocalQdrant(VectorStore):
    
    def _init_client(self):
        return QdrantClient(host="localhost", port=6333)

collection_name = "pre_screening"
lq = LocalQdrant(collection_name=collection_name)
lq._ensure_collection()

print(lq.get_collection_info())

treat_name_ko = "신경치료"
treat_name_en = "Root Canal Treatment"
treat_time = 70

data = [
    "이가 가만히 있어도 아파요.",
    "어금니가 자꾸 아파서 잠을 못자요.",
    "이가 욱신욱신 쑤시는 것 같아요.",
    "찬물을 마시면 너무 아파요.",
    "뜨거운 차를 마실 때 이가 찌릿해요.",
    "음식을 씹을 때 이가 너무 아파요.",
    "이가 변색되었어요, 다른 치아보다 더 어두워요.",
    "치아에 구멍이 있어요.",
    "이빨이 깨져서 왔어요.",
    "치아가 부러졌어요.",
    "이가 파절됐어요.",
    "잇몸에 고름주머니가 생겼어요.",
    "볼이 부었어요.",
    "뺨 쪽에 여드름 같은 게 생겼어요.",
    "이가 시큰시큰해요.",
    "약을 먹어도 통증이 가라앉지 않아요.",
    "잠자기 전에 이가 더 아파요.",
    "잇몸이 볼록하게 부어올랐어요.",
    "저작할 때 이가 아파요.",
    "달달한 음식을 먹으면 이가 시려요.",
    "이가 깨진 후로 계속 아파요.",
    "이가 불편해서 한쪽으로만 씹어요.",
    "맞은편 이로 씹을 때도 아파요.",
    "아파서 진통제를 계속 먹고 있어요.",
    "이가 저절로 욱신거려요.",
    "찌릿찌릿한 통증이 있어요.",
    "이빨이 아픈 쪽으로 잠을 못자요.",
    "이가 길어진 것 같이 불편해요.",
    "가만히 있어도 이가 쑤셔요.",
    "스케일링 한 후에 더 아파졌어요.",
    "이가 짓누르는 듯이 아파요.",
    "뜨거운 음식을 먹으면 아파요.",
    "이가 흔들리는 느낌이에요.",
    "이가 쑤시는데 약을 먹어도 안 나아요.",
    "치아가 불에 타는 것처럼 아파요.",
    "이가 누르면 아파요.",
    "이가 접촉할 때 아파요.",
    "음식을 씹을 때 불편해요.",
    "뜨거운 물에 이가 반응해요.",
    "예전에 때웠던 이가 아파요.",
    "이가 썩은 것 같아요.",
    "이에 구멍이 생겼어요.",
    "이가 불편해서 음식을 못 먹어요.",
    "밤에 이가 더 아파요.",
    "잇몸에 고름이 차있는 것 같아요.",
    "이가 시큰하고 아파요.",
    "이가 자꾸 쑤시는 느낌이에요.",
    "이가 때때로 찌릿해요.",
    "이가 검게 변했어요.",
    "이가 불편해서 양치질하기 힘들어요.",
    "이를 닦을 때도 아파요.",
    "이가 가만히 있어도 욱신거려요.",
    "이가 뜨거운 것만 대도 아파요.",
    "씹을 때 이가 깨질 것 같아요.",
    "이가 튀어나온 것 같이 느껴져요.",
    "이가 아파서 잠을 못 잤어요.",
    "이빨이 심하게 깨졌어요.",
    "충치가 너무 커진 것 같아요.",
    "이가 깨지고 나서 불편해요.",
    "씹을 때 이가 욱신욱신해요.",
    "이가 계속 아파서 진통제를 먹고 있어요.",
    "이가 쑤시는 느낌이 들어요.",
    "이가 아파서 먹는 것이 불편해요.",
    "뜨거운 것만 닿아도 이가 아파요.",
    "음식물이 이에 끼면 아파요.",
    "이가 단단한 음식 씹을 때 아파요.",
    "아래 치아를 누르면 위 치아가 아파요.",
    "잇몸에 작은 여드름 같은 것이 생겼어요.",
    "이가 다른 색으로 변했어요.",
    "이가 쑤시고 아파요.",
    "한쪽 뺨이 부어올랐어요.",
    "어금니가 아파서 약을 먹어도 안 나아요.",
    "이가 씹을 때마다 아파요.",
    "이가 너무 아파서 누워있기도 힘들어요.",
    "이가 아파서 머리까지 아파요.",
    "이가 깨지고 구멍이 생겼어요.",
    "이가 아파서 볼이 붓고 열이 나요.",
    "치아 주변이 볼록하게 부어올랐어요.",
    "이가 쑤시는 것 같아요.",
    "이가 아파서 입을 벌리기 힘들어요.",
    "이가 저절로 아파요.",
    "이가 두통을 유발해요.",
    "이가 이렇게 심하게 아픈 적이 없어요.",
    "이가 움직이는 것 같이 불편해요.",
    "이가 뜨거운 것에 민감해요.",
    "이가 아파서 밥을 못 먹어요.",
    "씹을 때 이가 불편해요.",
    "이가 아파서 잠을 못 잤어요.",
    "이가 깨지고 나서 계속 아파요.",
    "이가 아파서 일상생활이 힘들어요.",
    "이가 진한 색으로 변했어요.",
    "이가 아파서 약을 달고 살아요.",
    "이가 저절로 찌릿찌릿해요.",
    "이가 아파서 먹는 것이 괴로워요.",
    "이가 아파서 주변 치아도 다 아파요.",
    "뜨거운 음식을 먹으면 견딜 수 없어요.",
    "이가 찌릿하고 통증이 온 뺨으로 퍼져요.",
    "이가 아파서 음식을 씹을 수 없어요.",
    "이가 아파서 약을 달고 살아요.",
    "이가 아파서 밤에 깨요.",
]

for line in data:
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
