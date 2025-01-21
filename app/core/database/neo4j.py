import os
import tempfile

from neo4j import GraphDatabase
from pyvis.network import Network
from typing import List, Dict, Optional, Any

NEO4J_URL = os.getenv("NEO4J_URL")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))


def visualize_neo4j_graph():
    edge_labels={
        "HAS_EXAMINATION": "검진",
        "HAS_DIET_LIFE": "식이, 생활",
        "HAS_MEDICATION": "약물",
        "HAS_PREVENTION": "예방",
        "HAS_EXERCISE": "운동",
        "HAS_CAUSE": "원인",
        "HAS_REHABILITATION": "재활",
        "HAS_DEFINITION": "정의",
        "HAS_SYMPTOM": "증상",
        "HAS_DIAGNOSIS": "진단",
        "HAS_TREATMENT": "치료",
        "HAS_DISEASE": "질병",
        "HAS_DEPARTMENT": "진료과목",
        "HAS_CAUTION": "주의사항",
        }
     # Neo4j 연결 설정 (환경변수나 st.secrets에서 가져오는 것을 권장)
    driver = GraphDatabase.driver(
        uri=NEO4J_URL,
        auth=NEO4J_AUTH)  # 실제 인증정보로 변경 필요  
    # Pyvis 네트워크 객체 생성
    net = Network(
        height="400px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black")    
    with driver.session() as session:
        # 노드와 관계 쿼리 (예시 쿼리 - 실제 데이터 구조에 맞게 수정 필요)
        result = session.run("""
            MATCH (n:Department {name:"치과질환"})-[r]->(m)-[s:HAS_CAUTION]->(o)
            RETURN n, r, m, s, o LIMIT 500
        """)   
        # 노드와 엣지 추가
        nodes = set()
        for record in result:
            # 시작 노드 추가
            if record["n"].id not in nodes:
                node_type = "질병" if "disease_name" in record["n"].keys() else "기타"
                depth = 0  # Assuming initial depth for starting nodes
                color = "lightgreen" if node_type == "질병" else "lightcoral"
                net.add_node(
                    record["n"].id,
                    label=str(record["n"].get("name", "")),
                    title=str(record["n"].get("name", "")),
                    color=color
                )
                nodes.add(record["n"].id)  
            # 끝 노드 추가
            if record["m"].id not in nodes:
                depth = 1 # Assuming depth 1 for content nodes
                if "name" in record['m']:
                    label = record['m'].get("name")    
                    color = "lightblue"
                else:
                    label=f'{record["m"].get("written_by", "")} 콘텐츠'
                    color="skyblue"
                net.add_node(
                    record["m"].id,
                    label=label,
                    title=str(record["m"].get("intro", "")),
                    color=color
                )
                nodes.add(record["m"].id)  
            # 관계 추가
            net.add_edge(
                source=record["n"].id,
                to=record["m"].id,
                label=edge_labels[record["r"].type])   
            # 끝 노드 추가
            if record["o"].id not in nodes:
                depth = 1 # Assuming depth 1 for content nodes
                if "name" in record['o']:
                    label = record['o'].get("name")    
                    color = "lightblue"
                else:
                    label=f'{record["o"].get("written_by", "")} 콘텐츠'
                    color="skyblue"
                net.add_node(
                    record["o"].id,
                    label=label,
                    title=str(record["o"].get("intro", "")),
                    color=color
                )
                nodes.add(record["o"].id)   
            net.add_edge(
                source=record["m"].id,
                to=record["o"].id,
                label=edge_labels[record["s"].type])   
    # 임시 HTML 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)  
    # HTML 파일 읽기
    with open(tmp_file.name, 'r', encoding='utf-8') as f:
        html_data = f.read()   
    # 임시 파일 삭제
    os.unlink(tmp_file.name)   
    # Streamlit에 표시

    driver.close()
