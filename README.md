# Dencomm Medical Inquiry API

의료 상담 및 후속 관리를 위한 API 서비스입니다.

## 주요 기능

- 의료 상담 문의 처리
- 후속 관리 상담 처리
- 하이브리드 검색 (벡터 검색 + 그래프 검색)
- 적응형 RAG (Retrieval Augmented Generation)

## 시스템 구성

- FastAPI 기반 REST API 서버
- Ray Serve를 통한 서비스 배포 및 스케일링
- LangChain을 활용한 LLM 파이프라인 구성
- Qdrant 벡터 데이터베이스
- Neo4j 그래프 데이터베이스

## 주요 컴포넌트

### APIIngress
- 메인 API 서버
- CORS 미들웨어 설정
- 헬스체크 엔드포인트
- 자동 스케일링 설정 (1-3 replicas)

### MedicalInquiryService
- 의료 상담 처리 서비스
- 적응형 RAG 구현
  - 멀티쿼리 생성
  - BM25 + 벡터 검색 앙상블
  - 컨텍스트 압축 및 재순위화

### FollowupCareService
- 후속 관리 상담 서비스
- 하이브리드 검색 구현
  - 벡터 검색과 그래프 검색 결과 통합
  - Neo4j 기반 지식 그래프 활용

## 설치 및 실행

1. 필요 패키지 설치
   ```bash
   # 파이썬 패키지 설치
   pip install poetry
   poetry ilock && poetry install

   # qdrant vector db 설치 및 실행
   sudo docker pull qdrant/qdrant
   sudo docker network create qdrant_server
   sudo docker run -d -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    --network qdrant_server \
    --name qdrant_vdb \
    qdrant/qdrant
   ```

2. FastAPI 서버 실행
   ```bash
   poetry run serve app.main:build_app
   ```

3. Docker로 실행
   ```bash
   sudo docker compose build
   sudo docker compose up
   ```

4. API 문서 확인
   - Swagger UI: [http://localhost:8504/docs](http://localhost:8504/docs)
   - ReDoc: [http://localhost:8504/redoc](http://localhost:8504/redoc)

