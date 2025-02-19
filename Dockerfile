# 첫 번째 스테이지: requirements-stage
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 시스템 패키지 설치 및 캐시 정리
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /tmp

# pyproject.toml과 poetry.lock 복사
COPY pyproject.toml poetry.lock* /tmp/

# Poetry 설치
RUN pip install poetry && poetry lock && poetry self add poetry-plugin-export

# requirements.txt 생성
RUN poetry export -f requirements.txt --without-hashes --output requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# 두 번째 스테이지: 최종 이미지
FROM python:3.12-slim AS runtime

ARG NEO4J_URL
ARG NEO4J_USER
ARG NEO4J_PASSWORD
ENV NEO4J_URL=${NEO4J_URL}
ENV NEO4J_USER=${NEO4J_USER}
ENV NEO4J_PASSWORD=${NEO4J_PASSWORD}

# 작업 디렉토리 설정
WORKDIR /app

# builder 패키지 복사 및 의존성 설치
COPY --from=builder /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# 애플리케이션 코드 복사
WORKDIR  /app
COPY . .

# 헬스체크 설정
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8504/health || exit 1

# 포트 설정
EXPOSE 8504
CMD ["serve", "run", "app.main:build_app"]
