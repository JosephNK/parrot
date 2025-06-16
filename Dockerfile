# 멀티스테이지 빌드를 사용하여 이미지 크기 최적화
FROM python:3.11-slim AS builder

# Poetry 설치
RUN pip install poetry

# Poetry 설정 - 가상환경을 생성하지 않도록 설정
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=0 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 파일들 복사
COPY pyproject.toml poetry.lock ./

# 의존성만 시스템에 직접 설치 (가상환경 사용 안함)
RUN poetry config virtualenvs.create false && \
    poetry install --no-root && \
    rm -rf $POETRY_CACHE_DIR

# 개발 이미지 (builder 기반으로 개발 도구 모두 포함)
FROM builder AS development

# 시스템 패키지 설치 (개발용)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 비root 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 애플리케이션 코드 복사
COPY src/ ./src/
COPY scripts/ ./scripts/

# 모델 다운로드를 위한 디렉토리 생성
RUN mkdir -p /app/models_cache /app/cache && chown -R appuser:appuser /app

# 사용자 변경
USER appuser

# 포트 노출
EXPOSE 8000

# 개발 모드 명령어 (코드 변경 시 자동 재시작)
CMD ["python", "-m", "uvicorn", "src.parrot.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# 프로덕션 이미지
FROM python:3.11-slim AS production

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 비root 사용자 생성
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 작업 디렉토리 설정
WORKDIR /app

# builder 스테이지에서 설치된 Python 패키지들 복사
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# 애플리케이션 코드 복사
COPY src/ ./src/
COPY scripts/ ./scripts/

# 모델 다운로드를 위한 디렉토리 생성
RUN mkdir -p /app/models_cache /app/cache && chown -R appuser:appuser /app

# 사용자 변경
USER appuser

# 포트 노출 (API 서버용)
EXPOSE 8000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 기본 명령어 (프로덕션 모드로 API 서버 실행)
CMD ["python", "-m", "uvicorn", "src.parrot.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]