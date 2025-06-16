# Parrot Translation Service - Docker Commands

.PHONY: help build run stop clean logs shell test dev prod test-api download-models download-all-models download-nllb download-m2m download-hyperclova

# 기본 설정
IMAGE_NAME := parrot-translation
CONTAINER_NAME := parrot-api
PORT := 8000

help: ## 사용 가능한 명령어 표시
	@echo "Parrot Translation Service - Docker Commands"
	@echo "==========================================="
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Docker 이미지 빌드
	@echo "🔨 Building Docker image..."
	docker build -t $(IMAGE_NAME):latest .

build-no-cache: ## 캐시 없이 Docker 이미지 빌드
	@echo "🔨 Building Docker image without cache..."
	docker build --no-cache -t $(IMAGE_NAME):latest .

run: ## Docker 컨테이너 실행 (프로덕션 모드)
	@echo "🚀 Starting Parrot API in production mode..."
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		--env-file .env \
		-e PYTHONPATH=/app/src \
		-e RADIS_HOST=localhost \
		-v $(PWD)/models_cache:/app/models_cache \
		$(IMAGE_NAME):latest

dev: ## 개발 모드로 실행 (코드 변경 실시간 반영)
	@echo "🚀 Starting Parrot API in development mode..."
	docker run -d \
		--name $(CONTAINER_NAME)-dev \
		-p 8001:8000 \
		--env-file .env \
		-e PYTHONPATH=/app/src \
		-e LOG_LEVEL=DEBUG \
		-e RADIS_HOST=localhost \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/models_cache:/app/models_cache \
		$(IMAGE_NAME):latest \
		python -m uvicorn src.parrot.api.main:app --host 0.0.0.0 --port 8000 --reload

prod: ## 프로덕션 모드로 실행 (멀티 워커)
	@echo "🚀 Starting Parrot API in production mode with multiple workers..."
	docker run -d \
		--name $(CONTAINER_NAME)-prod \
		-p $(PORT):8000 \
		--env-file .env \
		-e PYTHONPATH=/app/src \
		-v $(PWD)/models_cache:/app/models_cache \
		$(IMAGE_NAME):latest \
		python -m uvicorn src.parrot.api.main:app --host 0.0.0.0 --port 8000 --workers 4

compose-up: ## Docker Compose로 프로덕션 서비스 시작
	@echo "🚀 Starting production services with Docker Compose..."
	docker-compose up -d

compose-dev: ## Docker Compose로 개발 모드 시작
	@echo "🚀 Starting development services with Docker Compose..."
	docker-compose --profile dev up -d

compose-down: ## Docker Compose 서비스 중지
	@echo "🛑 Stopping services..."
	-docker stop $$(docker ps -q --filter "name=parrot") 2>/dev/null || true
	docker-compose down --remove-orphans

compose-clean: ## Docker Compose 서비스 완전 정리 (볼륨 포함)
	@echo "🧹 Cleaning up all services and volumes..."
	-docker stop $$(docker ps -q --filter "name=parrot") 2>/dev/null || true
	-docker rm $$(docker ps -aq --filter "name=parrot") 2>/dev/null || true
	docker-compose down --volumes --remove-orphans
	@echo "✅ Cleanup completed"

compose-purge: ## Docker Compose 완전 제거 (이미지 포함)
	@echo "🗑️ Purging all services, volumes, and images..."
	-docker stop $$(docker ps -q --filter "name=parrot") 2>/dev/null || true
	-docker rm $$(docker ps -aq --filter "name=parrot") 2>/dev/null || true
	docker-compose down --volumes --remove-orphans --rmi all
	-docker network prune -f
	@echo "✅ Purge completed"

compose-logs: ## Docker Compose 로그 확인
	docker-compose logs -f

stop: ## 컨테이너 중지
	@echo "🛑 Stopping containers..."
	-docker stop $(CONTAINER_NAME) $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-prod
	-docker rm $(CONTAINER_NAME) $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-prod

clean: ## 이미지와 컨테이너 정리
	@echo "🧹 Cleaning up..."
	-docker stop $(CONTAINER_NAME) $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-prod
	-docker rm $(CONTAINER_NAME) $(CONTAINER_NAME)-dev $(CONTAINER_NAME)-prod
	-docker rmi $(IMAGE_NAME):latest
	docker system prune -f

logs: ## 컨테이너 로그 확인
	docker logs -f $(CONTAINER_NAME)

shell: ## 컨테이너 내부 쉘 접속
	docker exec -it $(CONTAINER_NAME) /bin/bash

test: ## 컨테이너 내에서 테스트 실행
	docker exec -it $(CONTAINER_NAME) python -m pytest

health: ## 헬스체크 확인
	@echo "🏥 Checking health..."
	curl -f http://localhost:$(PORT)/health || echo "Service is not healthy"

test-api: ## API 테스트 (번역 요청)
	@echo "🧪 Testing translation API..."
	@echo "Korean to Japanese translation test:"
	curl -G "http://localhost:8000/translate/ko2ja" \
		--data-urlencode "text=안녕하세요. 오늘 날씨가 정말 좋네요" \
		--data-urlencode "model=m2m-100-1.2b" | jq
	@echo "\nHealth check:"
	curl -f http://localhost:8000/health | jq

download-models: ## 모델 다운로드 (사용법: make download-models MODEL=nllb-200)
	@echo "📥 Downloading models..."
	@if [ -z "$(MODEL)" ]; then \
		echo "모델명을 지정해주세요. 예: make download-models MODEL=nllb-200"; \
		echo "사용 가능한 모델:"; \
		echo "  - nllb-200"; \
		echo "  - m2m-100-1.2b"; \
		echo "  - hyperclova-1.5b"; \
	else \
		docker exec -it $(CONTAINER_NAME) python scripts/download_models.py --model $(MODEL); \
	fi

status: ## 컨테이너 상태 확인
	@echo "📊 Container status:"
	docker ps -a --filter name=$(CONTAINER_NAME)