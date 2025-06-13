```
poetry run python scripts/download_models.py --model hyperclova-0.5b

poetry run python scripts/main.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model hyperclova-0.5b

poetry run python scripts/main.py --info
```