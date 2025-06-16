import redis
import json
import hashlib
import time
from typing import Optional, Dict, Any


class TranslationCache:
    def __init__(self, host="localhost", port=6379, db=0, expire_time=86400):
        """
        번역 캐시 시스템 초기화

        Args:
            host: Redis 서버 호스트
            port: Redis 서버 포트
            db: Redis 데이터베이스 번호
            expire_time: 캐시 만료 시간 (초 단위, 기본 24시간)
        """
        self.redis_client = redis.Redis(
            host=host, port=port, db=db, decode_responses=True
        )
        self.expire_time = expire_time

        # 연결 테스트
        try:
            self.redis_client.ping()
            print("✅ Redis 연결 성공!")
        except redis.ConnectionError:
            print("❌ Redis 연결 실패!")
            raise

    def _generate_cache_key(self, original_text: str) -> str:
        """
        원본 텍스트를 기반으로 캐시 키 생성

        Args:
            original_text: 원본 텍스트

        Returns:
            캐시 키 문자열
        """
        # 텍스트를 해시화하여 키 생성 (긴 텍스트도 고정 길이로)
        text_hash = hashlib.md5(original_text.encode("utf-8")).hexdigest()
        return f"translation:{text_hash}"

    def get_translation(self, original_text: str) -> Optional[Dict[str, Any]]:
        """
        캐시에서 번역 결과 조회

        Args:
            original_text: 원본 텍스트

        Returns:
            번역 데이터 딕셔너리 (없으면 None)
        """
        cache_key = self._generate_cache_key(original_text)

        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                print(f"🔍 캐시에서 발견: {original_text[:30]}...")
                return json.loads(cached_data)
            else:
                print(f"❌ 캐시에 없음: {original_text[:30]}...")
                return None
        except Exception as e:
            print(f"❌ 캐시 조회 오류: {e}")
            return None

    def save_translation(
        self, original_text: str, translated_text: str, translate_time: str
    ) -> bool:
        """
        번역 결과를 캐시에 저장

        Args:
            original_text: 원본 텍스트
            translated_text: 번역된 텍스트
            translate_time: 번역 소요 시간

        Returns:
            저장 성공 여부
        """
        cache_key = self._generate_cache_key(original_text)

        translation_data = {
            "original": original_text,
            "translated": translated_text,
            "translate_time": translate_time,
            "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        try:
            # JSON으로 직렬화하여 저장
            self.redis_client.setex(
                cache_key,
                self.expire_time,
                json.dumps(translation_data, ensure_ascii=False),
            )
            print(f"💾 캐시 저장 완료: {original_text[:30]}...")
            return True
        except Exception as e:
            print(f"❌ 캐시 저장 오류: {e}")
            return False

    def get_or_save_translation(
        self, original_text: str, translated_text: str, translate_time: str
    ) -> Dict[str, Any]:
        """
        캐시에서 번역 결과를 가져오거나, 없으면 저장 후 반환

        Args:
            original_text: 원본 텍스트
            translated_text: 번역된 텍스트
            translate_time: 번역 소요 시간

        Returns:
            번역 데이터 딕셔너리
        """
        # 먼저 캐시에서 조회
        cached_result = self.get_translation(original_text)
        if cached_result:
            return cached_result

        # 캐시에 없으면 저장
        self.save_translation(original_text, translated_text, translate_time)

        return {
            "original": original_text,
            "translated": translated_text,
            "translate_time": translate_time,
        }

    def delete_translation(self, original_text: str) -> bool:
        """
        캐시에서 번역 결과 삭제

        Args:
            original_text: 원본 텍스트

        Returns:
            삭제 성공 여부
        """
        cache_key = self._generate_cache_key(original_text)
        try:
            result = self.redis_client.delete(cache_key)
            if result:
                print(f"🗑️ 캐시 삭제 완료: {original_text[:30]}...")
            return bool(result)
        except Exception as e:
            print(f"❌ 캐시 삭제 오류: {e}")
            return False

    def get_cache_info(self, original_text: str) -> Dict[str, Any]:
        """
        캐시 정보 조회 (TTL 등)

        Args:
            original_text: 원본 텍스트

        Returns:
            캐시 정보 딕셔너리
        """
        cache_key = self._generate_cache_key(original_text)

        try:
            exists = self.redis_client.exists(cache_key)
            ttl = self.redis_client.ttl(cache_key)

            return {
                "exists": bool(exists),
                "ttl_seconds": ttl,
                "ttl_hours": round(ttl / 3600, 2) if ttl > 0 else 0,
                "cache_key": cache_key,
            }
        except Exception as e:
            print(f"❌ 캐시 정보 조회 오류: {e}")
            return {"exists": False, "error": str(e)}

    def clear_all_cache(self) -> int:
        """
        모든 번역 캐시 삭제

        Returns:
            삭제된 키 개수
        """
        try:
            keys = self.redis_client.keys("translation:*")
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                print(f"🗑️ {deleted_count}개 캐시 삭제 완료")
                return deleted_count
            return 0
        except Exception as e:
            print(f"❌ 전체 캐시 삭제 오류: {e}")
            return 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 정보 조회

        Returns:
            캐시 통계 딕셔너리
        """
        try:
            keys = self.redis_client.keys("translation:*")
            total_count = len(keys)

            # 메모리 사용량 (대략적)
            memory_usage = 0
            for key in keys[:10]:  # 샘플링
                try:
                    memory_usage += self.redis_client.memory_usage(key) or 0
                except:
                    pass

            return {
                "total_cached_translations": total_count,
                "estimated_memory_bytes": (
                    memory_usage * (total_count / min(10, total_count))
                    if total_count > 0
                    else 0
                ),
                "cache_keys_sample": keys[:5],
            }
        except Exception as e:
            print(f"❌ 캐시 통계 조회 오류: {e}")
            return {"error": str(e)}
