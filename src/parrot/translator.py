"""
Korean-Japanese Translator Module

한국어-일본어 번역을 위한 고수준 인터페이스
"""

from typing import Optional, List, Dict, Any

from .model import (
    HyperCLOVAXTranslationModel,
    M2MTranslationModel,
    MBartTranslationModel,
    NLLBTranslationModel,
    QwenTranslationModel,
    VarcoTranslationModel,
)
from .config import config


class KoreanJapaneseTranslator:
    """한국어-일본어 번역기"""

    def __init__(
        self,
        model_name: Optional[str] = None,
        auto_load: bool = True,
    ):
        """
        Args:
            model_name: 사용할 모델 이름 (기본값: NLLB-200)
            auth_token: Hugging Face 인증 토큰
            auto_load: 초기화 시 자동으로 모델 로드
        """
        self.model = None
        self.model_name = model_name
        self.load_model(model_name, auto_load)

    def load_model(
        self,
        model_name: Optional[str] = None,
        auto_load: bool = True,
    ) -> None:
        if model_name is None:
            model_name = config.SUPPORTED_MODELS["nllb-200"]
        elif model_name in config.SUPPORTED_MODELS:
            model_name = config.SUPPORTED_MODELS[model_name]

        # 모델 타입에 따라 적절한 클래스 선택
        if "nllb" in model_name.lower():
            self.model = NLLBTranslationModel(model_name)
        elif "m2m" in model_name.lower():
            self.model = M2MTranslationModel(model_name)
        elif "mbart" in model_name.lower():
            self.model = MBartTranslationModel(model_name)
        elif "hyperclova" in model_name.lower():
            self.model = HyperCLOVAXTranslationModel(model_name)
        elif "qwen" in model_name.lower():
            self.model = QwenTranslationModel(model_name)
        elif "varco" in model_name.lower():
            self.model = VarcoTranslationModel(model_name)

        if auto_load:
            self.model.load_model()

    # def unload_model(self) -> None:
    #     """현재 로드된 모델 언로드"""
    #     if hasattr(self, "model") and self.model is not None:
    #         self.model.unload_model()
    #         del self.model
    #         print("모델이 성공적으로 언로드되었습니다.")
    #     else:
    #         print("현재 로드된 모델이 없습니다.")

    def ko2ja(self, text: str, **kwargs) -> str:
        """
        한국어 → 일본어 번역

        Args:
            text: 번역할 한국어 텍스트
            **kwargs: 추가 번역 옵션

        Returns:
            일본어 번역 결과
        """
        return self.model.translate(
            text=text, source_lang="korean", target_lang="japanese", **kwargs
        )

    def ja2ko(self, text: str, **kwargs) -> str:
        """
        일본어 → 한국어 번역

        Args:
            text: 번역할 일본어 텍스트
            **kwargs: 추가 번역 옵션

        Returns:
            한국어 번역 결과
        """
        return self.model.translate(
            text=text, source_lang="japanese", target_lang="korean", **kwargs
        )

    def translate_batch(
        self, texts: List[str], source_lang: str, target_lang: str, **kwargs
    ) -> List[str]:
        """
        배치 번역

        Args:
            texts: 번역할 텍스트 리스트
            source_lang: 소스 언어
            target_lang: 타겟 언어
            **kwargs: 추가 번역 옵션

        Returns:
            번역 결과 리스트
        """
        results = []
        for text in texts:
            result = self.model.translate(
                text=text, source_lang=source_lang, target_lang=target_lang, **kwargs
            )
            results.append(result)
        return results

    def ko2ja_batch(self, texts: List[str], **kwargs) -> List[str]:
        """한국어 → 일본어 배치 번역"""
        return self.translate_batch(texts, "korean", "japanese", **kwargs)

    def ja2ko_batch(self, texts: List[str], **kwargs) -> List[str]:
        """일본어 → 한국어 배치 번역"""
        return self.translate_batch(texts, "japanese", "korean", **kwargs)

    def interactive_translate(self) -> None:
        """대화형 번역 인터페이스"""
        import sys

        print("=== 한국어-일본어 번역기 ===")
        print("명령어:")
        print("  'ko': 한국어 → 일본어")
        print("  'ja': 일본어 → 한국어")
        print("  'quit': 종료")
        print("-" * 30)

        # 인코딩 설정
        if sys.stdin.encoding != "utf-8":
            print(f"⚠️  터미널 인코딩: {sys.stdin.encoding}")
            print("UTF-8 설정을 권장합니다: export LANG=ko_KR.UTF-8")

        while True:
            try:
                # 안전한 입력 받기
                try:
                    command = input("\n번역 방향 (ko/ja/quit): ").strip().lower()
                except UnicodeDecodeError:
                    print("❌ 인코딩 오류. 영어로 명령어를 입력해주세요.")
                    continue

                if command == "quit":
                    print("번역기를 종료합니다.")
                    break
                elif command not in ["ko", "ja"]:
                    print("올바른 명령어를 입력하세요 (ko/ja/quit)")
                    continue

                # 텍스트 입력 받기
                try:
                    if command == "ko":
                        text = input("한국어 텍스트를 입력하세요: ").strip()
                    else:
                        text = input("일본어 텍스트를 입력하세요: ").strip()

                    if not text:
                        print("텍스트를 입력해주세요.")
                        continue

                    # 인코딩 확인 및 변환
                    try:
                        # UTF-8로 인코딩 테스트
                        text.encode("utf-8")
                    except UnicodeEncodeError:
                        print("❌ 텍스트 인코딩 오류")
                        continue

                except UnicodeDecodeError as e:
                    print(f"❌ 입력 인코딩 오류: {e}")
                    print("터미널 설정을 확인해주세요: export LANG=ko_KR.UTF-8")
                    continue
                except EOFError:
                    print("\n번역기를 종료합니다.")
                    break

                print("번역 중...")
                try:
                    if command == "ko":
                        result = self.ko2ja(text)
                        print(f"한국어: {text}")
                        print(f"일본어: {result}")
                    else:
                        result = self.ja2ko(text)
                        print(f"일본어: {text}")
                        print(f"한국어: {result}")
                except Exception as e:
                    print(f"❌ 번역 오류: {e}")

            except KeyboardInterrupt:
                print("\n번역기를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 예상치 못한 오류: {e}")
                print("다시 시도해주세요.")

    def get_info(self) -> Dict[str, Any]:
        """번역기 정보 반환"""
        model_info = self.model.get_model_info()
        return {
            **model_info,
            "supported_directions": ["Korean → Japanese", "Japanese → Korean"],
            "supported_models": config.SUPPORTED_MODELS,
        }

    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """사용 가능한 모델 목록 반환"""
        return config.SUPPORTED_MODELS.copy()
