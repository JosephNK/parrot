"""
Translation RAG Model Module

"""

import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer

Faiss = None


def lazyFaiss():
    global Faiss
    if Faiss is None:
        import faiss

        Faiss = faiss
        print("✓ FAISS loaded successfully.")
    return Faiss


class TranslationRagModel:
    def __init__(self):
        # RAG 관련 초기화
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("✓ Embedding Model loaded successfully.")
        self.terminology_db = {}
        self.faiss_index = None
        self.term_embeddings = []
        self.term_pairs = []

    def load_terminology_db(self) -> None:
        # 용어 매핑 정의
        self.terminology_db = {
            "ko2ja": [
                ("포카", "フォトカード"),
                ("굿즈", "グッズ"),
                ("덕질", "推し活"),
                ("부캐", "副垢"),
            ],
            "ja2ko": [
                ("フォトカード", "포카"),
                ("グッズ", "굿즈"),
                ("推し活", "덕질"),
                ("副垢", "부캐"),
            ],
            "ko2ko": [
                ("포카", "포토카드"),
                ("굿즈", "굿즈"),
                ("덕질", "오타쿠"),
                ("부캐", "부 캐릭터"),
            ],
        }
        self.build_index()

    def get_domain_from_lang(
        self,
        source_lang: str,
        target_lang: str,
        use_replacement: bool = False,
    ) -> str:
        if source_lang == "korean" and target_lang == "japanese":
            return "ko2ja" if not use_replacement else "ko2ko"
        elif source_lang == "japanese" and target_lang == "korean":
            return "ja2ko" if not use_replacement else "ja2ja"
        else:
            return "ko2ja"

    def build_index(self):
        """FAISS 인덱스 구축"""
        all_terms = []
        all_pairs = []

        for domain, term_pairs in self.terminology_db.items():
            for source_term, target_term in term_pairs:
                all_terms.append(source_term)
                all_pairs.append((source_term, target_term, domain))

        if all_terms:
            # 임베딩 생성
            embeddings = self.embedding_model.encode(all_terms)

            # FAISS 인덱스 구축
            dimension = embeddings.shape[1]
            self.faiss_index = lazyFaiss().IndexFlatIP(dimension)  # 코사인 유사도

            # 정규화 후 인덱스에 추가
            embeddings_normalized = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            self.faiss_index.add(embeddings_normalized.astype("float32"))

            self.term_embeddings = embeddings_normalized
            self.term_pairs = all_pairs

        print(f"✓ Terminology database loaded: {len(self.term_pairs)} terms")

    def retrieve_terminology(
        self,
        text: str,
        domain: Optional[str] = None,
        k: int = 5,
        threshold: float = 0.7,
    ) -> List[Tuple[str, str, str, float]]:
        """관련 용어 검색 - 도메인 필터링 포함"""
        if not self.faiss_index:
            return []

        words = text.split()
        retrieved_terms = []

        for word in words:
            if len(word) >= 2:
                query_embedding = self.embedding_model.encode([word])
                query_normalized = query_embedding / np.linalg.norm(query_embedding)

                scores, indices = self.faiss_index.search(
                    query_normalized.astype("float32"), k
                )

                for score, idx in zip(scores[0], indices[0]):
                    if score > threshold and idx < len(self.term_pairs):
                        source_term, target_term, term_domain = self.term_pairs[idx]
                        # 도메인 필터링 적용
                        if domain is None or term_domain == domain:
                            retrieved_terms.append(
                                (source_term, target_term, term_domain, score)
                            )

        # 중복 제거 (highest score만 유지)
        unique_terms = {}
        for source_term, target_term, term_domain, score in retrieved_terms:
            key = (source_term, target_term, term_domain)
            if key not in unique_terms or unique_terms[key][3] < score:
                unique_terms[key] = (source_term, target_term, term_domain, score)

        # 점수 순으로 정렬
        return sorted(unique_terms.values(), key=lambda x: x[3], reverse=True)

    def retrieve_replace_text_with_domain(
        self,
        text: str,
        domain: Optional[str] = None,
    ) -> str:
        # 1. 도메인별 관련 용어 검색
        retrieved_terms = self.retrieve_terminology(text, domain=domain)

        # 원문에서 특수 용어를 일반적인 단어로 교체
        preprocessed_text = text
        for source_term, target_term, _, _ in retrieved_terms:
            if source_term in preprocessed_text:
                # 임시로 일반적인 단어로 교체 (번역이 잘 되도록)
                preprocessed_text = preprocessed_text.replace(
                    source_term, f"{target_term}"
                )
        return preprocessed_text

    def retrieve_text_with_domain(
        self,
        text: str,
        domain: Optional[str] = None,
        max_terms: int = 3,  # 🔧 조정 가능하게
    ) -> str:
        # 1. 도메인별 관련 용어 검색
        retrieved_terms = self.retrieve_terminology(text, domain=domain)

        if not retrieved_terms:
            return text  # 용어가 없으면 원본 반환

        # 더 자연스러운 프롬프트 구성
        terminology_context = ", ".join(
            [
                f"'{source}' → '{target}'"
                for source, target, _, _ in retrieved_terms[:max_terms]
            ]
        )

        context_text = f"""{terminology_context}"""

        return context_text
