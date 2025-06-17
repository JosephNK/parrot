"""
Translation RAG Model Module

"""

import faiss
import numpy as np
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer


class TranslationRagModel:
    def __init__(self):
        # RAG 관련 초기화
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.terminology_db = {}
        self.faiss_index = None
        self.term_embeddings = []
        self.term_pairs = []

    def load_terminology_db(self) -> None:
        # 용어 매핑 정의
        self.terminology_db = {
            "kpop": [
                ("포카", "フォトカード"),
                ("앨범", "アルバム"),
                ("콘서트", "コンサート"),
                ("팬미팅", "ファンミーティング"),
                ("사인회", "サイン会"),
                ("굿즈", "グッズ"),
                ("덕질", "推し活"),
                ("최애", "推し"),
                ("본진", "本命"),
                ("부캐", "副垢"),
            ],
            "general": [
                ("안녕", "こんにちは"),
                ("감사합니다", "ありがとうございます"),
                ("미안해", "ごめん"),
                ("사랑해", "愛してる"),
            ],
        }

        self._build_faiss_index()
        print(f"Terminology database loaded: {len(self.term_pairs)} terms")

    def _build_faiss_index(self):
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
            self.faiss_index = faiss.IndexFlatIP(dimension)  # 코사인 유사도

            # 정규화 후 인덱스에 추가
            embeddings_normalized = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            self.faiss_index.add(embeddings_normalized.astype("float32"))

            self.term_embeddings = embeddings_normalized
            self.term_pairs = all_pairs

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
            if len(word) > 2:
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

        return retrieved_terms

    def retrieve_text_with_domain(self, text: str, domain: Optional[str] = None) -> str:
        # 1. 도메인별 관련 용어 검색
        retrieved_terms = self.retrieve_terminology(text, domain=domain)

        # 2. 컨텍스트 구성
        context_text = text
        if retrieved_terms:
            terminology_context = "\n".join(
                [
                    f"{source}: {target} ({term_domain})"
                    for source, target, term_domain, _ in retrieved_terms[:3]
                ]
            )
            context_text = f"용어집:\n{terminology_context}\n\n번역할 텍스트: {text}"

        return context_text
